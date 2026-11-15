"""
Data loading utilities for behavior classification.
Supports multi-video training with video-level validation split.
Per-video reference patch configuration.
Efficient caching: extracts DINOv3 features per video once, caches to disk.
"""

import os
import json
import yaml
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import cv2
import hashlib
import pickle

from utils.dinov3_extractor import (
    DINOv3Extractor,
    select_similar_patches,
    pad_patch_sequence,
    create_sliding_windows
)

from utils.data_config import VideoConfig, DataConfig


# Global cache for processed per-frame embeddings (avoids pickling issues with DataLoader workers)
_PROCESSED_FRAMES_CACHE = {}


def _get_video_cache_key(video_path: str, sample_rate: int, dinov3_model: str) -> str:
    """Generate a unique cache key for a video's raw features."""
    key_str = f"{video_path}_{sample_rate}_{dinov3_model}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_processed_cache_key(video_path: str, sample_rate: int, dinov3_model: str,
                             ref_patch: Tuple[int, int], similarity_threshold: float,
                             border_size: int, window_size: int, stride: int,
                             max_patches_per_frame: int) -> str:
    """Generate a unique cache key for fully processed windows."""
    key_str = f"{video_path}_{sample_rate}_{dinov3_model}_{ref_patch}_{similarity_threshold}_{border_size}_{window_size}_{stride}_{max_patches_per_frame}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_video_info(video_path: str) -> Tuple[int, float]:
    """Get video frame count and fps using OpenCV (fast, no GPU memory)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames, fps


class BehaviorDataset(Dataset):
    """
    Dataset for behavior classification with efficient video-level caching.
    Extracts DINOv3 features for each video ONCE, caches to disk (float16).
    Creates windows on-the-fly from cached features (fast CPU operations).
    """

    def __init__(
        self,
        video_configs: List[VideoConfig],
        behavior_list: List[str],
        dinov3_extractor: DINOv3Extractor,
        window_size: int = 16,
        stride: int = 1,
        similarity_threshold: float = 0.80,
        border_size: int = 1,
        max_patches_per_frame: int = 50,
        frame_sample_rate: int = 100,
        cache_dir: Optional[str] = None,
        mode: str = 'train'  # 'train', 'val', 'test'
    ):
        self.video_configs = video_configs
        self.behavior_list = behavior_list
        self.num_behaviors = len(behavior_list)
        self.behavior_to_idx = {b: i for i, b in enumerate(behavior_list)}

        self.dinov3_extractor = dinov3_extractor
        self.window_size = window_size
        self.stride = stride
        self.similarity_threshold = similarity_threshold
        self.border_size = border_size
        self.max_patches_per_frame = max_patches_per_frame
        self.frame_sample_rate = frame_sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mode = mode

        # Dinov3 model name for cache key
        self.dinov3_model = getattr(dinov3_extractor, 'model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m')

        # Build window index mapping: global_idx -> (video_idx, local_window_idx)
        self.window_index = []  # List of (video_idx, local_window_idx)
        self.video_window_counts = []  # Number of windows per video
        self.video_metadata = []  # Per-video metadata
        self.video_labels = []  # Per-video labels (list per frame)
        self.video_ref_patch_embeddings = {}  # Cache for reference patch embeddings

        self._build_index()

    def _build_index(self):
        """Build index of all windows across all videos WITHOUT extracting features."""
        for video_idx, video_config in enumerate(self.video_configs):
            video_path = video_config.video_path
            label_path = video_config.label_path
            ref_patch = video_config.reference_patch

            print(f"Indexing video {video_idx + 1}/{len(self.video_configs)}: {video_path}")
            print(f"  Reference patch: {ref_patch}")

            # Load labels
            labels = self._load_labels(label_path)
            self.video_labels.append(labels)

            # Get video info using OpenCV (fast, no GPU memory)
            try:
                total_frames, fps = _get_video_info(video_path)
                print(f"  Video: {total_frames} frames, {fps:.1f} FPS")
            except Exception as e:
                print(f"  Warning: Could not read video info: {e}")
                self.video_window_counts.append(0)
                self.video_metadata.append(None)
                continue

            # Calculate frame indices that would be sampled
            frame_indices = list(range(0, total_frames, self.frame_sample_rate))
            num_sampled_frames = len(frame_indices)

            if num_sampled_frames < self.window_size:
                print(f"  Warning: Not enough frames ({num_sampled_frames}) for window size ({self.window_size})")
                self.video_window_counts.append(0)
                self.video_metadata.append(None)
                continue

            # Calculate number of sliding windows
            num_windows = (num_sampled_frames - self.window_size) // self.stride + 1
            if num_windows <= 0:
                num_windows = 0

            print(f"  Sampled frames: {num_sampled_frames}, Estimated windows: {num_windows}")

            self.video_window_counts.append(num_windows)

            # Build global index
            for w in range(num_windows):
                self.window_index.append((video_idx, w))

            # Store metadata for this video
            self.video_metadata.append({
                'video_path': video_path,
                'ref_patch': ref_patch,
                'frame_indices': frame_indices,
                'total_frames': total_frames,
                'fps': fps,
                'num_sampled_frames': num_sampled_frames,
                'num_windows': num_windows
            })

        total_windows = len(self.window_index)
        print(f"Total windows across all videos: {total_windows}")

        # Pre-extract reference patches for all videos (only first frame)
        self._precompute_reference_patches()

    def precompute_all_features(self):
        """
        Pre-extract and cache features for ALL videos in this dataset.
        Call this in the main process BEFORE creating DataLoader to avoid
        feature extraction in worker processes (which causes CUDA/shared memory issues).
        """
        print(f"Pre-computing features for {len(self.video_configs)} videos...")
        for video_idx in range(len(self.video_configs)):
            metadata = self.video_metadata[video_idx]
            if metadata is None or metadata['num_windows'] == 0:
                continue

            # This will extract and cache features if not already cached
            self._get_or_extract_video_features(video_idx)

            # Also pre-compute processed frames (patch selection + padding)
            # This uses the global cache, so it's fast after first run
            self._get_or_compute_processed_frames(video_idx)

        print(f"Feature pre-computation complete for all videos.")

        # After pre-computation, we don't need the extractor in workers
        # Set to None to avoid pickling the DINOv3 model (which causes CUDA issues)
        self.dinov3_extractor = None

    def _precompute_reference_patches(self):
        """Extract reference patch embedding from first frame of each video."""
        for video_idx, video_config in enumerate(self.video_configs):
            metadata = self.video_metadata[video_idx]
            if metadata is None or metadata['num_windows'] == 0:
                self.video_ref_patch_embeddings[video_idx] = None
                continue

            video_path = video_config.video_path
            ref_patch = video_config.reference_patch

            cache_key = _get_video_cache_key(video_path, self.frame_sample_rate, self.dinov3_model)

            # Check cache for reference patch
            if self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}_ref_patch.pt"
                if cache_file.exists():
                    try:
                        cached = torch.load(cache_file, map_location='cpu', weights_only=False)
                        self.video_ref_patch_embeddings[video_idx] = cached['ref_patch_embedding']
                        print(f"  Video {video_idx}: Using cached reference patch")
                        continue
                    except Exception as e:
                        print(f"  Reference patch cache read failed: {e}")

            # Extract first frame only
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"  Warning: Could not read first frame of {video_path}")
                self.video_ref_patch_embeddings[video_idx] = None
                continue

            # Process first frame (BGR->RGB->Grayscale->3-channel as in notebook)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.repeat(frame[..., None], 3, axis=-1)

            # Extract features
            features = self.dinov3_extractor.extract_frame_features(frame)  # H_p x W_p x D

            # Get reference patch
            ref_row, ref_col = ref_patch
            H_p, W_p = features.shape[0], features.shape[1]
            if ref_row >= H_p or ref_col >= W_p:
                ref_row = H_p // 2
                ref_col = W_p // 2
                print(f"  Warning: Reference patch out of bounds, using center ({ref_row}, {ref_col})")

            ref_embedding = features[ref_row, ref_col]  # D

            self.video_ref_patch_embeddings[video_idx] = ref_embedding

            # Cache reference patch
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self.cache_dir / f"{cache_key}_ref_patch.pt"
                try:
                    torch.save({'ref_patch_embedding': ref_embedding, 'ref_patch_used': (ref_row, ref_col)}, cache_file)
                except Exception as e:
                    print(f"  Failed to cache reference patch: {e}")

    def _load_labels(self, label_path: str) -> List[List[str]]:
        """Load frame-wise labels from file."""
        if label_path.endswith('.json'):
            with open(label_path, 'r') as f:
                data = json.load(f)
            if not data:
                return [[]]
            max_idx = max(int(k) for k in data.keys())
            labels = [[] for _ in range(max_idx + 1)]
            for idx, behaviors in data.items():
                labels[int(idx)] = behaviors
            return labels
        elif label_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(label_path)
            if df.empty:
                return [[]]
            max_frame = df['frame_idx'].max()
            labels = [[] for _ in range(int(max_frame) + 1)]
            for _, row in df.iterrows():
                labels[int(row['frame_idx'])].append(row['behavior'])
            return labels
        else:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            labels = []
            for line in lines:
                line = line.strip()
                if line:
                    behaviors = [b.strip() for b in line.split(',')]
                else:
                    behaviors = []
                labels.append(behaviors)
            return labels

    def _get_or_extract_video_features(self, video_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached features for all sampled frames of a video, or extract and cache them.
        Features are stored in float16 on disk to save space, returned as float32.
        Returns: (features, cls_tokens) where features: (num_sampled_frames, H_p, W_p, D)
        """
        video_config = self.video_configs[video_idx]
        video_path = video_config.video_path
        metadata = self.video_metadata[video_idx]

        if metadata is None or metadata['num_windows'] == 0:
            return torch.empty(0), torch.empty(0)

        cache_key = _get_video_cache_key(video_path, self.frame_sample_rate, self.dinov3_model)

        # Check cache for full video features (including CLS tokens)
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}_features.pt"
            if cache_file.exists():
                try:
                    cached = torch.load(cache_file, map_location='cpu', weights_only=False)
                    # Verify the cached features match expected frame count
                    if cached['features'].shape[0] == metadata['num_sampled_frames']:
                        # Convert from float16 to float32 for model
                        features = cached['features'].float() if cached['features'].dtype == torch.float16 else cached['features']
                        cls_tokens = cached.get('cls_tokens')
                        if cls_tokens is not None:
                            cls_tokens = cls_tokens.float() if cls_tokens.dtype == torch.float16 else cls_tokens
                        return features, cls_tokens
                    else:
                        print(f"  Cached feature count mismatch, re-extracting...")
                except Exception as e:
                    print(f"  Feature cache read failed: {e}")

        # Extract features for all sampled frames
        frame_indices = metadata['frame_indices']
        print(f"  Extracting features for video {video_idx} ({len(frame_indices)} frames)...")

        cap = cv2.VideoCapture(video_path)
        features_list = []
        cls_tokens_list = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                features_list.append(torch.zeros(14, 14, 768))
                cls_tokens_list.append(torch.zeros(768))
                continue

            # Process frame (BGR->RGB->Grayscale->3-channel as in notebook)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.repeat(frame[..., None], 3, axis=-1)

            features, cls_token = self.dinov3_extractor.extract_frame_features(frame, return_cls=True)
            features_list.append(features)
            cls_tokens_list.append(cls_token)

        cap.release()

        if not features_list:
            return torch.empty(0), torch.empty(0)

        # Stack features: (num_sampled_frames, H_p, W_p, D)
        features = torch.stack(features_list)
        cls_tokens = torch.stack(cls_tokens_list)  # (num_sampled_frames, D)

        # Cache full video features with CLS tokens (store as float16 to save space)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{cache_key}_features.pt"
            try:
                torch.save({
                    'features': features.half(),  # Store as float16
                    'cls_tokens': cls_tokens.half(),
                    'frame_indices': frame_indices
                }, cache_file)
                print(f"  Cached features for video {video_idx} (float16, {features.numel() * 2 / 1e9:.2f} GB)")
            except Exception as e:
                print(f"  Failed to cache features: {e}")

        return features, cls_tokens

    def _get_or_compute_processed_frames(self, video_idx: int):
        """
        Get or compute processed per-frame embeddings (after patch selection + padding).
        Uses a global module-level cache to avoid pickling issues with DataLoader workers.
        Returns: (padded_embeddings, patch_mask, cls_tokens, frame_indices, labels)
        """
        global _PROCESSED_FRAMES_CACHE

        cache_key = (video_idx, self.frame_sample_rate, self.dinov3_model,
                    self.similarity_threshold, self.border_size, self.max_patches_per_frame)

        if cache_key in _PROCESSED_FRAMES_CACHE:
            return _PROCESSED_FRAMES_CACHE[cache_key]

        video_config = self.video_configs[video_idx]
        video_path = video_config.video_path
        ref_patch = video_config.reference_patch
        labels = self.video_labels[video_idx]
        metadata = self.video_metadata[video_idx]

        if metadata is None or metadata['num_windows'] == 0:
            result = (torch.empty(0, self.max_patches_per_frame, 768),
                     torch.empty(0, self.max_patches_per_frame, dtype=torch.bool),
                     torch.empty(0, 768),
                     [],
                     labels)
            _PROCESSED_FRAMES_CACHE[cache_key] = result
            return result

        ref_embedding = self.video_ref_patch_embeddings[video_idx]
        if ref_embedding is None:
            result = (torch.empty(0, self.max_patches_per_frame, 768),
                     torch.empty(0, self.max_patches_per_frame, dtype=torch.bool),
                     torch.empty(0, 768),
                     [],
                     labels)
            _PROCESSED_FRAMES_CACHE[cache_key] = result
            return result

        # Get cached video features with CLS tokens
        features, cls_tokens = self._get_or_extract_video_features(video_idx)
        if features.shape[0] == 0:
            result = (torch.empty(0, self.max_patches_per_frame, 768),
                     torch.empty(0, self.max_patches_per_frame, dtype=torch.bool),
                     torch.empty(0, 768),
                     metadata['frame_indices'],
                     labels)
            _PROCESSED_FRAMES_CACHE[cache_key] = result
            return result

        # Select similar patches for all frames
        selected_embeddings, _ = select_similar_patches(
            features,
            ref_embedding,
            threshold=self.similarity_threshold,
            border=self.border_size
        )

        # Pad to fixed length
        padded_embeddings, patch_mask = pad_patch_sequence(
            selected_embeddings,
            max_patches=self.max_patches_per_frame
        )

        frame_indices = metadata['frame_indices']
        result = (padded_embeddings, patch_mask, cls_tokens, frame_indices, labels)
        _PROCESSED_FRAMES_CACHE[cache_key] = result
        return result

    def __getitem__(self, idx):
        video_idx, local_window_idx = self.window_index[idx]

        # Get processed per-frame data (cached globally)
        padded_embeddings, patch_mask, cls_tokens, frame_indices, labels = self._get_or_compute_processed_frames(video_idx)

        if padded_embeddings.shape[0] == 0:
            return {
                'patch_embeddings': torch.zeros(self.window_size, self.max_patches_per_frame, 768),
                'cls_tokens': torch.zeros(self.window_size, 768),
                'patch_mask': torch.zeros(self.window_size, self.max_patches_per_frame, dtype=torch.bool),
                'labels': torch.zeros(self.num_behaviors, dtype=torch.float32),
                'metadata': {
                    'video_idx': video_idx,
                    'video_path': self.video_configs[video_idx].video_path,
                    'window_start_frame': -1,
                    'last_frame_idx': -1
                }
            }

        # Extract the specific window from the padded embeddings
        start_frame = local_window_idx * self.stride
        end_frame = start_frame + self.window_size

        window = padded_embeddings[start_frame:end_frame]
        window_patch_mask = patch_mask[start_frame:end_frame]
        window_cls_tokens = cls_tokens[start_frame:end_frame]

        # Create label for this window (label of the last frame in window)
        label = torch.zeros(self.num_behaviors, dtype=torch.float32)
        if end_frame - 1 < len(frame_indices):
            last_frame_idx = frame_indices[end_frame - 1]
            if last_frame_idx < len(labels):
                frame_labels = labels[last_frame_idx]
                for behavior in frame_labels:
                    if behavior in self.behavior_to_idx:
                        label[self.behavior_to_idx[behavior]] = 1.0

        metadata = {
            'video_idx': video_idx,
            'video_path': self.video_configs[video_idx].video_path,
            'window_start_frame': start_frame,
            'last_frame_idx': frame_indices[end_frame - 1] if end_frame - 1 < len(frame_indices) else -1
        }

        return {
            'patch_embeddings': window,
            'cls_tokens': window_cls_tokens,
            'patch_mask': window_patch_mask,
            'labels': label,
            'metadata': metadata
        }

    def __len__(self):
        return len(self.window_index)


class PredictionDataset(Dataset):
    """
    Dataset for prediction on unseen videos with efficient video-level caching.
    Extracts DINOv3 features for each video ONCE, caches to disk (float16).
    Processes windows from cached features with in-memory caching.
    """

    def __init__(
        self,
        video_configs: List[VideoConfig],
        dinov3_extractor: DINOv3Extractor,
        window_size: int = 16,
        stride: int = 1,
        similarity_threshold: float = 0.80,
        border_size: int = 1,
        max_patches_per_frame: int = 50,
        frame_sample_rate: int = 100,
        cache_dir: Optional[str] = None
    ):
        self.video_configs = video_configs
        self.dinov3_extractor = dinov3_extractor
        self.window_size = window_size
        self.stride = stride
        self.similarity_threshold = similarity_threshold
        self.border_size = border_size
        self.max_patches_per_frame = max_patches_per_frame
        self.frame_sample_rate = frame_sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.dinov3_model = getattr(dinov3_extractor, 'model_name', 'facebook/dinov3-vitb16-pretrain-lvd1689m')

        self.window_index = []
        self.video_window_counts = []
        self.video_metadata = []
        self.video_ref_patch_embeddings = {}

        self._build_index()
        self._precompute_reference_patches()

    def _build_index(self):
        """Build index of all windows without extracting features."""
        for video_idx, video_config in enumerate(self.video_configs):
            video_path = video_config.video_path
            ref_patch = video_config.reference_patch

            print(f"Indexing video {video_idx + 1}/{len(self.video_configs)}: {video_path}")
            print(f"  Reference patch: {ref_patch}")

            try:
                total_frames, fps = _get_video_info(video_path)
                print(f"  Video: {total_frames} frames, {fps:.1f} FPS")
            except Exception as e:
                print(f"  Warning: Could not read video info: {e}")
                self.video_window_counts.append(0)
                self.video_metadata.append(None)
                continue

            frame_indices = list(range(0, total_frames, self.frame_sample_rate))
            num_sampled_frames = len(frame_indices)

            if num_sampled_frames < self.window_size:
                print(f"  Warning: Not enough frames ({num_sampled_frames}) for window size ({self.window_size})")
                self.video_window_counts.append(0)
                self.video_metadata.append(None)
                continue

            num_windows = (num_sampled_frames - self.window_size) // self.stride + 1
            if num_windows <= 0:
                num_windows = 0

            print(f"  Sampled frames: {num_sampled_frames}, Estimated windows: {num_windows}")

            self.video_window_counts.append(num_windows)

            for w in range(num_windows):
                self.window_index.append((video_idx, w))

            self.video_metadata.append({
                'video_path': video_path,
                'ref_patch': ref_patch,
                'frame_indices': frame_indices,
                'total_frames': total_frames,
                'fps': fps,
                'num_sampled_frames': num_sampled_frames,
                'num_windows': num_windows
            })

        total_windows = len(self.window_index)
        print(f"Total prediction windows: {total_windows}")

    def _precompute_reference_patches(self):
        """Extract reference patch embedding from first frame of each video."""
        for video_idx, video_config in enumerate(self.video_configs):
            metadata = self.video_metadata[video_idx]
            if metadata is None or metadata['num_windows'] == 0:
                self.video_ref_patch_embeddings[video_idx] = None
                continue

            video_path = video_config.video_path
            ref_patch = video_config.reference_patch

            cache_key = _get_video_cache_key(video_path, self.frame_sample_rate, self.dinov3_model)

            if self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}_ref_patch.pt"
                if cache_file.exists():
                    try:
                        cached = torch.load(cache_file, map_location='cpu', weights_only=False)
                        self.video_ref_patch_embeddings[video_idx] = cached['ref_patch_embedding']
                        print(f"  Video {video_idx}: Using cached reference patch")
                        continue
                    except Exception as e:
                        print(f"  Reference patch cache read failed: {e}")

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"  Warning: Could not read first frame of {video_path}")
                self.video_ref_patch_embeddings[video_idx] = None
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.repeat(frame[..., None], 3, axis=-1)

            features = self.dinov3_extractor.extract_frame_features(frame)
            ref_row, ref_col = ref_patch
            H_p, W_p = features.shape[0], features.shape[1]
            if ref_row >= H_p or ref_col >= W_p:
                ref_row = H_p // 2
                ref_col = W_p // 2

            ref_embedding = features[ref_row, ref_col]
            self.video_ref_patch_embeddings[video_idx] = ref_embedding

            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file = self.cache_dir / f"{cache_key}_ref_patch.pt"
                try:
                    torch.save({'ref_patch_embedding': ref_embedding, 'ref_patch_used': (ref_row, ref_col)}, cache_file)
                except Exception as e:
                    print(f"  Failed to cache reference patch: {e}")

    def precompute_all_features(self):
        """
        Pre-extract and cache features for ALL videos in this dataset.
        Call this in the main process BEFORE creating DataLoader to avoid
        feature extraction in worker processes (which causes CUDA/shared memory issues).
        """
        print(f"Pre-computing features for {len(self.video_configs)} videos...")
        for video_idx in range(len(self.video_configs)):
            metadata = self.video_metadata[video_idx]
            if metadata is None or metadata['num_windows'] == 0:
                continue

            # This will extract and cache features if not already cached
            self._get_or_extract_video_features(video_idx)

            # Also pre-compute processed frames (patch selection + padding)
            self._get_or_compute_processed_frames(video_idx)

        print(f"Feature pre-computation complete for all videos.")

        # After pre-computation, we don't need the extractor in workers
        self.dinov3_extractor = None

    def _get_or_extract_video_features(self, video_idx: int):
        """Get cached features for all sampled frames of a video, or extract and cache them.
        Returns: (features, cls_tokens) or just features if no CLS tokens available
        """
        video_config = self.video_configs[video_idx]
        video_path = video_config.video_path
        metadata = self.video_metadata[video_idx]

        if metadata is None or metadata['num_windows'] == 0:
            return torch.empty(0), torch.empty(0)

        cache_key = _get_video_cache_key(video_path, self.frame_sample_rate, self.dinov3_model)

        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}_features.pt"
            if cache_file.exists():
                try:
                    cached = torch.load(cache_file, map_location='cpu', weights_only=False)
                    if cached['features'].shape[0] == metadata['num_sampled_frames']:
                        cls_tokens = cached.get('cls_tokens', None)
                        if cls_tokens is not None:
                            return cached['features'], cls_tokens
                        return cached['features'], torch.empty(0)
                except Exception as e:
                    print(f"  Feature cache read failed: {e}")

        frame_indices = metadata['frame_indices']
        print(f"  Extracting features for video {video_idx} ({len(frame_indices)} frames)...")

        cap = cv2.VideoCapture(video_path)
        features_list = []
        cls_tokens_list = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                features_list.append(torch.zeros(14, 14, 768))
                cls_tokens_list.append(torch.zeros(768))
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = np.repeat(frame[..., None], 3, axis=-1)

            features, cls_token = self.dinov3_extractor.extract_frame_features(frame, return_cls=True)
            features_list.append(features)
            cls_tokens_list.append(cls_token)

        cap.release()

        if not features_list:
            return torch.empty(0), torch.empty(0)

        features = torch.stack(features_list)
        cls_tokens = torch.stack(cls_tokens_list)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{cache_key}_features.pt"
            try:
                torch.save({
                    'features': features.half(),
                    'cls_tokens': cls_tokens.half(),
                    'frame_indices': frame_indices
                }, cache_file)
                print(f"  Cached features + CLS tokens for video {video_idx} (float16)")
            except Exception as e:
                print(f"  Failed to cache features: {e}")

        return features, cls_tokens

    def _get_or_compute_processed_frames(self, video_idx: int):
        """
        Get or compute processed per-frame embeddings (after patch selection + padding).
        Uses a global module-level cache to avoid pickling issues with DataLoader workers.
        Returns: (padded_embeddings, patch_mask, cls_tokens, frame_indices)
        """
        global _PROCESSED_FRAMES_CACHE

        cache_key = (video_idx, self.frame_sample_rate, self.dinov3_model,
                    self.similarity_threshold, self.border_size, self.max_patches_per_frame)

        if cache_key in _PROCESSED_FRAMES_CACHE:
            return _PROCESSED_FRAMES_CACHE[cache_key]

        video_config = self.video_configs[video_idx]
        video_path = video_config.video_path
        ref_patch = video_config.reference_patch
        metadata = self.video_metadata[video_idx]

        if metadata is None or metadata['num_windows'] == 0:
            result = (torch.empty(0, self.max_patches_per_frame, 768),
                     torch.empty(0, self.max_patches_per_frame, dtype=torch.bool),
                     torch.empty(0, 768),
                     [])
            _PROCESSED_FRAMES_CACHE[cache_key] = result
            return result

        ref_embedding = self.video_ref_patch_embeddings[video_idx]
        if ref_embedding is None:
            result = (torch.empty(0, self.max_patches_per_frame, 768),
                     torch.empty(0, self.max_patches_per_frame, dtype=torch.bool),
                     torch.empty(0, 768),
                     [])
            _PROCESSED_FRAMES_CACHE[cache_key] = result
            return result

        # Get cached video features with CLS tokens
        features, cls_tokens = self._get_or_extract_video_features(video_idx)
        if features.shape[0] == 0:
            result = (torch.empty(0, self.max_patches_per_frame, 768),
                     torch.empty(0, self.max_patches_per_frame, dtype=torch.bool),
                     torch.empty(0, 768),
                     metadata['frame_indices'])
            _PROCESSED_FRAMES_CACHE[cache_key] = result
            return result

        # Select similar patches for all frames
        selected_embeddings, _ = select_similar_patches(
            features,
            ref_embedding,
            threshold=self.similarity_threshold,
            border=self.border_size
        )

        # Pad to fixed length
        padded_embeddings, patch_mask = pad_patch_sequence(
            selected_embeddings,
            max_patches=self.max_patches_per_frame
        )

        frame_indices = metadata['frame_indices']
        result = (padded_embeddings, patch_mask, cls_tokens, frame_indices)
        _PROCESSED_FRAMES_CACHE[cache_key] = result
        return result

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        video_idx, local_window_idx = self.window_index[idx]

        # Get processed per-frame data (cached globally)
        padded_embeddings, patch_mask, cls_tokens, frame_indices = self._get_or_compute_processed_frames(video_idx)

        if padded_embeddings.shape[0] == 0:
            return {
                'patch_embeddings': torch.zeros(self.window_size, self.max_patches_per_frame, 768),
                'cls_tokens': torch.zeros(self.window_size, 768),
                'patch_mask': torch.zeros(self.window_size, self.max_patches_per_frame, dtype=torch.bool),
                'metadata': {
                    'video_idx': video_idx,
                    'video_path': self.video_configs[video_idx].video_path,
                    'window_start_frame': -1,
                    'last_frame_idx': -1,
                    'frame_indices': []
                }
            }

        # Extract the specific window from the padded embeddings
        start_frame = local_window_idx * self.stride
        end_frame = start_frame + self.window_size

        window = padded_embeddings[start_frame:end_frame]
        window_patch_mask = patch_mask[start_frame:end_frame]
        window_cls_tokens = cls_tokens[start_frame:end_frame]

        metadata_info = self.video_metadata[video_idx]
        last_frame_idx = frame_indices[end_frame - 1] if end_frame - 1 < len(frame_indices) else -1

        meta = {
            'video_idx': video_idx,
            'video_path': self.video_configs[video_idx].video_path,
            'window_start_frame': start_frame,
            'last_frame_idx': last_frame_idx,
            'frame_indices': frame_indices[start_frame:end_frame] if end_frame <= len(frame_indices) else []
        }

        return {
            'patch_embeddings': window,
            'cls_tokens': window_cls_tokens,
            'patch_mask': window_patch_mask,
            'metadata': meta
        }


def create_data_loaders_from_config(
    data_config: DataConfig,
    behavior_list: List[str],
    dinov3_extractor: DINOv3Extractor,
    config: Dict,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders from DataConfig."""

    train_dataset = BehaviorDataset(
        video_configs=data_config.train,
        behavior_list=behavior_list,
        dinov3_extractor=dinov3_extractor,
        window_size=config.get('window_size', 16),
        stride=config.get('stride', 1),
        similarity_threshold=config.get('patch_similarity_threshold', 0.80),
        border_size=config.get('border_size', 1),
        max_patches_per_frame=config.get('max_patches_per_frame', 50),
        frame_sample_rate=config.get('frame_sample_rate', 100),
        cache_dir=cache_dir,
        mode='train'
    )

    val_dataset = BehaviorDataset(
        video_configs=data_config.val,
        behavior_list=behavior_list,
        dinov3_extractor=dinov3_extractor,
        window_size=config.get('window_size', 16),
        stride=config.get('stride', 1),
        similarity_threshold=config.get('patch_similarity_threshold', 0.80),
        border_size=config.get('border_size', 1),
        max_patches_per_frame=config.get('max_patches_per_frame', 50),
        frame_sample_rate=config.get('frame_sample_rate', 100),
        cache_dir=cache_dir,
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


# Backward compatibility function
def create_data_loaders(
    train_videos: List[str],
    train_labels: List[str],
    val_videos: List[str],
    val_labels: List[str],
    behavior_list: List[str],
    dinov3_extractor: DINOv3Extractor,
    reference_patch_coords: Tuple[int, int],
    config: Dict,
    cache_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders (legacy interface)."""
    from utils.data_config import create_data_config_from_lists

    train_ref_patches = [reference_patch_coords] * len(train_videos)
    val_ref_patches = [reference_patch_coords] * len(val_videos)

    data_config = create_data_config_from_lists(
        train_videos, train_labels, train_ref_patches,
        val_videos, val_labels, val_ref_patches
    )

    return create_data_loaders_from_config(data_config, behavior_list, dinov3_extractor, config, cache_dir)


if __name__ == "__main__":
    print("Dataset module loaded successfully")