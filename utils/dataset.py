"""
Data loading utilities for behavior classification.
Supports multi-video training with video-level validation split.
Per-video reference patch configuration.
Lazy loading with disk caching for long videos.
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


def _get_video_cache_key(video_path: str, sample_rate: int, dinov3_model: str) -> str:
    """Generate a unique cache key for a video."""
    key_str = f"{video_path}_{sample_rate}_{dinov3_model}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_windows_cache_key(video_path: str, sample_rate: int, dinov3_model: str,
                           ref_patch: Tuple[int, int], similarity_threshold: float,
                           border_size: int, window_size: int, stride: int,
                           max_patches_per_frame: int, frame_sample_rate: int) -> str:
    """Generate a unique cache key for processed windows."""
    key_str = f"{video_path}_{sample_rate}_{dinov3_model}_{ref_patch}_{similarity_threshold}_{border_size}_{window_size}_{stride}_{max_patches_per_frame}_{frame_sample_rate}"
    return hashlib.md5(key_str.encode()).hexdigest()


class BehaviorDataset(Dataset):
    """
    Dataset for behavior classification with lazy loading.
    Handles multiple videos with video-level train/val split.
    Each video can have its own reference patch coordinates.
    Features are cached to disk to avoid re-processing long videos.
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

        self._build_index()

    def _build_index(self):
        """Build index of all windows across all videos without loading data."""
        for video_idx, video_config in enumerate(self.video_configs):
            video_path = video_config.video_path
            label_path = video_config.label_path
            ref_patch = video_config.reference_patch

            print(f"Indexing video {video_idx + 1}/{len(self.video_configs)}: {video_path}")
            print(f"  Reference patch: {ref_patch}")

            # Load labels
            labels = self._load_labels(label_path)
            self.video_labels.append(labels)

            # Check cache for pre-computed window count
            cache_key = _get_windows_cache_key(
                video_path, self.frame_sample_rate, self.dinov3_model,
                ref_patch, self.similarity_threshold, self.border_size,
                self.window_size, self.stride, self.max_patches_per_frame,
                self.frame_sample_rate
            )

            num_windows = 0
            frame_indices = None

            if self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}_windows.pt"
                if cache_file.exists():
                    try:
                        cached = torch.load(cache_file, map_location='cpu', weights_only=False)
                        num_windows = cached['num_windows']
                        frame_indices = cached['frame_indices']
                        print(f"  Using cached windows: {num_windows}")
                    except Exception as e:
                        print(f"  Cache read failed, will recompute: {e}")

            if num_windows == 0:
                # Need to compute window count - extract features once to get frame count
                features, frame_indices = self.dinov3_extractor.extract_video_features(
                    video_path,
                    sample_rate=self.frame_sample_rate
                )

                if features.shape[0] == 0:
                    print(f"  Warning: No frames extracted from {video_path}")
                    self.video_window_counts.append(0)
                    self.video_metadata.append(None)
                    continue

                # Get reference patch from first frame
                ref_row, ref_col = ref_patch
                H_p, W_p = features.shape[1], features.shape[2]

                if ref_row >= H_p or ref_col >= W_p:
                    ref_row = H_p // 2
                    ref_col = W_p // 2
                    print(f"  Warning: Reference patch out of bounds, using center ({ref_row}, {ref_col})")

                # Select similar patches to get patch counts per frame
                selected_embeddings, _ = select_similar_patches(
                    features,
                    features[0, ref_row, ref_col],
                    threshold=self.similarity_threshold,
                    border=self.border_size
                )

                # Pad to get max patches per frame
                padded_embeddings, patch_mask = pad_patch_sequence(
                    selected_embeddings,
                    max_patches=self.max_patches_per_frame
                )

                # Create sliding windows to get count
                windows, _ = create_sliding_windows(
                    padded_embeddings,
                    patch_mask,
                    window_size=self.window_size,
                    stride=self.stride
                )

                num_windows = windows.shape[0]

                # Cache the frame indices for label alignment
                if self.cache_dir:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = self.cache_dir / f"{cache_key}_windows.pt"
                    torch.save({
                        'num_windows': num_windows,
                        'frame_indices': frame_indices,
                        'ref_patch_used': (ref_row, ref_col)
                    }, cache_file)

            self.video_window_counts.append(num_windows)

            # Build global index
            for w in range(num_windows):
                self.window_index.append((video_idx, w))

            # Store metadata for this video
            self.video_metadata.append({
                'video_path': video_path,
                'ref_patch': ref_patch,
                'frame_indices': frame_indices,
                'num_frames': len(frame_indices) if frame_indices is not None else 0,
                'num_windows': num_windows
            })

        total_windows = len(self.window_index)
        print(f"Total windows across all videos: {total_windows}")

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

    def _process_video_windows(self, video_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single video to generate all its windows.
        Returns (windows_tensor, window_labels_tensor)
        """
        video_config = self.video_configs[video_idx]
        video_path = video_config.video_path
        label_path = video_config.label_path
        ref_patch = video_config.reference_patch
        labels = self.video_labels[video_idx]
        metadata = self.video_metadata[video_idx]

        if metadata is None or metadata['num_windows'] == 0:
            return torch.empty(0, self.window_size, self.max_patches_per_frame, 768), \
                   torch.empty(0, self.num_behaviors)

        cache_key = _get_windows_cache_key(
            video_path, self.frame_sample_rate, self.dinov3_model,
            ref_patch, self.similarity_threshold, self.border_size,
            self.window_size, self.stride, self.max_patches_per_frame,
            self.frame_sample_rate
        )

        # Try to load fully processed windows from cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}_full.pt"
            if cache_file.exists():
                try:
                    cached = torch.load(cache_file, map_location='cpu', weights_only=False)
                    return cached['windows'], cached['window_labels']
                except Exception as e:
                    print(f"  Full cache read failed: {e}")

        # Extract features
        features, frame_indices = self.dinov3_extractor.extract_video_features(
            video_path,
            sample_rate=self.frame_sample_rate
        )

        if features.shape[0] == 0:
            return torch.empty(0, self.window_size, self.max_patches_per_frame, 768), \
                   torch.empty(0, self.num_behaviors)

        # Get reference patch
        ref_row, ref_col = ref_patch
        H_p, W_p = features.shape[1], features.shape[2]
        if ref_row >= H_p or ref_col >= W_p:
            ref_row = H_p // 2
            ref_col = W_p // 2

        reference_patch = features[0, ref_row, ref_col]

        # Select similar patches
        selected_embeddings, selected_coords = select_similar_patches(
            features,
            reference_patch,
            threshold=self.similarity_threshold,
            border=self.border_size
        )

        # Pad to fixed length
        padded_embeddings, patch_mask = pad_patch_sequence(
            selected_embeddings,
            max_patches=self.max_patches_per_frame
        )

        # Create sliding windows
        windows, window_masks = create_sliding_windows(
            padded_embeddings,
            patch_mask,
            window_size=self.window_size,
            stride=self.stride
        )

        # Create labels for each window (label of the last frame in window)
        num_windows = windows.shape[0]
        window_labels = torch.zeros(num_windows, self.num_behaviors, dtype=torch.float32)

        for w in range(num_windows):
            last_frame_idx = frame_indices[w + self.window_size - 1]
            if last_frame_idx < len(labels):
                frame_labels = labels[last_frame_idx]
                for behavior in frame_labels:
                    if behavior in self.behavior_to_idx:
                        window_labels[w, self.behavior_to_idx[behavior]] = 1.0

        # Cache fully processed windows
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{cache_key}_full.pt"
            try:
                torch.save({
                    'windows': windows,
                    'window_labels': window_labels
                }, cache_file)
            except Exception as e:
                print(f"  Failed to cache full windows: {e}")

        return windows, window_labels

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        video_idx, local_window_idx = self.window_index[idx]

        # Process video windows on demand (or load from cache)
        windows, window_labels = self._process_video_windows(video_idx)

        if windows.shape[0] == 0:
            # Return empty tensor with correct shape
            return {
                'patch_embeddings': torch.zeros(self.window_size, self.max_patches_per_frame, 768),
                'labels': torch.zeros(self.num_behaviors, dtype=torch.float32),
                'metadata': {
                    'video_idx': video_idx,
                    'video_path': self.video_configs[video_idx].video_path,
                    'window_start_frame': -1,
                    'last_frame_idx': -1
                }
            }

        window = windows[local_window_idx]
        label = window_labels[local_window_idx]

        # Get frame indices for metadata
        metadata_info = self.video_metadata[video_idx]
        frame_indices = metadata_info['frame_indices']
        last_frame_idx = frame_indices[local_window_idx + self.window_size - 1] if local_window_idx + self.window_size - 1 < len(frame_indices) else -1

        metadata = {
            'video_idx': video_idx,
            'video_path': self.video_configs[video_idx].video_path,
            'window_start_frame': local_window_idx * self.stride,
            'last_frame_idx': last_frame_idx
        }

        return {
            'patch_embeddings': window,
            'labels': label,
            'metadata': metadata
        }


class PredictionDataset(Dataset):
    """
    Dataset for prediction on unseen videos with lazy loading.
    Does not require labels.
    Supports per-video reference patches.
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

        self._build_index()

    def _build_index(self):
        """Build index of all windows without loading data."""
        for video_idx, video_config in enumerate(self.video_configs):
            video_path = video_config.video_path
            ref_patch = video_config.reference_patch

            print(f"Indexing video {video_idx + 1}/{len(self.video_configs)}: {video_path}")
            print(f"  Reference patch: {ref_patch}")

            cache_key = _get_windows_cache_key(
                video_path, self.frame_sample_rate, self.dinov3_model,
                ref_patch, self.similarity_threshold, self.border_size,
                self.window_size, self.stride, self.max_patches_per_frame,
                self.frame_sample_rate
            )

            num_windows = 0
            frame_indices = None

            if self.cache_dir:
                cache_file = self.cache_dir / f"{cache_key}_windows.pt"
                if cache_file.exists():
                    try:
                        cached = torch.load(cache_file, map_location='cpu', weights_only=False)
                        num_windows = cached['num_windows']
                        frame_indices = cached['frame_indices']
                        print(f"  Using cached windows: {num_windows}")
                    except Exception as e:
                        print(f"  Cache read failed: {e}")

            if num_windows == 0:
                features, frame_indices = self.dinov3_extractor.extract_video_features(
                    video_path,
                    sample_rate=self.frame_sample_rate
                )

                if features.shape[0] == 0:
                    print(f"  Warning: No frames extracted from {video_path}")
                    self.video_window_counts.append(0)
                    self.video_metadata.append(None)
                    continue

                ref_row, ref_col = ref_patch
                H_p, W_p = features.shape[1], features.shape[2]
                if ref_row >= H_p or ref_col >= W_p:
                    ref_row = H_p // 2
                    ref_col = W_p // 2

                selected_embeddings, _ = select_similar_patches(
                    features,
                    features[0, ref_row, ref_col],
                    threshold=self.similarity_threshold,
                    border=self.border_size
                )

                padded_embeddings, patch_mask = pad_patch_sequence(
                    selected_embeddings,
                    max_patches=self.max_patches_per_frame
                )

                windows, _ = create_sliding_windows(
                    padded_embeddings,
                    patch_mask,
                    window_size=self.window_size,
                    stride=self.stride
                )

                num_windows = windows.shape[0]

                if self.cache_dir:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = self.cache_dir / f"{cache_key}_windows.pt"
                    torch.save({
                        'num_windows': num_windows,
                        'frame_indices': frame_indices,
                        'ref_patch_used': (ref_row, ref_col)
                    }, cache_file)

            self.video_window_counts.append(num_windows)

            for w in range(num_windows):
                self.window_index.append((video_idx, w))

            self.video_metadata.append({
                'video_path': video_path,
                'ref_patch': ref_patch,
                'frame_indices': frame_indices,
                'num_frames': len(frame_indices) if frame_indices is not None else 0,
                'num_windows': num_windows
            })

        total_windows = len(self.window_index)
        print(f"Total prediction windows: {total_windows}")

    def _process_video_windows(self, video_idx: int) -> torch.Tensor:
        """Process a single video to generate all its windows."""
        video_config = self.video_configs[video_idx]
        video_path = video_config.video_path
        ref_patch = video_config.reference_patch
        metadata = self.video_metadata[video_idx]

        if metadata is None or metadata['num_windows'] == 0:
            return torch.empty(0, self.window_size, self.max_patches_per_frame, 768)

        cache_key = _get_windows_cache_key(
            video_path, self.frame_sample_rate, self.dinov3_model,
            ref_patch, self.similarity_threshold, self.border_size,
            self.window_size, self.stride, self.max_patches_per_frame,
            self.frame_sample_rate
        )

        if self.cache_dir:
            cache_file = self.cache_dir / f"{cache_key}_full.pt"
            if cache_file.exists():
                try:
                    cached = torch.load(cache_file, map_location='cpu', weights_only=False)
                    return cached['windows']
                except Exception as e:
                    print(f"  Full cache read failed: {e}")

        features, frame_indices = self.dinov3_extractor.extract_video_features(
            video_path,
            sample_rate=self.frame_sample_rate
        )

        if features.shape[0] == 0:
            return torch.empty(0, self.window_size, self.max_patches_per_frame, 768)

        ref_row, ref_col = ref_patch
        H_p, W_p = features.shape[1], features.shape[2]
        if ref_row >= H_p or ref_col >= W_p:
            ref_row = H_p // 2
            ref_col = W_p // 2

        reference_patch = features[0, ref_row, ref_col]

        selected_embeddings, selected_coords = select_similar_patches(
            features,
            reference_patch,
            threshold=self.similarity_threshold,
            border=self.border_size
        )

        padded_embeddings, patch_mask = pad_patch_sequence(
            selected_embeddings,
            max_patches=self.max_patches_per_frame
        )

        windows, window_masks = create_sliding_windows(
            padded_embeddings,
            patch_mask,
            window_size=self.window_size,
            stride=self.stride
        )

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = self.cache_dir / f"{cache_key}_full.pt"
            try:
                torch.save({'windows': windows}, cache_file)
            except Exception as e:
                print(f"  Failed to cache full windows: {e}")

        return windows

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        video_idx, local_window_idx = self.window_index[idx]

        windows = self._process_video_windows(video_idx)

        if windows.shape[0] == 0:
            return {
                'patch_embeddings': torch.zeros(self.window_size, self.max_patches_per_frame, 768),
                'metadata': {
                    'video_idx': video_idx,
                    'video_path': self.video_configs[video_idx].video_path,
                    'window_start_frame': -1,
                    'last_frame_idx': -1,
                    'frame_indices': []
                }
            }

        window = windows[local_window_idx]

        metadata_info = self.video_metadata[video_idx]
        frame_indices = metadata_info['frame_indices']
        last_frame_idx = frame_indices[local_window_idx + self.window_size - 1] if local_window_idx + self.window_size - 1 < len(frame_indices) else -1

        metadata = {
            'video_idx': video_idx,
            'video_path': self.video_configs[video_idx].video_path,
            'window_start_frame': local_window_idx * self.stride,
            'last_frame_idx': last_frame_idx,
            'frame_indices': frame_indices[local_window_idx:local_window_idx + self.window_size] if local_window_idx + self.window_size <= len(frame_indices) else []
        }

        return {
            'patch_embeddings': window,
            'metadata': metadata
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