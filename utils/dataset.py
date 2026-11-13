"""
Data loading utilities for behavior classification.
Supports multi-video training with video-level validation split.
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

from utils.dinov3_extractor import (
    DINOv3Extractor,
    select_similar_patches,
    pad_patch_sequence,
    create_sliding_windows
)


class BehaviorDataset(Dataset):
    """
    Dataset for behavior classification.
    Handles multiple videos with video-level train/val split.
    """

    def __init__(
        self,
        video_paths: List[str],
        label_paths: List[str],
        behavior_list: List[str],
        dinov3_extractor: DINOv3Extractor,
        reference_patch_coords: Tuple[int, int],
        window_size: int = 16,
        stride: int = 1,
        similarity_threshold: float = 0.80,
        border_size: int = 1,
        max_patches_per_frame: int = 50,
        frame_sample_rate: int = 1,
        cache_dir: Optional[str] = None,
        mode: str = 'train'  # 'train', 'val', 'test'
    ):
        self.video_paths = video_paths
        self.label_paths = label_paths
        self.behavior_list = behavior_list
        self.num_behaviors = len(behavior_list)
        self.behavior_to_idx = {b: i for i, b in enumerate(behavior_list)}

        self.dinov3_extractor = dinov3_extractor
        self.reference_patch_coords = reference_patch_coords
        self.window_size = window_size
        self.stride = stride
        self.similarity_threshold = similarity_threshold
        self.border_size = border_size
        self.max_patches_per_frame = max_patches_per_frame
        self.frame_sample_rate = frame_sample_rate
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.mode = mode

        # Load labels and create windows
        self.windows = []
        self.window_labels = []
        self.window_metadata = []  # (video_idx, window_start_frame)

        self._load_all_data()

    def _load_all_data(self):
        """Load and process all videos."""
        for video_idx, (video_path, label_path) in enumerate(zip(self.video_paths, self.label_paths)):
            print(f"Processing video {video_idx + 1}/{len(self.video_paths)}: {video_path}")

            # Load labels
            labels = self._load_labels(label_path)

            # Extract DINOv3 features
            features, frame_indices = self.dinov3_extractor.extract_video_features(
                video_path,
                sample_rate=self.frame_sample_rate
            )

            if features.shape[0] == 0:
                print(f"  Warning: No frames extracted from {video_path}")
                continue

            # Get reference patch from first frame
            ref_row, ref_col = self.reference_patch_coords
            H_p, W_p = features.shape[1], features.shape[2]

            if ref_row >= H_p or ref_col >= W_p:
                ref_row = H_p // 2
                ref_col = W_p // 2
                print(f"  Warning: Reference patch out of bounds, using center ({ref_row}, {ref_col})")

            reference_patch = features[0, ref_row, ref_col]

            # Select similar patches for all frames
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
                # Last frame of window
                last_frame_idx = frame_indices[w + self.window_size - 1]

                if last_frame_idx < len(labels):
                    frame_labels = labels[last_frame_idx]
                    for behavior in frame_labels:
                        if behavior in self.behavior_to_idx:
                            window_labels[w, self.behavior_to_idx[behavior]] = 1.0

            self.windows.append(windows)
            self.window_labels.append(window_labels)

            # Store metadata
            for w in range(num_windows):
                self.window_metadata.append({
                    'video_idx': video_idx,
                    'window_start_frame': w * self.stride,
                    'last_frame_idx': frame_indices[w + self.window_size - 1]
                })

        # Concatenate all windows
        if self.windows:
            self.windows = torch.cat(self.windows, dim=0)
            self.window_labels = torch.cat(self.window_labels, dim=0)
        else:
            self.windows = torch.empty(0, self.window_size, self.max_patches_per_frame, 768)
            self.window_labels = torch.empty(0, self.num_behaviors)

        print(f"Total windows: {len(self.windows)}")

    def _load_labels(self, label_path: str) -> List[List[str]]:
        """Load frame-wise labels from file."""
        # Support multiple formats: JSON, CSV, or simple text
        if label_path.endswith('.json'):
            with open(label_path, 'r') as f:
                data = json.load(f)
            # Expect format: {"frame_idx": ["behavior1", "behavior2"], ...}
            labels = [[] for _ in range(max(data.keys()) + 1)]
            for idx, behaviors in data.items():
                labels[int(idx)] = behaviors
            return labels
        elif label_path.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(label_path)
            # Expect columns: frame_idx, behavior (one row per behavior per frame)
            max_frame = df['frame_idx'].max()
            labels = [[] for _ in range(max_frame + 1)]
            for _, row in df.iterrows():
                labels[int(row['frame_idx'])].append(row['behavior'])
            return labels
        else:
            # Text file: one line per frame, behaviors separated by commas
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

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            'patch_embeddings': self.windows[idx],  # (window_size, max_patches, D)
            'labels': self.window_labels[idx],      # (num_behaviors,)
            'metadata': self.window_metadata[idx]
        }


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
    """Create train and validation data loaders."""

    train_dataset = BehaviorDataset(
        video_paths=train_videos,
        label_paths=train_labels,
        behavior_list=behavior_list,
        dinov3_extractor=dinov3_extractor,
        reference_patch_coords=reference_patch_coords,
        window_size=config.get('window_size', 16),
        stride=config.get('stride', 1),
        similarity_threshold=config.get('similarity_threshold', 0.80),
        border_size=config.get('border_size', 1),
        max_patches_per_frame=config.get('max_patches_per_frame', 50),
        frame_sample_rate=config.get('frame_sample_rate', 1),
        cache_dir=cache_dir,
        mode='train'
    )

    val_dataset = BehaviorDataset(
        video_paths=val_videos,
        label_paths=val_labels,
        behavior_list=behavior_list,
        dinov3_extractor=dinov3_extractor,
        reference_patch_coords=reference_patch_coords,
        window_size=config.get('window_size', 16),
        stride=config.get('stride', 1),
        similarity_threshold=config.get('similarity_threshold', 0.80),
        border_size=config.get('border_size', 1),
        max_patches_per_frame=config.get('max_patches_per_frame', 50),
        frame_sample_rate=config.get('frame_sample_rate', 1),
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


class PredictionDataset(Dataset):
    """
    Dataset for prediction on unseen videos.
    Does not require labels.
    """

    def __init__(
        self,
        video_paths: List[str],
        dinov3_extractor: DINOv3Extractor,
        reference_patch_coords: Tuple[int, int],
        window_size: int = 16,
        stride: int = 1,
        similarity_threshold: float = 0.80,
        border_size: int = 1,
        max_patches_per_frame: int = 50,
        frame_sample_rate: int = 1
    ):
        self.video_paths = video_paths
        self.dinov3_extractor = dinov3_extractor
        self.reference_patch_coords = reference_patch_coords
        self.window_size = window_size
        self.stride = stride
        self.similarity_threshold = similarity_threshold
        self.border_size = border_size
        self.max_patches_per_frame = max_patches_per_frame
        self.frame_sample_rate = frame_sample_rate

        self.windows = []
        self.window_metadata = []

        self._process_videos()

    def _process_videos(self):
        for video_idx, video_path in enumerate(self.video_paths):
            print(f"Processing video {video_idx + 1}/{len(self.video_paths)}: {video_path}")

            features, frame_indices = self.dinov3_extractor.extract_video_features(
                video_path,
                sample_rate=self.frame_sample_rate
            )

            if features.shape[0] == 0:
                continue

            ref_row, ref_col = self.reference_patch_coords
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

            self.windows.append(windows)

            for w in range(windows.shape[0]):
                self.window_metadata.append({
                    'video_idx': video_idx,
                    'video_path': video_path,
                    'window_start_frame': w * self.stride,
                    'last_frame_idx': frame_indices[w + self.window_size - 1] if w + self.window_size - 1 < len(frame_indices) else -1,
                    'frame_indices': frame_indices[w:w + self.window_size]
                })

        if self.windows:
            self.windows = torch.cat(self.windows, dim=0)
        else:
            self.windows = torch.empty(0, self.window_size, self.max_patches_per_frame, 768)

        print(f"Total prediction windows: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return {
            'patch_embeddings': self.windows[idx],
            'metadata': self.window_metadata[idx]
        }


if __name__ == "__main__":
    # Test dataset creation
    print("Dataset module loaded successfully")