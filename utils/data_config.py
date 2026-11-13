"""
Data configuration utilities for loading video/label/reference_patch from JSON.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class VideoConfig:
    """Configuration for a single video."""
    video_path: str
    label_path: str
    reference_patch: Tuple[int, int]

    @classmethod
    def from_dict(cls, d: Dict) -> 'VideoConfig':
        return cls(
            video_path=d['video_path'],
            label_path=d['label_path'],
            reference_patch=tuple(d['reference_patch'])
        )


@dataclass
class DataConfig:
    """Complete data configuration with train/val/test splits."""
    train: List[VideoConfig]
    val: List[VideoConfig]
    test: List[VideoConfig]

    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'DataConfig':
        with open(json_path, 'r') as f:
            data = json.load(f)

        return cls(
            train=[VideoConfig.from_dict(v) for v in data.get('train', [])],
            val=[VideoConfig.from_dict(v) for v in data.get('val', [])],
            test=[VideoConfig.from_dict(v) for v in data.get('test', [])]
        )

    def get_train_lists(self) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
        """Get train video_paths, label_paths, reference_patches."""
        return (
            [v.video_path for v in self.train],
            [v.label_path for v in self.train],
            [v.reference_patch for v in self.train]
        )

    def get_val_lists(self) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
        """Get val video_paths, label_paths, reference_patches."""
        return (
            [v.video_path for v in self.val],
            [v.label_path for v in self.val],
            [v.reference_patch for v in self.val]
        )

    def get_test_lists(self) -> Tuple[List[str], List[str], List[Tuple[int, int]]]:
        """Get test video_paths, label_paths, reference_patches."""
        return (
            [v.video_path for v in self.test],
            [v.label_path for v in self.test],
            [v.reference_patch for v in self.test]
        )

    def get_predict_lists(self) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Get predict video_paths, reference_patches (no labels needed)."""
        # Use test split for prediction if available, otherwise train
        videos = self.test if self.test else self.train
        return (
            [v.video_path for v in videos],
            [v.reference_patch for v in videos]
        )


def load_data_config(json_path: Union[str, Path]) -> DataConfig:
    """Load data configuration from JSON file."""
    return DataConfig.from_json(json_path)


def create_data_config_from_lists(
    train_videos: List[str],
    train_labels: List[str],
    train_ref_patches: List[Tuple[int, int]],
    val_videos: List[str],
    val_labels: List[str],
    val_ref_patches: List[Tuple[int, int]],
    test_videos: Optional[List[str]] = None,
    test_labels: Optional[List[str]] = None,
    test_ref_patches: Optional[List[Tuple[int, int]]] = None
) -> DataConfig:
    """Create DataConfig from separate lists (backward compatibility)."""
    train = [
        VideoConfig(video_path=v, label_path=l, reference_patch=r)
        for v, l, r in zip(train_videos, train_labels, train_ref_patches)
    ]
    val = [
        VideoConfig(video_path=v, label_path=l, reference_patch=r)
        for v, l, r in zip(val_videos, val_labels, val_ref_patches)
    ]
    test = []
    if test_videos:
        test_labels = test_labels or [''] * len(test_videos)
        test_ref_patches = test_ref_patches or [(7, 7)] * len(test_videos)
        test = [
            VideoConfig(video_path=v, label_path=l, reference_patch=r)
            for v, l, r in zip(test_videos, test_labels, test_ref_patches)
        ]

    return DataConfig(train=train, val=val, test=test)


if __name__ == "__main__":
    # Test loading
    config = load_data_config('configs/data_config_example.json')
    print(f"Train videos: {len(config.train)}")
    print(f"Val videos: {len(config.val)}")
    print(f"Test videos: {len(config.test)}")
    for v in config.train:
        print(f"  {v.video_path} -> {v.label_path} (ref: {v.reference_patch})")