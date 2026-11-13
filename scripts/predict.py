#!/usr/bin/env python3
"""
Prediction script for DINOv3-based Behavior Classification.

Usage (new JSON config approach):
    python scripts/predict.py \
        --checkpoint outputs/experiment/checkpoint_best.pth \
        --config configs/behaviors.yaml \
        --data-config configs/data_config.json \
        --output-dir outputs/predictions

Usage (legacy approach - single reference patch for all videos):
    python scripts/predict.py \
        --checkpoint outputs/experiment/checkpoint_best.pth \
        --config configs/behaviors.yaml \
        --videos data/test_video1.mp4 data/test_video2.mp4 \
        --reference-patch 7 7 \
        --output-dir outputs/predictions
"""

import argparse
import os
import sys
import yaml
import json
import time
import torch.multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.behavior_classifier import BehaviorClassifier, create_model_from_config
from utils.dinov3_extractor import DINOv3Extractor
from utils.dataset import PredictionDataset
from utils.data_config import load_data_config, create_data_config_from_lists
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Predict behaviors using trained DINOv3-based classifier')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/behaviors.yaml',
                        help='Path to config YAML file')

    # Data (new JSON config approach)
    parser.add_argument('--data-config', dest='data_config', type=str, default=None,
                        help='Path to JSON data config file with train/val/test splits and per-video reference patches')

    # Data (legacy approach - for backward compatibility)
    parser.add_argument('--videos', nargs='+', default=None,
                        help='Paths to videos for prediction (legacy)')

    # Model parameters (must match training)
    parser.add_argument('--reference-patch', dest='reference_patch', nargs=2, type=int, default=[7, 7],
                        help='Reference patch coordinates (row col) - used for all videos in legacy mode')
    parser.add_argument('--window-size', dest='window_size', type=int, default=None,
                        help='Temporal window size (frames)')
    parser.add_argument('--similarity-threshold', dest='similarity_threshold', type=float, default=None,
                        help='Similarity threshold for patch selection')
    parser.add_argument('--border-size', dest='border_size', type=int, default=None,
                        help='Border size around selected patches')
    parser.add_argument('--max-patches-per-frame', dest='max_patches_per_frame', type=int, default=50,
                        help='Maximum patches per frame after padding')
    parser.add_argument('--frame-sample-rate', dest='frame_sample_rate', type=int, default=None,
                        help='Sample every N frames (overrides config)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride between windows')

    # Output
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='outputs/predictions',
                        help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for multi-label')
    parser.add_argument('--save-probs', dest='save_probs', action='store_true',
                        help='Save raw probabilities in addition to binary predictions')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=8,
                        help='Batch size for prediction')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config(args, config: Dict) -> Dict:
    """Merge command-line arguments with config file."""
    model_config = config.get('model', {})
    if args.window_size is not None:
        model_config['window_size'] = args.window_size
    if args.similarity_threshold is not None:
        model_config['patch_similarity_threshold'] = args.similarity_threshold
    if args.border_size is not None:
        model_config['border_size'] = args.border_size

    return {
        'model': model_config,
        'behaviors': config.get('behaviors', [])
    }


def load_model(checkpoint_path: str, config: Dict, num_behaviors: int, device: torch.device):
    """Load model from checkpoint."""
    model = create_model_from_config(config.get('config_path', 'configs/behaviors.yaml'), num_behaviors).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation macro_f1: {checkpoint.get('best_metric', 'unknown'):.4f}")

    return model


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Run prediction on dataset."""
    model.eval()
    all_probs = []
    all_preds = []
    all_metadata = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Predicting'):
            patch_embeddings = batch['patch_embeddings'].to(device)
            metadata = batch['metadata']

            logits = model(patch_embeddings)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)

            all_probs.append(probs)
            all_preds.append(preds)
            all_metadata.extend(metadata)

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)

    return all_probs, all_preds, all_metadata


def save_predictions(
    probs: np.ndarray,
    preds: np.ndarray,
    metadata: List[Dict],
    behavior_list: List[str],
    output_dir: Path,
    threshold: float,
    save_probs: bool = True
):
    """Save predictions to CSV and JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create DataFrame
    rows = []
    for i, meta in enumerate(metadata):
        row = {
            'video_idx': meta['video_idx'],
            'video_path': meta['video_path'],
            'window_start_frame': meta['window_start_frame'],
            'last_frame_idx': meta['last_frame_idx'],
        }

        # Add predictions for each behavior
        for j, behavior in enumerate(behavior_list):
            row[f'pred_{behavior}'] = int(preds[i, j])
            if save_probs:
                row[f'prob_{behavior}'] = float(probs[i, j])

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = output_dir / 'predictions.csv'
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")

    # Save JSON with more details
    json_data = {
        'threshold': threshold,
        'behaviors': behavior_list,
        'num_windows': len(metadata),
        'predictions': []
    }

    for i, meta in enumerate(metadata):
        pred_dict = {
            'window_idx': i,
            'video_idx': meta['video_idx'],
            'video_path': meta['video_path'],
            'window_start_frame': meta['window_start_frame'],
            'last_frame_idx': meta['last_frame_idx'],
            'probabilities': {behavior: float(probs[i, j]) for j, behavior in enumerate(behavior_list)},
            'predictions': {behavior: int(preds[i, j]) for j, behavior in enumerate(behavior_list)}
        }
        json_data['predictions'].append(pred_dict)

    json_path = output_dir / 'predictions.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Detailed predictions saved to: {json_path}")

    # Save per-video summary
    video_summaries = {}
    for i, meta in enumerate(metadata):
        vid_idx = meta['video_idx']
        vid_path = meta['video_path']
        if vid_idx not in video_summaries:
            video_summaries[vid_idx] = {
                'video_path': vid_path,
                'windows': [],
                'frame_predictions': {}
            }

        last_frame = meta['last_frame_idx']
        if last_frame >= 0:
            video_summaries[vid_idx]['frame_predictions'][last_frame] = {
                behavior: int(preds[i, j]) for j, behavior in enumerate(behavior_list)
            }
            video_summaries[vid_idx]['windows'].append({
                'window_start': meta['window_start_frame'],
                'last_frame': last_frame,
                'predictions': {behavior: int(preds[i, j]) for j, behavior in enumerate(behavior_list)}
            })

    summary_path = output_dir / 'video_summaries.json'
    with open(summary_path, 'w') as f:
        json.dump(video_summaries, f, indent=2)
    print(f"Video summaries saved to: {summary_path}")


def main():
    # Must set spawn method before any CUDA operations for DataLoader workers
    mp.set_start_method('spawn', force=True)
    # Use file system sharing strategy to avoid shared memory limits with CUDA tensors
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_args()

    # Load and merge config
    config = load_config(args.config)
    config = merge_config(args, config)
    config['config_path'] = args.config

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Load behavior list
    behavior_list = config['behaviors']
    num_behaviors = len(behavior_list)
    print(f"Behaviors ({num_behaviors}): {behavior_list}")

    # Initialize DINOv3 extractor
    dinov3_model = config['model'].get('dinov3_model', 'facebook/dinov3-vitb16-pretrain-lvd1689m')
    dinov3_extractor = DINOv3Extractor(model_name=dinov3_model, device=str(device))

    # Load model
    model = load_model(args.checkpoint, config, num_behaviors, device)

    # Load data config
    if args.data_config:
        # New approach: load from JSON file
        print(f"Loading data config from: {args.data_config}")
        data_config = load_data_config(args.data_config)
        # Use test split for prediction, fall back to val, then train
        predict_configs = data_config.test if data_config.test else (data_config.val if data_config.val else data_config.train)
        print(f"  Prediction videos: {len(predict_configs)}")
        for v in predict_configs:
            print(f"    Predict: {v.video_path} (ref: {v.reference_patch})")
    else:
        # Legacy approach: build from command-line args
        if not args.videos:
            parser.error("Either --data-config or --videos must be provided")

        print("Using legacy mode (single reference patch for all videos)")
        data_config = create_data_config_from_lists(
            train_videos=[], train_labels=[], train_ref_patches=[],
            val_videos=[], val_labels=[], val_ref_patches=[],
            test_videos=args.videos,
            test_labels=[''] * len(args.videos),  # No labels needed for prediction
            test_ref_patches=[tuple(args.reference_patch)] * len(args.videos)
        )
        predict_configs = data_config.test

    # Create prediction dataset
    combined_config = {**config['model'], **config.get('data', {})}
    dataset = PredictionDataset(
        video_configs=predict_configs,
        dinov3_extractor=dinov3_extractor,
        window_size=config['model'].get('window_size', 16),
        stride=args.stride,
        similarity_threshold=config['model'].get('patch_similarity_threshold', 0.80),
        border_size=config['model'].get('border_size', 1),
        max_patches_per_frame=args.max_patches_per_frame,
        frame_sample_rate=config['data'].get('frame_sample_rate', 100)
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Total prediction windows: {len(dataset)}")

    # Run prediction
    print("\nRunning prediction...")
    start_time = time.time()

    probs, preds, metadata = predict(model, loader, device, args.threshold)

    pred_time = time.time() - start_time
    print(f"Prediction completed in {pred_time:.2f}s ({len(dataset)/pred_time:.1f} windows/s)")

    # Save predictions
    save_predictions(
        probs, preds, metadata, behavior_list, output_dir, args.threshold, args.save_probs
    )

    # Print summary statistics
    print(f"\nPrediction Summary:")
    print(f"  Total windows: {len(preds)}")
    print(f"  Positive predictions per behavior:")
    for j, behavior in enumerate(behavior_list):
        pos_count = preds[:, j].sum()
        print(f"    {behavior}: {pos_count} ({pos_count/len(preds)*100:.1f}%)")

    # Save config used for prediction
    pred_config = {
        'checkpoint': args.checkpoint,
        'config': args.config,
        'data_config': args.data_config,
        'reference_patch': args.reference_patch,
        'window_size': config['model'].get('window_size', 16),
        'similarity_threshold': config['model'].get('patch_similarity_threshold', 0.80),
        'border_size': config['model'].get('border_size', 1),
        'threshold': args.threshold,
        'frame_sample_rate': config['data'].get('frame_sample_rate', 100),
        'stride': args.stride
    }
    with open(output_dir / 'prediction_config.yaml', 'w') as f:
        yaml.dump(pred_config, f)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()