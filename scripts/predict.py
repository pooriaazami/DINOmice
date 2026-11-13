#!/usr/bin/env python3
"""
Prediction script for DINOv3-based Behavior Classification.

Usage:
    python scripts/predict.py \
        --checkpoint outputs/experiment/checkpoint_best.pth \
        --config configs/behaviors.yaml \
        --videos data/test_video1.mp4 data/test_video2.mp4 \
        --reference_patch 7 7 \
        --window_size 16 \
        --similarity_threshold 0.8 \
        --border_size 1 \
        --output_dir outputs/predictions
"""

import argparse
import os
import sys
import yaml
import json
import time
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.behavior_classifier import BehaviorClassifier, create_model_from_config
from utils.dinov3_extractor import DINOv3Extractor
from utils.dataset import PredictionDataset, create_sliding_windows, pad_patch_sequence
from utils.dinov3_extractor import select_similar_patches
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Predict behaviors using trained DINOv3-based classifier')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/behaviors.yaml',
                        help='Path to config YAML file')

    # Input videos
    parser.add_argument('--videos', nargs='+', required=True,
                        help='Paths to videos for prediction')

    # Model parameters (must match training)
    parser.add_argument('--reference_patch', nargs=2, type=int, default=[7, 7],
                        help='Reference patch coordinates (row col)')
    parser.add_argument('--window_size', type=int, default=None,
                        help='Temporal window size (frames)')
    parser.add_argument('--similarity_threshold', type=float, default=None,
                        help='Similarity threshold for patch selection')
    parser.add_argument('--border_size', type=int, default=None,
                        help='Border size around selected patches')
    parser.add_argument('--max_patches_per_frame', type=int, default=50,
                        help='Maximum patches per frame after padding')
    parser.add_argument('--frame_sample_rate', type=int, default=1,
                        help='Sample every N frames')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride between windows')

    # Output
    parser.add_argument('--output_dir', type=str, default='outputs/predictions',
                        help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for multi-label')
    parser.add_argument('--save_probs', action='store_true',
                        help='Save raw probabilities in addition to binary predictions')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for prediction')
    parser.add_argument('--num_workers', type=int, default=4,
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

    # Create prediction dataset
    dataset = PredictionDataset(
        video_paths=args.videos,
        dinov3_extractor=dinov3_extractor,
        reference_patch_coords=tuple(args.reference_patch),
        window_size=config['model'].get('window_size', 16),
        stride=args.stride,
        similarity_threshold=config['model'].get('patch_similarity_threshold', 0.80),
        border_size=config['model'].get('border_size', 1),
        max_patches_per_frame=args.max_patches_per_frame,
        frame_sample_rate=args.frame_sample_rate
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
        'videos': args.videos,
        'reference_patch': args.reference_patch,
        'window_size': config['model'].get('window_size', 16),
        'similarity_threshold': config['model'].get('patch_similarity_threshold', 0.80),
        'border_size': config['model'].get('border_size', 1),
        'threshold': args.threshold,
        'frame_sample_rate': args.frame_sample_rate,
        'stride': args.stride
    }
    with open(output_dir / 'prediction_config.yaml', 'w') as f:
        yaml.dump(pred_config, f)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == '__main__':
    main()