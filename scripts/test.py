#!/usr/bin/env python3
"""
Test/Evaluation script for DINOv3-based Behavior Classification.

Usage (new JSON config approach):
    python scripts/test.py \
        --checkpoint outputs/experiment/checkpoint_best.pth \
        --config configs/behaviors.yaml \
        --data-config configs/data_config.json \
        --output-dir outputs/test_results

Usage (legacy approach - single reference patch for all videos):
    python scripts/test.py \
        --checkpoint outputs/experiment/checkpoint_best.pth \
        --config configs/behaviors.yaml \
        --test-videos data/test_video1.mp4 data/test_video2.mp4 \
        --test-labels data/test_labels1.json data/test_labels2.json \
        --reference-patch 7 7 \
        --output-dir outputs/test_results
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
    hamming_loss, accuracy_score, roc_auc_score
)
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.behavior_classifier import BehaviorClassifier, create_model_from_config
from utils.dinov3_extractor import DINOv3Extractor
from utils.dataset import BehaviorDataset
from utils.data_config import load_data_config, create_data_config_from_lists
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Test DINOv3-based Behavior Classifier')

    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/behaviors.yaml',
                        help='Path to config YAML file')

    # Data (new JSON config approach)
    parser.add_argument('--data-config', dest='data_config', type=str, default=None,
                        help='Path to JSON data config file with train/val/test splits and per-video reference patches')

    # Data (legacy approach - for backward compatibility)
    parser.add_argument('--test-videos', dest='test_videos', nargs='+', default=None,
                        help='Paths to test videos (legacy)')
    parser.add_argument('--test-labels', dest='test_labels', nargs='+', default=None,
                        help='Paths to test label files (legacy)')

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

    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold for multi-label')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=8,
                        help='Batch size for evaluation')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    # Output
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='outputs/test_results',
                        help='Output directory for test results')
    parser.add_argument('--plot-cm', dest='plot_cm', action='store_true',
                        help='Plot confusion matrices')

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config(args, config: Dict) -> Dict:
    model_config = config.get('model', {})
    if args.window_size is not None:
        model_config['window_size'] = args.window_size
    if args.similarity_threshold is not None:
        model_config['patch_similarity_threshold'] = args.similarity_threshold
    if args.border_size is not None:
        model_config['border_size'] = args.border_size

    return {
        'model': model_config,
        'training': config.get('training', {}),
        'data': config.get('data', {}),
        'behaviors': config.get('behaviors', [])
    }


def load_model(checkpoint_path: str, config: Dict, num_behaviors: int, device: torch.device):
    model = create_model_from_config(config.get('config_path', 'configs/behaviors.yaml'), num_behaviors).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Best validation macro_f1: {checkpoint.get('best_metric', 'unknown'):.4f}")

    return model


def compute_detailed_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    behavior_list: List[str],
    threshold: float = 0.5
) -> Dict:
    """Compute detailed multi-label classification metrics."""
    preds_binary = (predictions > threshold).astype(int)

    # Overall metrics
    macro_f1 = f1_score(targets, preds_binary, average='macro', zero_division=0)
    micro_f1 = f1_score(targets, preds_binary, average='micro', zero_division=0)
    weighted_f1 = f1_score(targets, preds_binary, average='weighted', zero_division=0)

    macro_precision = precision_score(targets, preds_binary, average='macro', zero_division=0)
    macro_recall = recall_score(targets, preds_binary, average='macro', zero_division=0)

    subset_accuracy = accuracy_score(targets, preds_binary)
    hamming = hamming_loss(targets, preds_binary)

    # Per-class metrics
    per_class_f1 = f1_score(targets, preds_binary, average=None, zero_division=0)
    per_class_precision = precision_score(targets, preds_binary, average=None, zero_division=0)
    per_class_recall = recall_score(targets, preds_binary, average=None, zero_division=0)

    # ROC AUC (if probabilities available)
    try:
        macro_auc = roc_auc_score(targets, predictions, average='macro')
        per_class_auc = roc_auc_score(targets, predictions, average=None)
    except ValueError:
        macro_auc = 0.0
        per_class_auc = np.zeros(len(behavior_list))

    metrics = {
        'overall': {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'weighted_f1': weighted_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'subset_accuracy': subset_accuracy,
            'hamming_loss': hamming,
            'macro_auc': macro_auc
        },
        'per_class': {}
    }

    for i, behavior in enumerate(behavior_list):
        metrics['per_class'][behavior] = {
            'f1': float(per_class_f1[i]),
            'precision': float(per_class_precision[i]),
            'recall': float(per_class_recall[i]),
            'auc': float(per_class_auc[i]) if i < len(per_class_auc) else 0.0,
            'support': int(targets[:, i].sum())
        }

    return metrics


def plot_confusion_matrices(
    predictions: np.ndarray,
    targets: np.ndarray,
    behavior_list: List[str],
    output_dir: Path,
    threshold: float = 0.5
):
    """Plot confusion matrices for each behavior."""
    preds_binary = (predictions > threshold).astype(int)
    n_behaviors = len(behavior_list)

    fig, axes = plt.subplots(2, (n_behaviors + 1) // 2, figsize=(5 * ((n_behaviors + 1) // 2), 10))
    axes = axes.flatten()

    for i, behavior in enumerate(behavior_list):
        if i >= len(axes):
            break

        cm = confusion_matrix(targets[:, i], preds_binary[:, i])

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Pred 0', 'Pred 1'],
                    yticklabels=['True 0', 'True 1'])
        axes[i].set_title(f'{behavior}')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')

    # Hide unused subplots
    for i in range(n_behaviors, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrices saved to: {output_dir / 'confusion_matrices.png'}")


def plot_per_class_metrics(metrics: Dict, behavior_list: List[str], output_dir: Path):
    """Plot per-class metrics as bar chart."""
    behaviors = behavior_list
    f1_scores = [metrics['per_class'][b]['f1'] for b in behaviors]
    precisions = [metrics['per_class'][b]['precision'] for b in behaviors]
    recalls = [metrics['per_class'][b]['recall'] for b in behaviors]
    aucs = [metrics['per_class'][b]['auc'] for b in behaviors]

    x = np.arange(len(behaviors))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(x - 1.5*width, f1_scores, width, label='F1', color='steelblue')
    ax.bar(x - 0.5*width, precisions, width, label='Precision', color='orange')
    ax.bar(x + 0.5*width, recalls, width, label='Recall', color='green')
    ax.bar(x + 1.5*width, aucs, width, label='AUC', color='red')

    ax.set_xlabel('Behavior')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Per-class metrics plot saved to: {output_dir / 'per_class_metrics.png'}")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """Evaluate model on test set."""
    model.eval()
    all_probs = []
    all_targets = []
    all_metadata = []

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            patch_embeddings = batch['patch_embeddings'].to(device)
            cls_tokens = batch.get('cls_tokens')
            if cls_tokens is not None:
                cls_tokens = cls_tokens.to(device)
            patch_mask = batch.get('patch_mask')
            if patch_mask is not None:
                patch_mask = patch_mask.to(device)
            labels = batch['labels'].to(device)
            metadata = batch['metadata']

            logits = model(patch_embeddings, cls_tokens=cls_tokens, patch_mask=patch_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(labels.cpu().numpy())
            all_metadata.extend(metadata)

    avg_loss = total_loss / len(loader)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    print(f"Test Loss: {avg_loss:.4f}")

    return all_probs, all_targets, all_metadata


def save_results(
    metrics: Dict,
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    metadata: List[Dict],
    behavior_list: List[str],
    output_dir: Path,
    threshold: float
):
    """Save detailed results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save detailed predictions
    rows = []
    for i, meta in enumerate(metadata):
        row = {
            'video_idx': meta['video_idx'],
            'video_path': meta.get('video_path', ''),
            'window_start_frame': meta['window_start_frame'],
            'last_frame_idx': meta['last_frame_idx'],
        }
        for j, behavior in enumerate(behavior_list):
            row[f'true_{behavior}'] = int(targets[i, j])
            row[f'pred_{behavior}'] = int(preds[i, j])
            row[f'prob_{behavior}'] = float(probs[i, j])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'detailed_predictions.csv', index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Macro F1:     {metrics['overall']['macro_f1']:.4f}")
    print(f"Micro F1:     {metrics['overall']['micro_f1']:.4f}")
    print(f"Weighted F1:  {metrics['overall']['weighted_f1']:.4f}")
    print(f"Macro Prec:   {metrics['overall']['macro_precision']:.4f}")
    print(f"Macro Rec:    {metrics['overall']['macro_recall']:.4f}")
    print(f"Subset Acc:   {metrics['overall']['subset_accuracy']:.4f}")
    print(f"Hamming Loss: {metrics['overall']['hamming_loss']:.4f}")
    print(f"Macro AUC:    {metrics['overall']['macro_auc']:.4f}")
    print(f"{'='*60}")
    print(f"\nPer-Class Results:")
    print(f"{'Behavior':<25} {'F1':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8} {'Support':>8}")
    print(f"{'-'*65}")
    for behavior in behavior_list:
        m = metrics['per_class'][behavior]
        print(f"{behavior:<25} {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['auc']:>8.4f} {m['support']:>8d}")


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
        test_configs = data_config.test if data_config.test else data_config.val
        print(f"  Test videos: {len(test_configs)}")
        for v in test_configs:
            print(f"    Test: {v.video_path} (ref: {v.reference_patch})")
    else:
        # Legacy approach: build from command-line args
        if not args.test_videos or not args.test_labels:
            parser.error("Either --data-config or both --test-videos and --test-labels must be provided")

        if len(args.test_videos) != len(args.test_labels):
            parser.error("Number of test videos must match number of test labels")

        print("Using legacy mode (single reference patch for all videos)")
        data_config = create_data_config_from_lists(
            train_videos=[], train_labels=[], train_ref_patches=[],
            val_videos=[], val_labels=[], val_ref_patches=[],
            test_videos=args.test_videos,
            test_labels=args.test_labels,
            test_ref_patches=[tuple(args.reference_patch)] * len(args.test_videos)
        )
        test_configs = data_config.test

    # Create test dataset
    combined_config = {**config['training'], **config['model'], **config['data']}
    test_dataset = BehaviorDataset(
        video_configs=test_configs,
        behavior_list=behavior_list,
        dinov3_extractor=dinov3_extractor,
        window_size=config['model'].get('window_size', 16),
        stride=args.stride,
        similarity_threshold=config['model'].get('patch_similarity_threshold', 0.80),
        border_size=config['model'].get('border_size', 1),
        max_patches_per_frame=args.max_patches_per_frame,
        frame_sample_rate=config['data'].get('frame_sample_rate', 100),
        mode='test'
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Test windows: {len(test_dataset)}")

    # Run evaluation
    print("\nRunning evaluation...")
    start_time = time.time()

    probs, targets, metadata = evaluate(model, test_loader, device, args.threshold)

    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f}s")

    # Compute metrics
    preds = (probs > args.threshold).astype(int)
    metrics = compute_detailed_metrics(probs, targets, behavior_list, args.threshold)

    # Save results
    save_results(metrics, probs, preds, targets, metadata, behavior_list, output_dir, args.threshold)

    # Plot if requested
    if args.plot_cm:
        plot_confusion_matrices(probs, targets, behavior_list, output_dir, args.threshold)
        plot_per_class_metrics(metrics, behavior_list, output_dir)

    # Save test config
    test_config = {
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
    with open(output_dir / 'test_config.yaml', 'w') as f:
        yaml.dump(test_config, f)

    print(f"\nAll test results saved to: {output_dir}")


if __name__ == '__main__':
    main()