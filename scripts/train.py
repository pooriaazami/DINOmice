#!/usr/bin/env python3
"""
Training script for DINOv3-based Behavior Classification.

Usage (new JSON config approach):
    python scripts/train.py \
        --config configs/behaviors.yaml \
        --data-config configs/data_config.json \
        --output-dir outputs/experiment1

Usage (legacy approach - single reference patch for all videos):
    python scripts/train.py \
        --config configs/behaviors.yaml \
        --train-videos data/video1.mp4 data/video2.mp4 \
        --train-labels data/labels1.json data/labels2.json \
        --val-videos data/val_video.mp4 \
        --val-labels data/val_labels.json \
        --reference-patch 7 7 \
        --output-dir outputs/experiment1

Test/Dry-run mode (single batch only):
    python scripts/train.py \
        --config configs/behaviors.yaml \
        --data-config configs/data_config.json \
        --output-dir outputs/test_run \
        --test
"""

import argparse
import os
import sys
import yaml
import json
import time
import traceback
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score, classification_report
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.behavior_classifier import BehaviorClassifier, create_model_from_config
from utils.dinov3_extractor import DINOv3Extractor
from utils.dataset import create_data_loaders_from_config, create_data_loaders
from utils.data_config import load_data_config, create_data_config_from_lists


def parse_args():
    parser = argparse.ArgumentParser(description='Train DINOv3-based Behavior Classifier')

    # Config
    parser.add_argument('--config', type=str, default='configs/behaviors.yaml',
                        help='Path to config YAML file')

    # Data (new JSON config approach)
    parser.add_argument('--data-config', dest='data_config', type=str, default=None,
                        help='Path to JSON data config file with train/val/test splits and per-video reference patches')

    # Data (legacy approach - for backward compatibility)
    parser.add_argument('--train-videos', dest='train_videos', nargs='+', default=None,
                        help='Paths to training videos (legacy)')
    parser.add_argument('--train-labels', dest='train_labels', nargs='+', default=None,
                        help='Paths to training label files (legacy)')
    parser.add_argument('--val-videos', dest='val_videos', nargs='+', default=None,
                        help='Paths to validation videos (legacy)')
    parser.add_argument('--val-labels', dest='val_labels', nargs='+', default=None,
                        help='Paths to validation label files (legacy)')

    # Model parameters (override config)
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

    # Training parameters
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--learning-rate', dest='learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--num-epochs', dest='num_epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=None,
                        help='Weight decay')
    parser.add_argument('--gradient-clip', dest='gradient_clip', type=float, default=None,
                        help='Gradient clipping value')
    parser.add_argument('--early-stopping-patience', dest='early_stopping_patience', type=int, default=None,
                        help='Early stopping patience')
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=None,
                        help='Number of data loader workers')

    # Output
    parser.add_argument('--output-dir', dest='output_dir', type=str, default='outputs/experiment',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, default=None,
                        help='Experiment name for TensorBoard')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')

    # DINOv3
    parser.add_argument('--dinov3-model', dest='dinov3_model', type=str, default=None,
                        help='DINOv3 model name')

    # Architecture
    parser.add_argument('--architecture', type=str, default=None, choices=['gru', 'transformer'],
                        help='Model architecture: gru (hierarchical) or transformer (flat)')

    # Test/Dry-run mode
    parser.add_argument('--test', '--dry-run', dest='test_mode', action='store_true',
                        help='Test mode: run only 1 batch for training and validation to verify code works')

    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_config(args, config: Dict) -> Dict:
    """Merge command-line arguments with config file."""
    # Model config
    model_config = config.get('model', {})
    if args.window_size is not None:
        model_config['window_size'] = args.window_size
    if args.similarity_threshold is not None:
        model_config['patch_similarity_threshold'] = args.similarity_threshold
    if args.border_size is not None:
        model_config['border_size'] = args.border_size
    if args.dinov3_model is not None:
        model_config['dinov3_model'] = args.dinov3_model
    if args.architecture is not None:
        model_config['architecture'] = args.architecture

    # Training config
    train_config = config.get('training', {})
    if args.batch_size is not None:
        train_config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        train_config['learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        train_config['num_epochs'] = args.num_epochs
    if args.weight_decay is not None:
        train_config['weight_decay'] = args.weight_decay
    if args.gradient_clip is not None:
        train_config['gradient_clip'] = args.gradient_clip
    if args.early_stopping_patience is not None:
        train_config['early_stopping_patience'] = args.early_stopping_patience
    if args.num_workers is not None:
        train_config['num_workers'] = args.num_workers

    # Data config
    data_config = config.get('data', {})
    if args.frame_sample_rate is not None:
        data_config['frame_sample_rate'] = args.frame_sample_rate

    return {
        'model': model_config,
        'training': train_config,
        'data': data_config,
        'behaviors': config.get('behaviors', [])
    }


def compute_metrics(predictions: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict:
    """Compute multi-label classification metrics."""
    preds_binary = (predictions > threshold).astype(int)

    # Macro F1
    macro_f1 = f1_score(targets, preds_binary, average='macro', zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(targets, preds_binary, average=None, zero_division=0)

    # Subset accuracy (exact match)
    subset_acc = np.mean(np.all(preds_binary == targets, axis=1))

    # Hamming loss
    hamming_loss_val = np.mean(preds_binary != targets)

    return {
        'macro_f1': macro_f1,
        'per_class_f1': per_class_f1,
        'subset_accuracy': subset_acc,
        'hamming_loss': hamming_loss_val
    }


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    gradient_clip: float = 1.0,
    epoch: int = 0,
    writer: SummaryWriter = None,
    global_step: int = 0,
    test_mode: bool = False
) -> Tuple[float, Dict, int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        patch_embeddings = batch['patch_embeddings'].to(device)  # (B, W, P, D)
        cls_tokens = batch.get('cls_tokens')
        if cls_tokens is not None:
            cls_tokens = cls_tokens.to(device)
        patch_mask = batch.get('patch_mask')
        if patch_mask is not None:
            patch_mask = patch_mask.to(device)
        labels = batch['labels'].to(device)  # (B, num_behaviors)

        optimizer.zero_grad()

        logits = model(patch_embeddings, cls_tokens=cls_tokens, patch_mask=patch_mask)
        loss = criterion(logits, labels)

        loss.backward()

        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        total_loss += loss.item()

        # Batch-level TensorBoard logging
        if writer is not None:
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            # Log learning rate at batch level too
            writer.add_scalar('LearningRate/batch', optimizer.param_groups[0]['lr'], global_step)

        # Store predictions
        preds = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.detach().cpu().numpy())

        pbar.set_postfix({'loss': loss.item()})

        global_step += 1

        # Test mode: only process one batch
        if test_mode:
            break

    avg_loss = total_loss / (batch_idx + 1)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss

    return avg_loss, metrics, global_step


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    writer: SummaryWriter = None,
    global_step: int = 0,
    test_mode: bool = False
) -> Tuple[float, Dict, int]:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, batch in enumerate(pbar):
            patch_embeddings = batch['patch_embeddings'].to(device)
            cls_tokens = batch.get('cls_tokens')
            if cls_tokens is not None:
                cls_tokens = cls_tokens.to(device)
            patch_mask = batch.get('patch_mask')
            if patch_mask is not None:
                patch_mask = patch_mask.to(device)
            labels = batch['labels'].to(device)

            logits = model(patch_embeddings, cls_tokens=cls_tokens, patch_mask=patch_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            # Batch-level TensorBoard logging for validation
            if writer is not None:
                writer.add_scalar('Loss/val_batch', loss.item(), global_step)

            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

            global_step += 1

            # Test mode: only process one batch
            if test_mode:
                break

    avg_loss = total_loss / (batch_idx + 1)
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = avg_loss

    return avg_loss, metrics, global_step


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    best_metric: float,
    config: Dict,
    output_dir: Path,
    is_best: bool = False,
    max_retries: int = 3,
    retry_delay: float = 1.0
):
    """Save model checkpoint with retry logic for I/O errors."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'config': config
    }

    # Save latest with retry
    for attempt in range(max_retries):
        try:
            torch.save(checkpoint, output_dir / 'checkpoint_latest.pth')
            break
        except (IOError, OSError, RuntimeError) as e:
            if attempt < max_retries - 1:
                print(f"  Warning: Checkpoint save failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"  Error: Failed to save latest checkpoint after {max_retries} attempts: {e}")
                traceback.print_exc()

    # Save best with retry
    if is_best:
        for attempt in range(max_retries):
            try:
                torch.save(checkpoint, output_dir / 'checkpoint_best.pth')
                print(f"  Saved best model (macro_f1: {best_metric:.4f})")
                break
            except (IOError, OSError, RuntimeError) as e:
                if attempt < max_retries - 1:
                    print(f"  Warning: Best checkpoint save failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"  Error: Failed to save best checkpoint after {max_retries} attempts: {e}")
                    traceback.print_exc()


def main():
    # Must set spawn method before any CUDA operations for DataLoader workers
    mp.set_start_method('spawn', force=True)
    # Use file system sharing strategy to avoid shared memory limits with CUDA tensors
    torch.multiprocessing.set_sharing_strategy('file_system')

    args = parse_args()

    # Load and merge config
    config = load_config(args.config)
    config = merge_config(args, config)

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Setup TensorBoard
    experiment_name = args.experiment_name or output_dir.name
    writer = SummaryWriter(log_dir=output_dir / 'tensorboard' / experiment_name)

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

    # Create model
    model = create_model_from_config(args.config, num_behaviors).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data config
    if args.data_config:
        # New approach: load from JSON file
        print(f"Loading data config from: {args.data_config}")
        data_config = load_data_config(args.data_config)
        print(f"  Train videos: {len(data_config.train)}")
        print(f"  Val videos: {len(data_config.val)}")
        print(f"  Test videos: {len(data_config.test)}")
        for v in data_config.train:
            print(f"    Train: {v.video_path} (ref: {v.reference_patch})")
        for v in data_config.val:
            print(f"    Val:   {v.video_path} (ref: {v.reference_patch})")
    else:
        # Legacy approach: build from command-line args
        if not args.train_videos or not args.train_labels or not args.val_videos or not args.val_labels:
            parser.error("Either --data-config or all of --train-videos, --train-labels, --val-videos, --val-labels must be provided")

        if len(args.train_videos) != len(args.train_labels):
            parser.error("Number of train videos must match number of train labels")
        if len(args.val_videos) != len(args.val_labels):
            parser.error("Number of val videos must match number of val labels")

        print("Using legacy mode (single reference patch for all videos)")
        data_config = create_data_config_from_lists(
            train_videos=args.train_videos,
            train_labels=args.train_labels,
            train_ref_patches=[tuple(args.reference_patch)] * len(args.train_videos),
            val_videos=args.val_videos,
            val_labels=args.val_labels,
            val_ref_patches=[tuple(args.reference_patch)] * len(args.val_videos)
        )

    # Create data loaders
    combined_config = {**config['training'], **config['model'], **config['data']}
    train_loader, val_loader = create_data_loaders_from_config(
        data_config=data_config,
        behavior_list=behavior_list,
        dinov3_extractor=dinov3_extractor,
        config=combined_config,
        cache_dir=args.output_dir + '/cache' if args.output_dir else None
    )

    print(f"Train windows: {len(train_loader.dataset)}")
    print(f"Val windows: {len(val_loader.dataset)}")

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # Ensure numeric config values are floats (YAML may parse as strings if quoted)
    lr = float(config['training'].get('learning_rate', 1e-4))
    weight_decay = float(config['training'].get('weight_decay', 1e-5))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_macro_f1 = 0.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_macro_f1 = checkpoint.get('best_metric', 0.0)
        print(f"Resumed from epoch {start_epoch}, best macro_f1: {best_macro_f1:.4f}")

    # Training loop
    num_epochs = int(config['training'].get('num_epochs', 50))
    early_stopping_patience = int(config['training'].get('early_stopping_patience', 10))
    gradient_clip = float(config['training'].get('gradient_clip', 1.0))

    patience_counter = 0
    global_step = 0

    # Test mode: run only 1 epoch with 1 batch each
    if args.test_mode:
        num_epochs = 1
        print(f"\n*** TEST/DRY-RUN MODE: Running only 1 epoch with 1 batch per phase ***")

    print(f"\nStarting training for {num_epochs} epochs...")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_metrics, global_step = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=gradient_clip, epoch=epoch,
            writer=writer, global_step=global_step, test_mode=args.test_mode
        )

        # Validate
        val_loss, val_metrics, global_step = validate(
            model, val_loader, criterion, device, epoch=epoch,
            writer=writer, global_step=global_step, test_mode=args.test_mode
        )

        epoch_time = time.time() - epoch_start

        # Log epoch-level metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('MacroF1/train', train_metrics['macro_f1'], epoch)
        writer.add_scalar('MacroF1/val', val_metrics['macro_f1'], epoch)
        writer.add_scalar('SubsetAcc/train', train_metrics['subset_accuracy'], epoch)
        writer.add_scalar('SubsetAcc/val', val_metrics['subset_accuracy'], epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Per-class F1
        for i, (behavior, f1) in enumerate(zip(behavior_list, val_metrics['per_class_f1'])):
            writer.add_scalar(f'PerClassF1/val/{behavior}', f1, epoch)

        # Print progress
        print(f"Epoch {epoch:3d} | "
              f"Train: Loss={train_loss:.4f} MacroF1={train_metrics['macro_f1']:.4f} SubsetAcc={train_metrics['subset_accuracy']:.4f} | "
              f"Val: Loss={val_loss:.4f} MacroF1={val_metrics['macro_f1']:.4f} SubsetAcc={val_metrics['subset_accuracy']:.4f} | "
              f"Time={epoch_time:.1f}s")

        # Learning rate scheduling
        scheduler.step(val_metrics['macro_f1'])

        # Check for improvement
        is_best = val_metrics['macro_f1'] > best_macro_f1
        if is_best:
            best_macro_f1 = val_metrics['macro_f1']
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint (skip in test mode unless it's the best)
        if not args.test_mode or is_best:
            save_checkpoint(
                model, optimizer, epoch, best_macro_f1, config, output_dir, is_best
            )

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        # Test mode: exit after first epoch
        if args.test_mode:
            print(f"\n*** TEST/DRY-RUN COMPLETE ***")
            break

    writer.close()
    print(f"\nTraining complete! Best validation macro_f1: {best_macro_f1:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()