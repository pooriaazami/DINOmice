#!/usr/bin/env python3
"""
Main entry point for DINOv3-based Behavior Classification Pipeline.

This script provides a unified CLI for training, testing, and prediction.

Usage (new JSON config approach):
    python main.py train --config configs/behaviors.yaml --data_config configs/data_config.json --output_dir outputs/exp1
    python main.py test --checkpoint outputs/exp1/checkpoint_best.pth --config configs/behaviors.yaml --data_config configs/data_config.json --output_dir outputs/test_results
    python main.py predict --checkpoint outputs/exp1/checkpoint_best.pth --config configs/behaviors.yaml --data_config configs/data_config.json --output_dir outputs/predictions

Usage (legacy approach):
    python main.py train --config configs/behaviors.yaml --train_videos data/v1.mp4 data/v2.mp4 --train_labels data/l1.json data/l2.json --val_videos data/val.mp4 --val_labels data/val_labels.json --output_dir outputs/exp1
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='DINOv3-based Behavior Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (new JSON config approach - RECOMMENDED):
  # Train
  python main.py train --config configs/behaviors.yaml \\
      --data_config configs/data_config.json \\
      --output_dir outputs/exp1

  # Test
  python main.py test --checkpoint outputs/exp1/checkpoint_best.pth \\
      --config configs/behaviors.yaml \\
      --data_config configs/data_config.json \\
      --output_dir outputs/test_results

  # Predict
  python main.py predict --checkpoint outputs/exp1/checkpoint_best.pth \\
      --config configs/behaviors.yaml \\
      --data_config configs/data_config.json \\
      --output_dir outputs/predictions

Examples (legacy approach - single reference patch for all videos):
  # Train
  python main.py train --config configs/behaviors.yaml \\
      --train_videos data/v1.mp4 data/v2.mp4 \\
      --train_labels data/l1.json data/l2.json \\
      --val_videos data/val.mp4 \\
      --val_labels data/val_labels.json \\
      --reference_patch 7 7 \\
      --output_dir outputs/exp1

  # Test
  python main.py test --checkpoint outputs/exp1/checkpoint_best.pth \\
      --config configs/behaviors.yaml \\
      --test_videos data/test.mp4 \\
      --test_labels data/test_labels.json \\
      --reference_patch 7 7 \\
      --output_dir outputs/test_results

  # Predict
  python main.py predict --checkpoint outputs/exp1/checkpoint_best.pth \\
      --config configs/behaviors.yaml \\
      --videos data/new_video.mp4 \\
      --reference_patch 7 7 \\
      --output_dir outputs/predictions
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='configs/behaviors.yaml')
    train_parser.add_argument('--data_config', type=str, default=None,
                              help='Path to JSON data config file with train/val/test splits and per-video reference patches (RECOMMENDED)')
    # Legacy args
    train_parser.add_argument('--train_videos', nargs='+', default=None,
                              help='Paths to training videos (legacy)')
    train_parser.add_argument('--train_labels', nargs='+', default=None,
                              help='Paths to training label files (legacy)')
    train_parser.add_argument('--val_videos', nargs='+', default=None,
                              help='Paths to validation videos (legacy)')
    train_parser.add_argument('--val_labels', nargs='+', default=None,
                              help='Paths to validation label files (legacy)')
    train_parser.add_argument('--reference_patch', nargs=2, type=int, default=[7, 7],
                              help='Reference patch coordinates (row col) - used for all videos in legacy mode')
    train_parser.add_argument('--window_size', type=int, default=None)
    train_parser.add_argument('--similarity_threshold', type=float, default=None)
    train_parser.add_argument('--border_size', type=int, default=None)
    train_parser.add_argument('--max_patches_per_frame', type=int, default=50)
    train_parser.add_argument('--frame_sample_rate', type=int, default=None)
    train_parser.add_argument('--batch_size', type=int, default=None)
    train_parser.add_argument('--learning_rate', type=float, default=None)
    train_parser.add_argument('--num_epochs', type=int, default=None)
    train_parser.add_argument('--weight_decay', type=float, default=None)
    train_parser.add_argument('--gradient_clip', type=float, default=None)
    train_parser.add_argument('--early_stopping_patience', type=int, default=None)
    train_parser.add_argument('--num_workers', type=int, default=None)
    train_parser.add_argument('--output_dir', type=str, default='outputs/experiment')
    train_parser.add_argument('--experiment_name', type=str, default=None)
    train_parser.add_argument('--resume', type=str, default=None)
    train_parser.add_argument('--device', type=str, default=None)
    train_parser.add_argument('--dinov3_model', type=str, default=None)
    train_parser.add_argument('--test', '--dry-run', action='store_true',
                              help='Test mode: run only 1 batch for training and validation to verify code works')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test/evaluate the model')
    test_parser.add_argument('--checkpoint', type=str, required=True)
    test_parser.add_argument('--config', type=str, default='configs/behaviors.yaml')
    test_parser.add_argument('--data_config', type=str, default=None,
                             help='Path to JSON data config file with train/val/test splits and per-video reference patches (RECOMMENDED)')
    # Legacy args
    test_parser.add_argument('--test_videos', nargs='+', default=None,
                             help='Paths to test videos (legacy)')
    test_parser.add_argument('--test_labels', nargs='+', default=None,
                             help='Paths to test label files (legacy)')
    test_parser.add_argument('--reference_patch', nargs=2, type=int, default=[7, 7],
                             help='Reference patch coordinates (row col) - used for all videos in legacy mode')
    test_parser.add_argument('--window_size', type=int, default=None)
    test_parser.add_argument('--similarity_threshold', type=float, default=None)
    test_parser.add_argument('--border_size', type=int, default=None)
    test_parser.add_argument('--max_patches_per_frame', type=int, default=50)
    test_parser.add_argument('--frame_sample_rate', type=int, default=None)
    test_parser.add_argument('--stride', type=int, default=1)
    test_parser.add_argument('--threshold', type=float, default=0.5)
    test_parser.add_argument('--batch_size', type=int, default=8)
    test_parser.add_argument('--num_workers', type=int, default=4)
    test_parser.add_argument('--device', type=str, default=None)
    test_parser.add_argument('--output_dir', type=str, default='outputs/test_results')
    test_parser.add_argument('--plot_cm', action='store_true')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict on new videos')
    pred_parser.add_argument('--checkpoint', type=str, required=True)
    pred_parser.add_argument('--config', type=str, default='configs/behaviors.yaml')
    pred_parser.add_argument('--data_config', type=str, default=None,
                             help='Path to JSON data config file with train/val/test splits and per-video reference patches (RECOMMENDED)')
    # Legacy args
    pred_parser.add_argument('--videos', nargs='+', default=None,
                             help='Paths to videos for prediction (legacy)')
    pred_parser.add_argument('--reference_patch', nargs=2, type=int, default=[7, 7],
                             help='Reference patch coordinates (row col) - used for all videos in legacy mode')
    pred_parser.add_argument('--window_size', type=int, default=None)
    pred_parser.add_argument('--similarity_threshold', type=float, default=None)
    pred_parser.add_argument('--border_size', type=int, default=None)
    pred_parser.add_argument('--max_patches_per_frame', type=int, default=50)
    pred_parser.add_argument('--frame_sample_rate', type=int, default=None)
    pred_parser.add_argument('--stride', type=int, default=1)
    pred_parser.add_argument('--threshold', type=float, default=0.5)
    pred_parser.add_argument('--batch_size', type=int, default=8)
    pred_parser.add_argument('--num_workers', type=int, default=4)
    pred_parser.add_argument('--device', type=str, default=None)
    pred_parser.add_argument('--output_dir', type=str, default='outputs/predictions')
    pred_parser.add_argument('--save_probs', action='store_true')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Import and run appropriate script
    if args.command == 'train':
        from scripts.train import main as train_main
        sys.argv = ['train.py'] + _args_to_argv(args, 'train')
        train_main()
    elif args.command == 'test':
        from scripts.test import main as test_main
        sys.argv = ['test.py'] + _args_to_argv(args, 'test')
        test_main()
    elif args.command == 'predict':
        from scripts.predict import main as predict_main
        sys.argv = ['predict.py'] + _args_to_argv(args, 'predict')
        predict_main()


def _args_to_argv(args, command):
    """Convert parsed args back to argv format for sub-scripts."""
    argv = []

    # Common args
    common_args = ['config', 'data_config', 'reference_patch', 'window_size', 'similarity_threshold',
                   'border_size', 'max_patches_per_frame', 'frame_sample_rate',
                   'device', 'output_dir']

    if command == 'train':
        specific_args = ['train_videos', 'train_labels', 'val_videos', 'val_labels',
                         'batch_size', 'learning_rate', 'num_epochs', 'weight_decay',
                         'gradient_clip', 'early_stopping_patience', 'num_workers',
                         'experiment_name', 'resume', 'dinov3_model', 'test']
    elif command == 'test':
        specific_args = ['checkpoint', 'test_videos', 'test_labels', 'stride',
                         'threshold', 'batch_size', 'num_workers', 'plot_cm']
    elif command == 'predict':
        specific_args = ['checkpoint', 'videos', 'stride', 'threshold',
                         'batch_size', 'num_workers', 'save_probs']

    for arg_name in common_args + specific_args:
        value = getattr(args, arg_name, None)
        if value is not None:
            if isinstance(value, list):
                argv.extend([f'--{arg_name.replace("_", "-")}'] + [str(v) for v in value])
            elif isinstance(value, bool):
                if value:
                    argv.append(f'--{arg_name.replace("_", "-")}')
            else:
                argv.extend([f'--{arg_name.replace("_", "-")}', str(value)])

    return argv


if __name__ == '__main__':
    main()