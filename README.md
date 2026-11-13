# DINOv3-based Behavior Classification Pipeline

A PyTorch-based pipeline for multi-label behavior classification from videos using DINOv3 visual features and hierarchical GRU architecture.

## Pipeline Overview

```
Video Frames → DINOv3 → Patch Features → Patch Selection → Patch GRU → Frame Embeddings
                                                                    ↓
                                                         Window GRU → Window Embedding
                                                                    ↓
                                                          FC Layers → Multi-label Output
```

### Key Components

1. **DINOv3 Feature Extractor**: Extracts patch-level embeddings (14×14 grid, 768-dim) for each frame
2. **Patch Selection**: For each frame, selects patches similar to a reference patch (from first frame) with configurable similarity threshold and border expansion
3. **Patch GRU**: Processes selected patch embeddings in spatial order to create frame-level embeddings
4. **Window GRU**: Processes sequence of frame embeddings over a temporal window
5. **Classification Head**: Fully connected layers for multi-label behavior prediction

## Installation

```bash
cd DINO_based_behavior_classifier
pip install -r requirements.txt
```

## Configuration

Edit `configs/behaviors.yaml` to define:
- List of behavior classes
- Model hyperparameters (GRU sizes, dropout, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- Data parameters (frame sample rate, image size)

## Label Format

Labels are provided frame-wise. Supported formats:

### JSON (recommended)
```json
{
  "0": ["grooming"],
  "1": ["grooming", "eating"],
  "2": ["eating"],
  "3": ["resting"]
}
```

### CSV
```csv
frame_idx,behavior
0,grooming
1,grooming
1,eating
2,eating
3,resting
```

### Text (one line per frame, comma-separated)
```
grooming
grooming,eating
eating
resting
```

## Training

```bash
python scripts/train.py \
    --config configs/behaviors.yaml \
    --train_videos data/video1.mp4 data/video2.mp4 \
    --train_labels data/labels1.json data/labels2.json \
    --val_videos data/val_video.mp4 \
    --val_labels data/val_labels.json \
    --reference_patch 7 7 \
    --window_size 16 \
    --similarity_threshold 0.8 \
    --border_size 1 \
    --output_dir outputs/experiment1 \
    --experiment_name exp1
```

### Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--reference_patch` | Reference patch coordinates (row col) | 7 7 |
| `--window_size` | Temporal window size in frames | 16 (from config) |
| `--similarity_threshold` | Patch similarity threshold (0-1) | 0.8 (from config) |
| `--border_size` | Border around selected patches | 1 (from config) |
| `--batch_size` | Training batch size | 4 (from config) |
| `--learning_rate` | Learning rate | 1e-4 (from config) |
| `--num_epochs` | Number of epochs | 50 (from config) |
| `--early_stopping_patience` | Early stopping patience | 10 (from config) |

### Video-Level Validation Split

The pipeline uses **video-level split** for validation (not random frame split):
- Training videos: `--train_videos`
- Validation videos: `--val_videos`
- Each video is processed independently with its own labels

## Prediction

```bash
python scripts/predict.py \
    --checkpoint outputs/experiment1/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --videos data/test_video1.mp4 data/test_video2.mp4 \
    --reference_patch 7 7 \
    --window_size 16 \
    --similarity_threshold 0.8 \
    --border_size 1 \
    --output_dir outputs/predictions \
    --threshold 0.5 \
    --save_probs
```

### Output Files

- `predictions.csv`: Frame-level predictions with probabilities
- `predictions.json`: Detailed predictions with metadata
- `video_summaries.json`: Per-video summary of predictions

## Testing/Evaluation

```bash
python scripts/test.py \
    --checkpoint outputs/experiment1/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --test_videos data/test_video1.mp4 data/test_video2.mp4 \
    --test_labels data/test_labels1.json data/test_labels2.json \
    --reference_patch 7 7 \
    --window_size 16 \
    --similarity_threshold 0.8 \
    --border_size 1 \
    --output_dir outputs/test_results \
    --threshold 0.5 \
    --plot_cm
```

### Metrics Computed

- **Macro F1** (primary metric)
- Micro F1, Weighted F1
- Macro Precision, Macro Recall
- Subset Accuracy (exact match)
- Hamming Loss
- Macro AUC
- Per-class F1, Precision, Recall, AUC

### Visualizations (with `--plot_cm`)

- Confusion matrices for each behavior
- Per-class metrics bar chart

## Architecture Details

### Model: `models/behavior_classifier.py`

- **PatchGRU**: Bidirectional GRU processing patches in spatial (row-major) order
- **WindowGRU**: Bidirectional GRU processing frame embeddings temporally
- **Classifier**: 3-layer MLP with dropout and ReLU

### Feature Extraction: `utils/dinov3_extractor.py`

- DINOv3 ViT-B/16 (patch size 16)
- Features normalized to unit length
- Similarity = cosine similarity (dot product of normalized vectors)
- Patch selection with border expansion

### Dataset: `utils/dataset.py`

- `BehaviorDataset`: For training/validation with labels
- `PredictionDataset`: For inference without labels
- Sliding window creation with configurable stride
- Variable patch count per frame handled via padding + masking

## Example Usage

### 1. Prepare Data
```
data/
├── train_video1.mp4
├── train_video2.mp4
├── val_video.mp4
├── test_video.mp4
├── train_labels1.json
├── train_labels2.json
├── val_labels.json
└── test_labels.json
```

### 2. Configure Behaviors
Edit `configs/behaviors.yaml`:
```yaml
behaviors:
  - "grooming"
  - "eating"
  - "drinking"
  - "rearing"
  - "walking"
  - "resting"
  - "exploring"
```

### 3. Train
```bash
python scripts/train.py \
    --config configs/behaviors.yaml \
    --train_videos data/train_video1.mp4 data/train_video2.mp4 \
    --train_labels data/train_labels1.json data/train_labels2.json \
    --val_videos data/val_video.mp4 \
    --val_labels data/val_labels.json \
    --reference_patch 7 7 \
    --output_dir outputs/my_experiment
```

### 4. Monitor with TensorBoard
```bash
tensorboard --logdir outputs/my_experiment/tensorboard
```

### 5. Test
```bash
python scripts/test.py \
    --checkpoint outputs/my_experiment/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --test_videos data/test_video.mp4 \
    --test_labels data/test_labels.json \
    --reference_patch 7 7 \
    --output_dir outputs/test_results \
    --plot_cm
```

### 6. Predict on New Videos
```bash
python scripts/predict.py \
    --checkpoint outputs/my_experiment/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --videos data/new_video.mp4 \
    --reference_patch 7 7 \
    --output_dir outputs/new_predictions
```

## Reference Notebook

See `2_DINO_sim.ipynb` for interactive exploration of:
- DINOv3 feature extraction
- Patch similarity computation
- Patch selection with border
- Visualization of selected regions

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- transformers, accelerate (for DINOv3)

## Citation

If you use this pipeline, please cite DINOv3:
```
@article{dinov3,
  title={DINOv3: ...},
  author={...},
  journal={...},
  year={2024}
}
```