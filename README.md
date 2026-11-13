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

### 1. Model & Training Config: `configs/behaviors.yaml`

Edit to define:
- List of behavior classes
- Model hyperparameters (GRU sizes, dropout, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- Data parameters (frame sample rate)

```yaml
behaviors:
  - "grooming"
  - "eating"
  - "drinking"
  - "rearing"
  - "walking"
  - "resting"
  - "exploring"

model:
  dinov3_model: "facebook/dinov3-vitb16-pretrain-lvd1689m"
  window_size: 16
  patch_similarity_threshold: 0.80
  border_size: 1
  patch_gru_hidden: 256
  window_gru_hidden: 512
  fc_hidden: 512
  dropout: 0.3
  bidirectional: true

training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 50
  weight_decay: 1e-5
  gradient_clip: 1.0
  early_stopping_patience: 10

data:
  frame_sample_rate: 100
```

### 2. Data Config: `configs/data_config.json` (RECOMMENDED)

New JSON-based configuration for per-video reference patches and video-level splits:

```json
{
  "train": [
    {
      "video_path": "data/train_video1.mp4",
      "label_path": "data/train_labels1.json",
      "reference_patch": [7, 7]
    },
    {
      "video_path": "data/train_video2.mp4",
      "label_path": "data/train_labels2.json",
      "reference_patch": [6, 8]
    }
  ],
  "val": [
    {
      "video_path": "data/val_video.mp4",
      "label_path": "data/val_labels.json",
      "reference_patch": [7, 7]
    }
  ],
  "test": [
    {
      "video_path": "data/test_video.mp4",
      "label_path": "data/test_labels.json",
      "reference_patch": [7, 7]
    }
  ]
}
```

**Key advantage**: Each video can have its own reference patch coordinates!

See `configs/data_config_example.json` for a complete template.

### 3. Label Format

Labels are provided frame-wise. Supported formats:

**JSON (recommended)**
```json
{
  "0": ["grooming"],
  "1": ["grooming", "eating"],
  "2": ["eating"],
  "3": ["resting"]
}
```

**CSV**
```csv
frame_idx,behavior
0,grooming
1,grooming
1,eating
2,eating
3,resting
```

**Text (one line per frame, comma-separated)**
```
grooming
grooming,eating
eating
resting
```

## Quick Start (Recommended: JSON Config)

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

### 2. Create Data Config
Copy `configs/data_config_example.json` to `configs/data_config.json` and edit with your paths and per-video reference patches.

### 3. Train
```bash
# Using main.py (unified CLI)
python main.py train \
    --config configs/behaviors.yaml \
    --data-config configs/data_config.json \
    --output-dir outputs/my_experiment \
    --experiment-name my_experiment

# Or directly with train.py
python scripts/train.py \
    --config configs/behaviors.yaml \
    --data-config configs/data_config.json \
    --output-dir outputs/my_experiment \
    --experiment-name my_experiment
```

### 4. Monitor with TensorBoard
```bash
tensorboard --logdir outputs/my_experiment/tensorboard
```
Wait for at least 1 epoch to complete before event files appear.

### 5. Test/Evaluate
```bash
# Using main.py
python main.py test \
    --checkpoint outputs/my_experiment/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --data-config configs/data_config.json \
    --output-dir outputs/test_results \
    --plot-cm

# Or directly with test.py
python scripts/test.py \
    --checkpoint outputs/my_experiment/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --data-config configs/data_config.json \
    --output-dir outputs/test_results \
    --plot-cm
```

### 6. Predict on New Videos
```bash
# Using main.py
python main.py predict \
    --checkpoint outputs/my_experiment/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --data-config configs/data_config.json \
    --output-dir outputs/predictions \
    --save-probs

# Or directly with predict.py
python scripts/predict.py \
    --checkpoint outputs/my_experiment/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --data-config configs/data_config.json \
    --output-dir outputs/predictions \
    --save-probs
```

---

## Legacy CLI Approach (Single Reference Patch for All Videos)

If you prefer not to use the JSON config, you can use the legacy CLI arguments. Note: **all videos must share the same reference patch**.

### Training (Legacy)
```bash
python main.py train \
    --config configs/behaviors.yaml \
    --train-videos data/train_video1.mp4 data/train_video2.mp4 \
    --train-labels data/train_labels1.json data/train_labels2.json \
    --val-videos data/val_video.mp4 \
    --val-labels data/val_labels.json \
    --reference-patch 7 7 \
    --output-dir outputs/my_experiment
```

### Testing (Legacy)
```bash
python main.py test \
    --checkpoint outputs/my_experiment/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --test-videos data/test_video.mp4 \
    --test-labels data/test_labels.json \
    --reference-patch 7 7 \
    --output-dir outputs/test_results \
    --plot-cm
```

### Prediction (Legacy)
```bash
python main.py predict \
    --checkpoint outputs/my_experiment/checkpoint_best.pth \
    --config configs/behaviors.yaml \
    --videos data/new_video.mp4 \
    --reference-patch 7 7 \
    --output-dir outputs/predictions \
    --save-probs
```

---

## CLI Argument Reference

All scripts now use **hyphenated flags** (e.g., `--data-config`, `--reference-patch`) with underscore attributes internally.

### Common Arguments
| Flag | Description |
|------|-------------|
| `--config` | Path to behaviors.yaml |
| `--data-config` | Path to data_config.json (recommended) |
| `--output-dir` | Output directory |
| `--device` | `cuda` or `cpu` (auto-detected) |

### Model Override Arguments
| Flag | Description |
|------|-------------|
| `--reference-patch` | Reference patch row col (legacy only) |
| `--window-size` | Temporal window size |
| `--similarity-threshold` | Patch similarity threshold (0-1) |
| `--border-size` | Border size around selected patches |
| `--max-patches-per-frame` | Max patches per frame after padding |
| `--frame-sample-rate` | Sample every N frames |

### Training Specific
| Flag | Description |
|------|-------------|
| `--train-videos` | Training video paths (legacy) |
| `--train-labels` | Training label paths (legacy) |
| `--val-videos` | Validation video paths (legacy) |
| `--val-labels` | Validation label paths (legacy) |
| `--batch-size` | Training batch size |
| `--learning-rate` | Learning rate |
| `--num-epochs` | Number of epochs |
| `--weight-decay` | Weight decay |
| `--gradient-clip` | Gradient clipping value |
| `--early-stopping-patience` | Early stopping patience |
| `--num-workers` | Data loader workers |
| `--experiment-name` | TensorBoard experiment name |
| `--resume` | Checkpoint to resume from |
| `--dinov3-model` | DINOv3 model name |

### Test Specific
| Flag | Description |
|------|-------------|
| `--checkpoint` | Model checkpoint path (required) |
| `--test-videos` | Test video paths (legacy) |
| `--test-labels` | Test label paths (legacy) |
| `--stride` | Stride between windows |
| `--threshold` | Classification threshold (default 0.5) |
| `--plot-cm` | Plot confusion matrices |

### Prediction Specific
| Flag | Description |
|------|-------------|
| `--checkpoint` | Model checkpoint path (required) |
| `--videos` | Video paths for prediction (legacy) |
| `--stride` | Stride between windows |
| `--threshold` | Classification threshold (default 0.5) |
| `--save-probs` | Save raw probabilities |
| `--batch-size` | Batch size |
| `--num-workers` | Data loader workers |

---

## Output Files

### Training
- `outputs/{experiment}/checkpoint_best.pth` - Best model (by macro F1)
- `outputs/{experiment}/checkpoint_latest.pth` - Latest epoch
- `outputs/{experiment}/config.yaml` - Merged config
- `outputs/{experiment}/tensorboard/` - TensorBoard logs

### Testing
- `outputs/test_results/metrics.json` - Detailed metrics
- `outputs/test_results/detailed_predictions.csv` - Per-window predictions
- `outputs/test_results/test_config.yaml` - Test config used
- `outputs/test_results/confusion_matrices.png` - (with `--plot-cm`)
- `outputs/test_results/per_class_metrics.png` - (with `--plot-cm`)

### Prediction
- `outputs/predictions/predictions.csv` - Frame-level predictions
- `outputs/predictions/predictions.json` - Detailed predictions
- `outputs/predictions/video_summaries.json` - Per-video summaries
- `outputs/predictions/prediction_config.yaml` - Config used

---

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
- **Matches notebook exactly**: BGR→RGB→Grayscale→3-channel repeat

### Dataset: `utils/dataset.py`
- `BehaviorDataset`: For training/validation with labels
- `PredictionDataset`: For inference without labels
- Sliding window creation with configurable stride
- Variable patch count per frame handled via padding + masking

### Data Config: `utils/data_config.py`
- `VideoConfig`: Single video configuration (path, labels, reference_patch)
- `DataConfig`: Train/val/test splits with per-video reference patches
- `load_data_config()` / `create_data_config_from_lists()` helpers

---

## Metrics Computed

- **Macro F1** (primary metric)
- Micro F1, Weighted F1
- Macro Precision, Macro Recall
- Subset Accuracy (exact match)
- Hamming Loss
- Macro AUC
- Per-class F1, Precision, Recall, AUC

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- transformers, accelerate (for DINOv3)
- scikit-learn, pandas, matplotlib, seaborn, tqdm, pyyaml

Install all:
```bash
pip install -r requirements.txt
```

---

## Reference Notebook

See `2_DINO_sim.ipynb` for interactive exploration of:
- DINOv3 feature extraction
- Patch similarity computation
- Patch selection with border
- Visualization of selected regions

---

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