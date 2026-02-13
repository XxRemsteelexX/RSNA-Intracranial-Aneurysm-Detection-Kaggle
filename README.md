# RSNA 2025 Intracranial Aneurysm Detection

Comprehensive 3D deep learning solution for detecting and localizing intracranial aneurysms from CT angiography scans. This repository documents the complete research pipeline, from data preprocessing through ensemble optimization.

**Competition**: [RSNA 2025 Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)

## Key Results

- **Best Single Model**: SE-ResNet18 Stable (AUC: 0.8585)
- **Best Ensemble**: META_E_top3_weighted (AUC: 0.8624)
- **Total Models Trained**: 105 models (21 architectures x 5 folds)
- **Ensemble Configurations Tested**: 51
- **Key Finding**: Smaller models outperform larger ones on limited data

## Quick Start

```bash
# Clone repository
git clone https://github.com/XxRemsteelexX/RSNA-Intracranial-Aneurysm-Detection-Kaggle.git
cd RSNA-Intracranial-Aneurysm-Detection-Kaggle

# Setup environment
conda create -n rsna_kaggle python=3.11
conda activate rsna_kaggle
conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
pip install -r requirements.txt

# Run full pipeline
bash scripts/07_run_full_pipeline.sh
```

## Project Overview

### Problem Statement

Detect and localize intracranial aneurysms across 13 anatomical locations in 3D CT angiography scans using deep learning. This is a multi-label classification problem with 14 binary targets:

1. Left Infraclinoid Internal Carotid Artery
2. Right Infraclinoid Internal Carotid Artery
3. Left Supraclinoid Internal Carotid Artery
4. Right Supraclinoid Internal Carotid Artery
5. Left Middle Cerebral Artery
6. Right Middle Cerebral Artery
7. Anterior Communicating Artery
8. Left Anterior Cerebral Artery
9. Right Anterior Cerebral Artery
10. Left Posterior Communicating Artery
11. Right Posterior Communicating Artery
12. Basilar Tip
13. Other Posterior Circulation
14. **Aneurysm Present** (global flag)

### Dataset Statistics

- **Total Scans**: 4,348 CT angiography volumes
- **Training Set**: 3,478 samples per fold (5-fold stratified CV)
- **Validation Set**: 870 samples per fold
- **Patch Size**: 64x64x64 voxels (32mm^3 at 0.5mm spacing)
- **Positive Rate**: 42.8% (aneurysm present)
- **Class Imbalance**: Highly imbalanced across anatomical locations

### Approach

**1. Data Preprocessing**
- DICOM -> NIfTI conversion with spatial metadata preservation
- HU windowing [-100, 300] for CTA visualization
- ROI extraction: 64^3 patches centered on brain tissue
- Z-score normalization

**2. Model Architecture**
- 3D Convolutional Neural Networks (CNNs)
- Focus on SE-ResNet family (Squeeze-and-Excitation blocks)
- Trained from scratch on medical imaging data
- 21 architectures tested across 5 folds

**3. Training Strategy**
- Stratified 5-fold cross-validation
- Adam optimizer (lr=0.0005 optimal)
- Cosine annealing LR schedule
- BCEWithLogitsLoss with class weighting
- Early stopping (patience=15)
- FP16 mixed precision training

**4. Ensemble Optimization**
- 51 ensemble configurations tested
- Simple mean averaging optimal
- 5-6 models hit sweet spot
- SE-ResNet exclusive ensembles best

## Results Summary

### Top 10 Individual Models

| Rank | Model | AUC | Batch Size | Notes |
|------|-------|-----|------------|-------|
| 1 | **SE-ResNet18 Stable** | **0.8585** | 12 | LR=0.0005, patience=15 |
| 2 | SE-ResNet18 | 0.8551 | 8 | Original configuration |
| 3 | ConvNeXt-Large fine-tuned | 0.8540 | 2 | Frozen -> fine-tuned |
| 4 | SE-ResNet34 | 0.8538 | 8 | |
| 5 | SE-ResNet50 | 0.8528 | 4 | |
| 6 | DenseNet-121 | 0.8514 | 8 | Full training |
| 7 | DenseNet-121 (v2) | 0.8499 | 8 | Repeat run |
| 8 | ResNet-18 | 0.8498 | 8 | Standard ResNet |
| 9 | DenseNet-121 (v3) | 0.8494 | 8 | LR=0.0005 |
| 10 | EfficientNet-B0 | 0.8492 | 8 | |

### Top 5 Ensembles

| Rank | Ensemble | AUC (Macro) | AUC (Aneurysm) | Models | Method |
|------|----------|-------------|----------------|--------|--------|
| 1 | E5_005_seresnet5 | **0.8624** | 0.8269 | 5 SE-ResNet | Simple mean |
| 2 | META_E_top3_weighted | **0.8624** | 0.8249 | 6 SE-ResNet | Weighted |
| 3 | E3_004_seresnet_only | 0.8619 | 0.8248 | 3 SE-ResNet | Simple mean |
| 4 | FIXED_E5_003_seresnet_alt1 | 0.8619 | **0.8298** | 5 SE-ResNet | Simple mean |
| 5 | META_E_top3_ensembles | 0.8619 | 0.8259 | 6 SE-ResNet | Simple mean |

### Key Findings

#### 1. Smaller Models Outperform Larger Ones

**Most Important Discovery:**
- SE-ResNet18 (0.8585) > SE-ResNet34 (0.8538) > SE-ResNet50 (0.8528)
- DenseNet-121 (0.8514) > DenseNet-169 (0.8430)
- EfficientNet-B0 (0.8492) > B2 (0.8472) > B4 (0.8428)
- ResNet-18 (0.8498) > ResNet-34 (0.8365)

**Reason**: ~4,000 training samples insufficient for large models -> overfitting

#### 2. SE Blocks Critical for Medical Imaging

- SE-ResNet18 (0.8585) vs ResNet-18 (0.8498) = **+8.7% improvement**
- Squeeze-and-Excitation attention mechanism enhances feature learning
- All top 10 ensembles use SE-ResNet exclusively

#### 3. Ensemble Sweet Spot: 5-6 Models

| Models | Best AUC | Performance |
|--------|----------|-------------|
| 3 | 0.8619 | 99.94% of best |
| 5-6 | **0.8624** | **OPTIMAL** |
| 10 | 0.8618 | Diminishing returns |
| 45 | 0.8582 | -0.5% (worse) |
| 65 | 0.8582 | -0.5% (worse) |

#### 4. Simple Mean > Weighted Averaging

- Weighted averaging: <0.05% improvement
- Not worth added complexity
- Exception: Meta-ensembling benefits from weighting

#### 5. Test-Time Augmentation (TTA) Not Worth It

- TTA=4: +0.06% improvement (marginal)
- TTA=8: -4.0% degradation (catastrophic)
- 4-8x slower inference
- **Recommendation**: Skip TTA for production

## Repository Structure

```
rsna_github/
+-- README.md # This file
+-- PROGRESS_STATUS.md # Development tracking
+-- requirements.txt # Python dependencies
+-- .gitignore # Git ignore rules
|
+-- scripts/ # Sequential pipeline scripts
| +-- 01_dicom_to_volume.py # DICOM -> NIfTI conversion
| +-- 02_create_roi_patches.py # Extract 64^3 ROI patches
| +-- 03_create_cv_splits.py # Generate 5-fold CV splits
| +-- 04_train_model.py # Main training script (21 architectures)
| +-- 05_ensemble_inference.py # Ensemble prediction
| +-- 06_inference_with_tta.py # Test-time augmentation
| +-- 07_run_full_pipeline.sh # Full pipeline automation
| |
| +-- utils/ # Modular utilities
| +-- architectures.py # Model definitions
| +-- data_loading.py # Dataset & augmentation
| +-- metrics.py # Training & evaluation
|
+-- notebooks/ # Jupyter analysis notebooks
| +-- 01_data_exploration_and_eda.ipynb
| +-- 02_preprocessing_and_roi_extraction.ipynb
| +-- 03_architecture_selection_rationale.ipynb
| +-- 04_training_results_analysis.ipynb
| +-- 05_ensemble_strategy_design.ipynb
| +-- 06_ensemble_evaluation_results.ipynb
|
+-- docs/ # Comprehensive documentation
| +-- MODEL_DATABASE.md # All 105 model results
| +-- ENSEMBLE_RESULTS.md # 51 ensemble experiments
| +-- GITHUB_ORGANIZATION_PLAN.md # Repository design plan
| +-- EXECUTION_SUMMARY.md # Quick reference
| +-- PROGRESS_STATUS.md # Development status
|
+-- configs/ # Configuration files
| +-- architectures.yaml # Model configurations
| +-- hyperparameters.yaml # Training settings
| +-- augmentation.yaml # Data augmentation
| +-- ensemble_configs/ # 51 ensemble configs
|
+-- launch_scripts/ # Training automation
| +-- launch_single_model.sh
| +-- launch_5fold_training.sh
| +-- launch_parallel_training.sh
| +-- monitor_training.sh
|
+-- tests/ # Unit tests
| +-- test_preprocessing.py
| +-- test_models.py
| +-- test_inference.py
|
+-- models/ # Saved checkpoints (gitignored)
+-- logs/ # Training logs (gitignored)
+-- results/ # Predictions & metrics (gitignored)
+-- data/ # Dataset (gitignored)
 +-- raw/ # Original DICOM files
 +-- volumes_nifti/ # Converted NIfTI volumes
 +-- patches_roi/ # Extracted 64^3 patches
 +-- cv_splits/ # 5-fold CV indices
 +-- train_labels_14class.csv # Multi-label annotations
```

## Pipeline Execution

### Sequential Workflow

#### Step 1: DICOM to NIfTI Conversion

```bash
python scripts/01_dicom_to_volume.py \
 --dicom-dir data/raw/train_images \
 --output-dir data/volumes_nifti \
 --num-workers 16
```

**Output**: ~4,348 NIfTI volumes with preserved spatial metadata

#### Step 2: ROI Patch Extraction

```bash
python scripts/02_create_roi_patches.py \
 --nifti-dir data/volumes_nifti \
 --output-dir data/patches_roi \
 --patch-size 64 \
 --window-min -100 \
 --window-max 300
```

**Output**: 64x64x64 voxel patches centered on brain tissue, z-score normalized

#### Step 3: Cross-Validation Split Generation

```bash
python scripts/03_create_cv_splits.py \
 --labels-csv data/train_labels_14class.csv \
 --output-dir data/cv_splits \
 --n-folds 5 \
 --seed 42
```

**Output**: Stratified 5-fold splits ensuring balanced positive/negative ratios

#### Step 4: Model Training

```bash
# Single model training
python scripts/04_train_model.py \
 --data-dir data/patches_roi \
 --labels-csv data/train_labels_14class.csv \
 --cv-dir data/cv_splits \
 --fold 0 \
 --arch seresnet18_stable \
 --batch-size 12 \
 --lr 0.0005 \
 --epochs 50 \
 --patience 15 \
 --output models/fold0_seresnet18_stable
```

**Supported Architectures**:
- SE-ResNet family: seresnet10, seresnet14, seresnet18, seresnet34, seresnet50
- DenseNet family: densenet121, densenet169
- ResNet family: resnet18, resnet34, resnet50
- EfficientNet family: efficientnet_b0, efficientnet_b2, efficientnet_b3, efficientnet_b4
- MobileNet family: mobilenetv3
- Vision Transformers: vit, swin, convnext
- Other: inception, unet3d

#### Step 5: Ensemble Inference

```bash
# Production-ready ensemble (6 models, weighted)
python scripts/05_ensemble_inference.py \
 --config-id META_E_top3_weighted \
 --method weighted \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 models/fold1_seresnet34_p64/best_model.pth \
 models/fold1_seresnet18_p64/best_model.pth \
 models/fold3_seresnet18_p64/best_model.pth \
 --weights 2.5862 2.5862 1.7243 1.7243 1.7243 0.8619 \
 --val-dir data/patches_roi \
 --val-csv data/train_labels_14class.csv \
 --output results/ensemble_predictions.csv
```

#### Step 6: Test-Time Augmentation (Optional)

```bash
python scripts/06_inference_with_tta.py \
 --model models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 --arch seresnet18_stable \
 --val-dir data/patches_roi \
 --val-csv data/train_labels_14class.csv \
 --tta-count 4 \
 --output results/tta_predictions.csv
```

**Note**: TTA=4 provides minimal benefit (+0.06%) with 4x slower inference. Skip for production.

### Parallel Training

Train multiple models simultaneously on dual GPUs:

```bash
# Launch parallel training (2 GPUs)
bash launch_scripts/launch_parallel_training.sh

# Monitor training progress
bash launch_scripts/monitor_training.sh

# View specific log
tail -f logs/train_fold0_seresnet18_stable.log
```

## Hardware Requirements

### Author's Training Setup

All 105 models were trained locally across two workstations:

**Main Workstation**:
- **GPUs**: 2x NVIDIA RTX 5090 (32GB VRAM each)
- **CPU**: AMD/Intel multi-core (16+ threads)
- **RAM**: 128GB
- **Storage**: 4TB NVMe SSD (2x 2TB WD Black SN850X in RAID 0)
- **OS**: Ubuntu 22.04 LTS

**Secondary PC**:
- **GPUs**: 1x NVIDIA RTX 3090 Ti (24GB VRAM) + 1x RTX 3090 (24GB VRAM)
- **Total Training GPUs**: 4

This multi-GPU setup enabled parallel training of different architectures across the 21-architecture x 5-fold experimental space.

### Minimum Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (RTX 4090 or RTX 3090 recommended)
- **RAM**: 64GB
- **Storage**: 500GB SSD
- **CUDA**: 12.1+
- **PyTorch**: 2.0+

### Batch Sizes

| Model | Batch Size | Params |
|-------|------------|--------|
| SE-ResNet18 | 12 | ~11M |
| SE-ResNet34 | 8 | ~22M |
| SE-ResNet50 | 4 | ~28M |
| DenseNet-121 | 8 | ~8M |

## Model Architecture Details

### SE-ResNet18 Stable (Best Model)

```
Input: (1, 64, 64, 64) # Single-channel 3D CT patch
+-- Conv3d(1->64, 7x7x7, stride=2) # Initial feature extraction
+-- BatchNorm3d + ReLU + MaxPool3d
|
+-- Layer1: 2 x SE-BasicBlock(64->64) # Spatial size: 16x16x16
+-- Layer2: 2 x SE-BasicBlock(64->128, stride=2) # 8x8x8
+-- Layer3: 2 x SE-BasicBlock(128->256, stride=2) # 4x4x4
+-- Layer4: 2 x SE-BasicBlock(256->512, stride=2) # 2x2x2
|
+-- AdaptiveAvgPool3d -> (512, 1, 1, 1)
+-- Flatten -> (512,)
+-- Linear(512->14) -> Multi-label output
```

**SE-BasicBlock**:
```
Input: x
+-- Conv3d(3x3x3) -> BN -> ReLU -> Conv3d(3x3x3) -> BN
+-- Squeeze-and-Excitation:
| +-- Global average pooling -> (C,)
| +-- FC(C->C/16) -> ReLU -> FC(C/16->C) -> Sigmoid
| +-- Channel-wise multiplication
+-- Residual connection: x + SE(Conv(x))
```

**Parameters**: ~11M
**VRAM**: ~2.5GB (batch size 12)

## Data Augmentation

Medical-appropriate 3D augmentations that preserve anatomical validity:

```python
VolumeAugmentation(
 rotation_range=15, # +/-15 degrees (preserves anatomy)
 flip=True, # All axes (valid for brain)
 zoom_range=(0.9, 1.1), # Conservative zoom
 shift_range=0.1, # +/-10% translation
 brightness_range=0.2, # +/-20% intensity
 contrast_range=0.2 # +/-20% contrast
)
```

**Rationale**:
- Rotations limited to +/-15 degrees to avoid unrealistic anatomical positions
- All flips valid (bilateral symmetry of brain)
- Conservative zoom to preserve vascular structure scale
- Intensity augmentations account for scanner/protocol variations

## Training Configuration

### Hyperparameters (Optimized)

```yaml
# Model
architecture: seresnet18_stable
patch_size: 64
num_classes: 14

# Optimization
optimizer: Adam
learning_rate: 0.0005 # Optimal (0.001 causes instability)
weight_decay: 1e-4
lr_scheduler: CosineAnnealingLR
min_lr: 1e-6

# Training
batch_size: 12 # Optimal for SE-ResNet18
epochs: 50
patience: 15 # Early stopping
gradient_clip: 1.0

# Data
augmentation: true
num_workers: 8
pin_memory: true

# Loss
criterion: BCEWithLogitsLoss
class_weights: computed # From training set distribution
label_smoothing: 0.0

# Precision
mixed_precision: true # FP16 for faster training
```

### Class Weights

Computed from training set to handle class imbalance:

| Class | Positive Rate | Weight |
|-------|---------------|--------|
| Aneurysm Present | 42.8% | 1.34 |
| Left Infraclinoid ICA | 8.2% | 11.2 |
| Right Infraclinoid ICA | 7.9% | 11.6 |
| Left Supraclinoid ICA | 12.4% | 7.1 |
| Right Supraclinoid ICA | 11.8% | 7.5 |
| Left MCA | 18.3% | 4.5 |
| Right MCA | 17.6% | 4.7 |
| Anterior Communicating | 9.4% | 9.6 |
| Left ACA | 5.3% | 17.9 |
| Right ACA | 5.1% | 18.6 |
| Left PComm | 6.8% | 13.7 |
| Right PComm | 6.2% | 15.1 |
| Basilar Tip | 7.4% | 12.5 |
| Other Posterior | 1.2% | 82.3 |

## Ensemble Strategies

### Recommended Production Ensembles

#### 1. Maximum Performance: META_E_top3_weighted

**AUC**: 0.8624 (tied best)

```bash
python scripts/05_ensemble_inference.py \
 --config-id META_E_top3_weighted \
 --method weighted \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 models/fold1_seresnet34_p64/best_model.pth \
 models/fold1_seresnet18_p64/best_model.pth \
 models/fold3_seresnet18_p64/best_model.pth \
 --weights 2.5862 2.5862 1.7243 1.7243 1.7243 0.8619
```

**Pros**: Most robust, 6 diverse models, meta-ensemble stability
**Cons**: Slightly slower (6 models)

#### 2. Balanced: E5_005_seresnet5

**AUC**: 0.8624 (tied best)

```bash
python scripts/05_ensemble_inference.py \
 --config-id E5_005_seresnet5 \
 --method mean \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth \
 models/fold1_seresnet34_p64/best_model.pth \
 models/fold1_seresnet18_p64/best_model.pth
```

**Pros**: Simple mean, excellent performance, 5 models
**Cons**: None

#### 3. Fast Inference: E3_004_seresnet_only

**AUC**: 0.8619 (99.94% of best)

```bash
python scripts/05_ensemble_inference.py \
 --config-id E3_004_seresnet_only \
 --method mean \
 --models \
 models/fold0_seresnet10_p64/best_model.pth \
 models/fold0_seresnet34_p64/best_model.pth \
 models/fold0_seresnet18_stable_bs12_lr0005/best_model.pth
```

**Pros**: Only 3 models (2x faster), all fold 0 (easy deployment)
**Cons**: Slightly lower AUC (-0.05%)

## Reproducibility

### Random Seeds

All experiments use fixed random seeds:

```python
import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Cross-Validation

Stratified 5-fold CV with:
- Balanced label distribution across folds (+/-2% tolerance)
- Patient-level splitting (no data leakage)
- Seeded random state for reproducible splits

### Environment

```yaml
python: 3.11
pytorch: 2.7.0+cu128
cuda: 12.8
cudnn: 9.3.0
numpy: 2.2.3
pandas: 2.2.3
scikit-learn: 1.6.1
nibabel: 5.3.2
scipy: 1.15.1
```

Full dependency list in `requirements.txt`.

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{dalbey2025rsna_aneurysm,
 title={RSNA 2025 Intracranial Aneurysm Detection: Comprehensive Deep Learning Pipeline},
 author={Dalbey, Glenn},
 year={2025},
 publisher={GitHub},
 url={https://github.com/XxRemsteelexX/RSNA-Intracranial-Aneurysm-Detection-Kaggle},
 note={Best AUC: 0.8624 (ensemble), 0.8585 (single model)}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- **RSNA** for organizing the challenge
- **Kaggle** for hosting the competition
- **PyTorch** team for the deep learning framework
- **Medical imaging community** for CTA preprocessing best practices

## Author's Note

This was my first Kaggle competition. I trained 105 models and tested 51 ensemble configurations locally on my hardware: 2x NVIDIA RTX 5090 GPUs on my main workstation and an additional RTX 3090 Ti + RTX 3090 on my second PC.

Unfortunately, I didn't realize that submissions had to be made 24 hours before the competition deadline--you can only select from previously submitted models during the final day. Despite not being able to submit my final results, this work represents a comprehensive exploration of 3D medical imaging.

The key finding--that smaller models significantly outperform larger ones on limited medical imaging datasets--emerged from methodically testing the hypothesis that SOTA large models would work best, documenting their failures, and iterating toward simpler solutions. Sometimes the best discoveries come from what doesn't work.

## Contact

- **Author**: Glenn Dalbey
- **GitHub**: [@XxRemsteelexX](https://github.com/XxRemsteelexX)
- **Email**: dalbeyglenn@gmail.com

## Documentation

For detailed information, see:

- [MODEL_DATABASE.md](docs/MODEL_DATABASE.md) - Complete results for all 105 models
- [ENSEMBLE_RESULTS.md](docs/ENSEMBLE_RESULTS.md) - All 51 ensemble experiments
- [GITHUB_ORGANIZATION_PLAN.md](docs/GITHUB_ORGANIZATION_PLAN.md) - Repository design rationale
- [PROGRESS_STATUS.md](docs/PROGRESS_STATUS.md) - Development tracking

---

**Status**: Production-Ready | **Best AUC**: 0.8624 (ensemble) | **Last Updated**: 2025-10-17
