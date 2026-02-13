#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) inference script for RSNA aneurysm detection.

Supports:
- All model architectures from train_eric3d_optimized.py
- 8-flip TTA (X, Y, Z axis combinations)
- Batch processing for efficiency
- Validation set evaluation
- Test set prediction generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score
import json

# Import model architectures from training script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_eric3d_optimized import (
    ResNet3D, DenseNet3D, EfficientNet3D, VisionTransformer3D, UNet3D,
    MobileNetV2_3D, MobileNetV3_3D, MobileNetV4_3D, SwinTransformer3D,
    ConvNeXt3D, Inception3D, SEResNet3D, LABEL_COLS
)


# --- TTA Augmentation Functions ---

def apply_tta_flip(volume, flip_x=False, flip_y=False, flip_z=False):
    """
    Apply TTA flip augmentations to a 3D volume.

    Args:
        volume: (C, D, H, W) tensor
        flip_x: flip along D axis (left-right)
        flip_y: flip along H axis (anterior-posterior)
        flip_z: flip along W axis (superior-inferior)

    Returns:
        Flipped volume
    """
    if flip_x:
        volume = torch.flip(volume, dims=[1])
    if flip_y:
        volume = torch.flip(volume, dims=[2])
    if flip_z:
        volume = torch.flip(volume, dims=[3])
    return volume


def get_tta_flips(num_augs=8):
    """
    Get TTA flip configurations.

    Args:
        num_augs: number of augmentations (4 or 8)

    Returns:
        List of (flip_x, flip_y, flip_z) tuples
    """
    if num_augs == 8:
        # All 8 combinations of 3 axes
        return [
            (False, False, False),  # Original
            (True, False, False),   # Flip X
            (False, True, False),   # Flip Y
            (False, False, True),   # Flip Z
            (True, True, False),    # Flip XY
            (True, False, True),    # Flip XZ
            (False, True, True),    # Flip YZ
            (True, True, True),     # Flip XYZ
        ]
    elif num_augs == 4:
        # 4 common flips (faster)
        return [
            (False, False, False),  # Original
            (True, False, False),   # Flip X (left-right)
            (False, True, False),   # Flip Y
            (True, True, False),    # Flip XY
        ]
    else:
        raise ValueError(f"num_augs must be 4 or 8, got {num_augs}")


# --- Dataset ---

class Eric3DInferenceDataset(Dataset):
    """Dataset for inference with TTA support."""

    def __init__(self, patch_files, labels_df=None, patch_size=64):
        self.patch_files = patch_files
        self.labels_df = labels_df
        self.patch_size = patch_size

        # Map series_uid to labels (if provided)
        if labels_df is not None:
            self.uid_to_labels = {}
            for _, row in labels_df.iterrows():
                uid = row['series_uid']
                labels = row[LABEL_COLS].values.astype(np.float32)
                self.uid_to_labels[uid] = labels
        else:
            self.uid_to_labels = None

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        h5_path = self.patch_files[idx]
        series_uid = h5_path.stem

        # Load patch
        with h5py.File(h5_path, 'r') as f:
            g = f[f'patches_{self.patch_size}']
            patch = np.array(g['data'][0], dtype=np.float32)  # (D, H, W)

        # Add channel dimension
        patch = patch[np.newaxis, ...]  # (1, D, H, W)

        # Get labels if available
        if self.uid_to_labels is not None:
            labels = self.uid_to_labels[series_uid].copy()
            return torch.from_numpy(patch).float(), torch.from_numpy(labels).float(), series_uid
        else:
            return torch.from_numpy(patch).float(), series_uid


# --- TTA Inference ---

def predict_with_tta(model, patch, tta_flips, device):
    """
    Run TTA inference on a single patch.

    Args:
        model: trained model
        patch: (1, D, H, W) tensor
        tta_flips: list of (flip_x, flip_y, flip_z) tuples
        device: torch device

    Returns:
        (14,) array of averaged predictions
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for flip_x, flip_y, flip_z in tta_flips:
            # Apply TTA flip
            aug_patch = apply_tta_flip(patch.unsqueeze(0), flip_x, flip_y, flip_z)
            aug_patch = aug_patch.to(device)

            # Forward pass
            logits = model(aug_patch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs[0])

    # Average predictions across all augmentations
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds


def predict_batch_with_tta(model, batch, tta_flips, device):
    """
    Run TTA inference on a batch of patches (more efficient).

    Args:
        model: trained model
        batch: (B, 1, D, H, W) tensor
        tta_flips: list of (flip_x, flip_y, flip_z) tuples
        device: torch device

    Returns:
        (B, 14) array of averaged predictions
    """
    model.eval()
    B = batch.shape[0]
    all_preds = np.zeros((B, len(tta_flips), 14))

    with torch.no_grad():
        for aug_idx, (flip_x, flip_y, flip_z) in enumerate(tta_flips):
            # Apply TTA flip to entire batch
            aug_batch = apply_tta_flip(batch, flip_x, flip_y, flip_z)
            aug_batch = aug_batch.to(device)

            # Forward pass
            logits = model(aug_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds[:, aug_idx, :] = probs

    # Average predictions across all augmentations
    avg_preds = all_preds.mean(axis=1)
    return avg_preds


# --- Evaluation ---

def evaluate_with_tta(model, dataloader, tta_flips, device):
    """
    Evaluate model with TTA on validation set.

    Returns:
        mean_auc, per_class_aucs, predictions, labels, uids
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_uids = []

    for batch_data in tqdm(dataloader, desc="TTA Inference"):
        if len(batch_data) == 3:  # Has labels
            patches, labels, uids = batch_data
            labels_np = labels.numpy()
            all_labels.append(labels_np)
        else:  # No labels
            patches, uids = batch_data
            labels_np = None

        # Run TTA prediction
        preds = predict_batch_with_tta(model, patches, tta_flips, device)
        all_preds.append(preds)
        all_uids.extend(uids)

    all_preds = np.vstack(all_preds)

    if all_labels:
        all_labels = np.vstack(all_labels)

        # Compute per-class AUC
        aucs = []
        for i in range(all_labels.shape[1]):
            try:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
            except:
                aucs.append(0.0)

        mean_auc = np.mean(aucs)
        return mean_auc, aucs, all_preds, all_labels, all_uids
    else:
        return None, None, all_preds, None, all_uids


# --- Model Loading ---

def load_model(arch, checkpoint_path, device, num_classes=14):
    """
    Load trained model from checkpoint.

    Args:
        arch: architecture name
        checkpoint_path: path to .pth file
        device: torch device
        num_classes: number of output classes

    Returns:
        loaded model
    """
    # Create model
    if arch == 'densenet121':
        model = DenseNet3D(num_classes=num_classes, block_config=(6, 12, 24, 16))
    elif arch == 'densenet169':
        model = DenseNet3D(num_classes=num_classes, block_config=(6, 12, 32, 32))
    elif arch == 'efficientnet_b0':
        model = EfficientNet3D(num_classes=num_classes, variant='b0')
    elif arch == 'efficientnet_b2':
        model = EfficientNet3D(num_classes=num_classes, variant='b2')
    elif arch == 'efficientnet_b3':
        model = EfficientNet3D(num_classes=num_classes, variant='b2')
    elif arch == 'efficientnet_b4':
        model = EfficientNet3D(num_classes=num_classes, variant='b2')
    elif arch == 'efficientnet_b7':
        model = EfficientNet3D(num_classes=num_classes, variant='b2')
    elif arch == 'vit':
        model = VisionTransformer3D(num_classes=num_classes)
    elif arch == 'unet3d':
        model = UNet3D(num_classes=num_classes)
    elif arch == 'mobilenetv2':
        model = MobileNetV2_3D(num_classes=num_classes)
    elif arch == 'mobilenetv3':
        model = MobileNetV3_3D(num_classes=num_classes)
    elif arch == 'mobilenetv4':
        model = MobileNetV4_3D(num_classes=num_classes, variant='medium')
    elif arch == 'swin':
        model = SwinTransformer3D(num_classes=num_classes)
    elif arch == 'convnext':
        model = ConvNeXt3D(num_classes=num_classes)
    elif arch == 'inception':
        model = Inception3D(num_classes=num_classes)
    elif arch == 'seresnet10':
        model = SEResNet3D(num_classes=num_classes, depth=10)
    elif arch == 'seresnet14':
        model = SEResNet3D(num_classes=num_classes, depth=14)
    elif arch == 'seresnet18':
        model = SEResNet3D(num_classes=num_classes, depth=18)
    elif arch == 'seresnet34':
        model = SEResNet3D(num_classes=num_classes, depth=34)
    elif arch == 'seresnet50':
        model = SEResNet3D(num_classes=num_classes, depth=50)
    elif arch == 'seresnet101':
        model = SEResNet3D(num_classes=num_classes, depth=101)
    else:
        depth_map = {'resnet18': 18, 'resnet34': 34, 'resnet50': 50, 'resnet101': 101}
        depth = depth_map[arch]
        model = ResNet3D(num_classes=num_classes, depth=depth)

    # Load weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"- Loaded {arch} model from {checkpoint_path}")
    return model


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description='TTA inference for RSNA aneurysm detection')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--arch', type=str, required=True,
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                                'densenet121', 'densenet169',
                                'efficientnet_b0', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b7',
                                'vit', 'unet3d', 'mobilenetv2', 'mobilenetv3', 'mobilenetv4',
                                'swin', 'convnext', 'inception',
                                'seresnet10', 'seresnet14', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101'])
    parser.add_argument('--data-dir', type=str, required=True, help='Directory with H5 patches')
    parser.add_argument('--labels-csv', type=str, default=None, help='CSV with labels (for validation)')
    parser.add_argument('--cv-dir', type=str, default=None, help='CV split directory (for validation)')
    parser.add_argument('--fold', type=int, default=0, help='Fold number (for validation)')
    parser.add_argument('--output', type=str, required=True, help='Output directory for predictions')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-augs', type=int, default=8, choices=[4, 8],
                        help='Number of TTA augmentations (4 or 8)')
    parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'],
                        help='val: evaluate on validation set, test: generate test predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get TTA flips
    tta_flips = get_tta_flips(args.num_augs)
    print(f"\nTTA Configuration: {args.num_augs} augmentations")
    for i, (fx, fy, fz) in enumerate(tta_flips):
        flip_str = f"X" if fx else ""
        flip_str += f"Y" if fy else ""
        flip_str += f"Z" if fz else ""
        if not flip_str:
            flip_str = "Original"
        print(f"  Aug {i+1}: {flip_str}")

    # Load model
    model = load_model(args.arch, args.checkpoint, device)

    # Prepare data
    data_dir = Path(args.data_dir)

    if args.mode == 'val':
        # Validation mode: evaluate on validation set with labels
        if not args.labels_csv or not args.cv_dir:
            raise ValueError("--labels-csv and --cv-dir required for validation mode")

        labels_df = pd.read_csv(args.labels_csv)
        cv_dir = Path(args.cv_dir) / f"fold_{args.fold}"
        val_idx = np.load(cv_dir / 'val_indices.npy')

        # Get validation files
        all_patches = list(data_dir.glob("*.h5"))
        uid_to_patch = {p.stem: p for p in all_patches}
        val_files = [uid_to_patch[labels_df.iloc[i]['series_uid']]
                    for i in val_idx if labels_df.iloc[i]['series_uid'] in uid_to_patch]

        print(f"\nValidation set: {len(val_files)} files")

        # Create dataset
        val_dataset = Eric3DInferenceDataset(val_files, labels_df, args.patch_size)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                               shuffle=False, num_workers=4)

        # Run TTA evaluation
        print("\nRunning TTA inference...")
        mean_auc, class_aucs, preds, labels, uids = evaluate_with_tta(
            model, val_loader, tta_flips, device
        )

        # Print results
        print(f"\n{'='*80}")
        print(f"TTA Results (Fold {args.fold}, {args.num_augs} augmentations)")
        print(f"{'='*80}")
        print(f"Mean AUC: {mean_auc:.4f}\n")
        print("Per-class AUC:")
        for i, label in enumerate(LABEL_COLS):
            print(f"  {label:50s}: {class_aucs[i]:.4f}")
        print(f"{'='*80}")

        # Save results
        results = {
            'mean_auc': float(mean_auc),
            'class_aucs': {label: float(auc) for label, auc in zip(LABEL_COLS, class_aucs)},
            'num_augs': args.num_augs,
            'arch': args.arch,
            'fold': args.fold
        }

        with open(output_dir / 'tta_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save predictions
        pred_df = pd.DataFrame(preds, columns=LABEL_COLS)
        pred_df.insert(0, 'series_uid', uids)
        pred_df.to_csv(output_dir / 'tta_predictions.csv', index=False)

        print(f"\nResults saved to {output_dir}")

    else:
        # Test mode: generate predictions for test set
        test_files = list(data_dir.glob("*.h5"))
        print(f"\nTest set: {len(test_files)} files")

        # Create dataset (no labels)
        test_dataset = Eric3DInferenceDataset(test_files, labels_df=None,
                                             patch_size=args.patch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)

        # Run TTA inference
        print("\nGenerating TTA predictions...")
        _, _, preds, _, uids = evaluate_with_tta(model, test_loader, tta_flips, device)

        # Save predictions
        pred_df = pd.DataFrame(preds, columns=LABEL_COLS)
        pred_df.insert(0, 'series_uid', uids)
        pred_df.to_csv(output_dir / 'test_predictions_tta.csv', index=False)

        print(f"\nTest predictions saved to {output_dir / 'test_predictions_tta.csv'}")
        print(f"Shape: {preds.shape}")


if __name__ == '__main__':
    main()
