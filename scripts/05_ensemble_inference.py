#!/usr/bin/env python3
"""
Ensemble inference for RSNA aneurysm detection.

Supports:
- Multiple model architectures
- Optional TTA for each model
- Weighted or simple averaging
- Validation and test set prediction
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score
import json

from inference_tta import (
    Eric3DInferenceDataset, load_model, predict_batch_with_tta,
    get_tta_flips, LABEL_COLS
)
from torch.utils.data import DataLoader


class EnsembleModel:
    """Ensemble of multiple models with optional TTA."""

    def __init__(self, model_configs, device, use_tta=True, num_augs=8):
        """
        Args:
            model_configs: list of dicts with keys 'arch', 'checkpoint', 'weight'
            device: torch device
            use_tta: whether to use TTA
            num_augs: number of TTA augmentations (4 or 8)
        """
        self.models = []
        self.weights = []
        self.device = device
        self.use_tta = use_tta
        self.tta_flips = get_tta_flips(num_augs) if use_tta else [(False, False, False)]

        print(f"\n{'='*80}")
        print(f"Loading Ensemble Models (TTA: {use_tta}, Augs: {num_augs if use_tta else 1})")
        print(f"{'='*80}")

        # Load each model
        for i, config in enumerate(model_configs):
            arch = config['arch']
            checkpoint = config['checkpoint']
            weight = config.get('weight', 1.0)

            model = load_model(arch, checkpoint, device)
            self.models.append(model)
            self.weights.append(weight)

            print(f"  Model {i+1}: {arch:20s} | Weight: {weight:.2f} | {checkpoint}")

        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]

        print(f"\nNormalized weights: {[f'{w:.3f}' for w in self.weights]}")
        print(f"{'='*80}\n")

    def predict_batch(self, batch):
        """
        Predict on a batch using all models in ensemble.

        Args:
            batch: (B, 1, D, H, W) tensor

        Returns:
            (B, 14) array of weighted predictions
        """
        all_preds = []

        for model, weight in zip(self.models, self.weights):
            # Get predictions with TTA
            preds = predict_batch_with_tta(model, batch, self.tta_flips, self.device)
            all_preds.append(preds * weight)

        # Weighted average
        ensemble_preds = np.sum(all_preds, axis=0)
        return ensemble_preds


def evaluate_ensemble(ensemble, dataloader):
    """
    Evaluate ensemble on validation/test set.

    Returns:
        mean_auc, per_class_aucs, predictions, labels, uids
    """
    all_preds = []
    all_labels = []
    all_uids = []

    for batch_data in tqdm(dataloader, desc="Ensemble Inference"):
        if len(batch_data) == 3:  # Has labels
            patches, labels, uids = batch_data
            labels_np = labels.numpy()
            all_labels.append(labels_np)
        else:  # No labels
            patches, uids = batch_data
            labels_np = None

        # Get ensemble predictions
        preds = ensemble.predict_batch(patches)
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


def main():
    parser = argparse.ArgumentParser(description='Ensemble inference for RSNA aneurysm detection')
    parser.add_argument('--config', type=str, required=True,
                        help='JSON config file with model specifications')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--labels-csv', type=str, default=None)
    parser.add_argument('--cv-dir', type=str, default=None)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--use-tta', action='store_true', default=True,
                        help='Use TTA for each model')
    parser.add_argument('--num-augs', type=int, default=8, choices=[4, 8])
    parser.add_argument('--mode', type=str, default='val', choices=['val', 'test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    model_configs = config['models']
    print(f"\nLoaded config with {len(model_configs)} models")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create ensemble
    ensemble = EnsembleModel(model_configs, device, args.use_tta, args.num_augs)

    # Prepare data
    data_dir = Path(args.data_dir)

    if args.mode == 'val':
        # Validation mode
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

        # Run ensemble evaluation
        print("\nRunning ensemble inference...\n")
        mean_auc, class_aucs, preds, labels, uids = evaluate_ensemble(ensemble, val_loader)

        # Print results
        print(f"\n{'='*80}")
        print(f"Ensemble Results (Fold {args.fold})")
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
            'num_models': len(model_configs),
            'use_tta': args.use_tta,
            'num_augs': args.num_augs if args.use_tta else 1,
            'fold': args.fold,
            'model_configs': model_configs
        }

        with open(output_dir / 'ensemble_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Save predictions
        pred_df = pd.DataFrame(preds, columns=LABEL_COLS)
        pred_df.insert(0, 'series_uid', uids)
        pred_df.to_csv(output_dir / 'ensemble_predictions.csv', index=False)

        print(f"\nResults saved to {output_dir}")

    else:
        # Test mode
        test_files = list(data_dir.glob("*.h5"))
        print(f"\nTest set: {len(test_files)} files")

        # Create dataset
        test_dataset = Eric3DInferenceDataset(test_files, labels_df=None,
                                             patch_size=args.patch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=4)

        # Run ensemble inference
        print("\nGenerating ensemble predictions...\n")
        _, _, preds, _, uids = evaluate_ensemble(ensemble, test_loader)

        # Save predictions
        pred_df = pd.DataFrame(preds, columns=LABEL_COLS)
        pred_df.insert(0, 'series_uid', uids)
        pred_df.to_csv(output_dir / 'test_predictions_ensemble.csv', index=False)

        print(f"\nTest predictions saved to {output_dir / 'test_predictions_ensemble.csv'}")
        print(f"Shape: {preds.shape}")


if __name__ == '__main__':
    main()
