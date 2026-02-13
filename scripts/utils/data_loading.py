"""
Data Loading and Augmentation for 3D Medical Imaging
====================================================

Efficient data pipeline for training 3D CNNs on volumetric medical images.

Components:
----------
1. VolumeAugmentation: 3D-specific augmentations (rotation, flip, zoom, etc.)
2. PatchDataset: PyTorch Dataset for NPZ patch files
3. Data loaders with efficient multi-processing

Key Design Decisions:
--------------------
- Medical-appropriate augmentations (no unrealistic transforms)
- Memory-efficient patch loading
- On-the-fly augmentation (no disk storage)
- Reproducible with fixed random seeds

Competition: RSNA 2025 Intracranial Aneurysm Detection
Author: Glenn Dalbey
Date: 2025-10-17
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import scipy.ndimage as ndi


# ============================================================================
# 3D VOLUME AUGMENTATION
# ============================================================================

class VolumeAugmentation:
    """
    3D volume augmentation for medical imaging.

    Implements medical-appropriate augmentations:
    - Rotations: Small angles (+/-15 degrees) to preserve anatomy
    - Flips: All axes (anatomically valid)
    - Zoom: Conservative range (0.9-1.1x)
    - Shifts: Small translations
    - Intensity: Brightness/contrast variations

    All augmentations preserve:
    - Spatial relationships between structures
    - HU value distributions
    - Anatomical validity

    Args:
        rotation_range: Max rotation angle in degrees (default: 15)
        flip: Enable random flipping (default: True)
        zoom_range: (min, max) zoom factors (default: (0.9, 1.1))
        shift_range: Max shift as fraction of size (default: 0.1)
        brightness_range: Brightness variation (default: 0.2)
        contrast_range: Contrast variation (default: 0.2)
    """

    def __init__(
        self,
        rotation_range: float = 15,
        flip: bool = True,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        shift_range: float = 0.1,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2
    ):
        self.rotation_range = rotation_range
        self.flip = flip
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply random augmentations to 3D volume.

        Args:
            volume: Input volume (D, H, W) or (C, D, H, W)

        Returns:
            Augmented volume with same shape
        """
        # Ensure writable copy
        volume = np.array(volume, dtype=np.float32, copy=True)
        original_shape = volume.shape

        # Handle channel dimension
        has_channels = len(volume.shape) == 4
        if has_channels:
            # Process each channel separately
            channels = []
            for c in range(volume.shape[0]):
                channels.append(self._augment_single_channel(volume[c]))
            return np.stack(channels, axis=0)
        else:
            return self._augment_single_channel(volume)

    def _augment_single_channel(self, volume: np.ndarray) -> np.ndarray:
        """Augment single channel volume."""
        original_shape = volume.shape

        # 1. Random rotation (3 axes)
        if self.rotation_range > 0:
            angles = np.random.uniform(-self.rotation_range,
                                      self.rotation_range, 3)
            # Rotate around each axis
            volume = ndi.rotate(volume, angles[0], axes=(1, 2),
                               reshape=False, order=1)
            volume = ndi.rotate(volume, angles[1], axes=(0, 2),
                               reshape=False, order=1)
            volume = ndi.rotate(volume, angles[2], axes=(0, 1),
                               reshape=False, order=1)

        # 2. Random flip (all axes with 50% probability each)
        if self.flip:
            if np.random.rand() > 0.5:
                volume = np.ascontiguousarray(np.flip(volume, axis=0))
            if np.random.rand() > 0.5:
                volume = np.ascontiguousarray(np.flip(volume, axis=1))
            if np.random.rand() > 0.5:
                volume = np.ascontiguousarray(np.flip(volume, axis=2))

        # 3. Random zoom
        if self.zoom_range:
            zoom_factor = np.random.uniform(*self.zoom_range)
            volume = ndi.zoom(volume, zoom_factor, order=1)
            # Crop/pad back to original size
            volume = self._resize_to_shape(volume, original_shape)

        # 4. Random shift
        if self.shift_range > 0:
            shift = [int(s * self.shift_range * np.random.uniform(-1, 1))
                    for s in original_shape]
            volume = ndi.shift(volume, shift, order=1)

        # 5. Random brightness
        if self.brightness_range > 0:
            brightness = 1.0 + np.random.uniform(-self.brightness_range,
                                                 self.brightness_range)
            volume = volume * brightness

        # 6. Random contrast
        if self.contrast_range > 0:
            contrast = 1.0 + np.random.uniform(-self.contrast_range,
                                               self.contrast_range)
            mean = volume.mean()
            volume = mean + (volume - mean) * contrast

        # Clip to valid range [0, 1]
        volume = np.clip(volume, 0, 1)

        return np.ascontiguousarray(volume, dtype=np.float32)

    def _resize_to_shape(self, volume: np.ndarray,
                        target_shape: Tuple[int, ...]) -> np.ndarray:
        """Crop or pad volume to target shape."""
        current = volume.shape

        # Pad if smaller
        pad_width = []
        for cur, tgt in zip(current, target_shape):
            diff = tgt - cur
            pad_before = diff // 2
            pad_after = diff - pad_before
            pad_width.append((max(0, pad_before), max(0, pad_after)))

        if any(p[0] > 0 or p[1] > 0 for p in pad_width):
            volume = np.pad(volume, pad_width, mode='constant',
                          constant_values=0)

        # Crop if larger
        slices = []
        for i in range(len(target_shape)):
            if volume.shape[i] > target_shape[i]:
                diff = volume.shape[i] - target_shape[i]
                start = diff // 2
                slices.append(slice(start, start + target_shape[i]))
            else:
                slices.append(slice(None))

        if any(s != slice(None) for s in slices):
            volume = volume[tuple(slices)]

        return np.ascontiguousarray(volume)


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class PatchDataset(Dataset):
    """
    PyTorch Dataset for 3D medical image patches.

    Loads patches from NPZ files with optional augmentation.

    Args:
        patch_files: List of paths to NPZ files
        labels_df: DataFrame with series_uid and label columns
        label_cols: List of label column names
        patch_size: Expected patch size (default: 64)
        augment: Apply augmentation (default: False)
        transform: Optional custom transform function
    """

    def __init__(
        self,
        patch_files: List[Path],
        labels_df: pd.DataFrame,
        label_cols: List[str],
        patch_size: int = 64,
        augment: bool = False,
        transform: Optional[Callable] = None
    ):
        self.patch_files = patch_files
        self.labels_df = labels_df
        self.label_cols = label_cols
        self.patch_size = patch_size
        self.transform = transform

        # Setup augmentation
        if augment and transform is None:
            self.transform = VolumeAugmentation(
                rotation_range=15,
                flip=True,
                zoom_range=(0.9, 1.1),
                shift_range=0.1,
                brightness_range=0.2,
                contrast_range=0.2
            )

        # Create mapping from series_uid to labels
        self.uid_to_labels = {}
        for _, row in labels_df.iterrows():
            uid = row['series_uid']
            labels = row[label_cols].values.astype(np.float32)
            self.uid_to_labels[uid] = labels

    def __len__(self) -> int:
        return len(self.patch_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load patch and labels.

        Returns:
            patch: (1, D, H, W) tensor
            labels: (num_classes,) tensor
        """
        patch_path = self.patch_files[idx]
        series_uid = patch_path.stem

        # Load patch from NPZ
        data = np.load(patch_path)
        patch = data['patch'].astype(np.float32)  # (D, H, W)

        # Ensure correct size
        if patch.shape != (self.patch_size, self.patch_size, self.patch_size):
            raise ValueError(
                f"Patch size mismatch: expected {self.patch_size}^3, "
                f"got {patch.shape}"
            )

        # Apply augmentation/transform
        if self.transform:
            patch = self.transform(patch)

        # Add channel dimension: (D, H, W) -> (1, D, H, W)
        patch = patch[np.newaxis, ...]

        # Get labels
        if series_uid in self.uid_to_labels:
            labels = self.uid_to_labels[series_uid].copy()
        else:
            # Handle missing labels (shouldn't happen with proper filtering)
            labels = np.zeros(len(self.label_cols), dtype=np.float32)

        # Convert to tensors
        patch_tensor = torch.from_numpy(patch).float()
        labels_tensor = torch.from_numpy(labels).float()

        return patch_tensor, labels_tensor


# ============================================================================
# DATA LOADER UTILITIES
# ============================================================================

def create_dataloaders(
    train_files: List[Path],
    val_files: List[Path],
    labels_df: pd.DataFrame,
    label_cols: List[str],
    batch_size: int = 8,
    num_workers: int = 4,
    patch_size: int = 64,
    augment_train: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders.

    Args:
        train_files: Training patch files
        val_files: Validation patch files
        labels_df: Labels DataFrame
        label_cols: Label column names
        batch_size: Batch size
        num_workers: Number of worker processes
        patch_size: Patch size
        augment_train: Apply augmentation to training set

    Returns:
        train_loader, val_loader
    """
    # Create datasets
    train_dataset = PatchDataset(
        train_files, labels_df, label_cols,
        patch_size=patch_size, augment=augment_train
    )

    val_dataset = PatchDataset(
        val_files, labels_df, label_cols,
        patch_size=patch_size, augment=False
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


def filter_available_files(
    series_uids: List[str],
    patch_dir: Path
) -> List[Path]:
    """
    Filter series UIDs to only those with available patch files.

    Args:
        series_uids: List of series UIDs
        patch_dir: Directory containing patch files

    Returns:
        List of available patch file paths
    """
    available_files = []

    for uid in series_uids:
        patch_path = patch_dir / f"{uid}.npz"
        if patch_path.exists():
            available_files.append(patch_path)

    return available_files


# ============================================================================
# TESTING
# ============================================================================

if __name__ == '__main__':
    print("Testing data loading utilities...")

    # Test augmentation
    print("\n1. Testing VolumeAugmentation...")
    aug = VolumeAugmentation()
    test_volume = np.random.rand(64, 64, 64).astype(np.float32)

    augmented = aug(test_volume)
    print(f"   Original shape: {test_volume.shape}")
    print(f"   Augmented shape: {augmented.shape}")
    print(f"   Value range: [{augmented.min():.3f}, {augmented.max():.3f}]")

    # Test with channel dimension
    test_volume_4d = np.random.rand(1, 64, 64, 64).astype(np.float32)
    augmented_4d = aug(test_volume_4d)
    print(f"   4D augmented shape: {augmented_4d.shape}")

    print("\nData loading tests complete!")
