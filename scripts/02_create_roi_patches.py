#!/usr/bin/env python3
"""
Step 2: ROI Patch Extraction from 3D Volumes
=============================================

Extract 64x64x64 voxel patches from NIfTI volumes centered on brain tissue
regions of interest (ROI). This approach balances computational efficiency
with information preservation for 3D CNN training.

Rationale for 64x64x64 Patches:
-------------------------------
1. Computational Efficiency: Full brain volumes (512x512x200+) require
   excessive memory and computation for 3D CNNs
2. Information Preservation: 64^3 voxels capture sufficient anatomical
   context for aneurysm detection (typical aneurysm: 3-15mm diameter)
3. Spatial Resolution: At 0.5mm spacing, 64 voxels = 32mm physical size,
   adequate for Circle of Willis coverage
4. Model Compatibility: Most 3D CNN architectures designed for 64^3 inputs
5. Empirical Performance: Achieved 0.8585 AUC with this patch size

Processing Pipeline:
-------------------
1. Load NIfTI volume
2. Apply HU windowing (-100 to 300 for brain CTA)
3. Compute brain mask (thresholding)
4. Extract center-of-mass for ROI centering
5. Extract 64x64x64 patch centered on ROI
6. Apply z-score normalization
7. Save as compressed NPZ file

Input:
    - NIfTI volumes: data/volumes_nifti/*.nii.gz
    - Labels CSV: data/train_labels_14class.csv (optional, for filtering)

Output:
    - Patch files: data/patches_roi/{series_uid}.npz
    - Metadata: data/patch_metadata.csv
    - Extraction log: logs/02_patch_extraction.log

Competition: RSNA 2025 Intracranial Aneurysm Detection
Author: Eric Yebladja
Date: 2025-10-17
"""

import sys
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict
import argparse
from datetime import datetime
import json

import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy import ndimage


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Configure logging to file and console.

    Args:
        output_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    log_dir = output_dir.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "02_patch_extraction.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def apply_cta_windowing(volume: np.ndarray, window_min: int = -100, window_max: int = 300) -> np.ndarray:
    """
    Apply Hounsfield Unit (HU) windowing for brain CTA.

    Standard CTA brain window:
    - Window center: 100 HU
    - Window width: 400 HU
    - Range: [-100, 300] HU

    This enhances visualization of:
    - Blood vessels (50-100 HU with contrast)
    - Brain parenchyma (20-40 HU)
    - CSF (0-15 HU)

    Args:
        volume: 3D array in Hounsfield Units
        window_min: Minimum HU value (default: -100)
        window_max: Maximum HU value (default: 300)

    Returns:
        Windowed volume clipped to [window_min, window_max]
    """
    return np.clip(volume, window_min, window_max)


def compute_brain_mask(volume: np.ndarray, threshold: int = -50) -> np.ndarray:
    """
    Compute binary brain mask using simple thresholding.

    Brain tissue has HU values > -50 (air is -1000, bone is +1000).
    This separates brain from air/background.

    Args:
        volume: 3D volume in HU
        threshold: HU threshold for brain (default: -50)

    Returns:
        Binary mask (True = brain tissue)
    """
    mask = volume > threshold

    # Apply morphological operations to clean mask
    # 1. Close small holes inside brain
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3, 3)))

    # 2. Remove small disconnected regions
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3, 3)))

    return mask


def extract_roi_patch(
    volume: np.ndarray,
    patch_size: int = 64,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract patch centered on region of interest.

    Strategy:
    1. If mask provided, center on center-of-mass of masked region
    2. Otherwise, center on volume center
    3. Extract patch_size^3 voxels
    4. Pad with zeros if patch extends beyond volume boundaries

    Args:
        volume: 3D volume (H, W, D)
        patch_size: Edge length of cubic patch (default: 64)
        mask: Optional binary mask for ROI centering

    Returns:
        patch: Extracted patch (patch_size, patch_size, patch_size)
        center: Center coordinates in original volume (3,)
    """
    # Determine patch center
    if mask is not None and np.any(mask):
        # Center on center-of-mass of masked region
        center = np.array(ndimage.center_of_mass(mask), dtype=np.int32)
    else:
        # Default to volume center
        center = np.array(volume.shape, dtype=np.int32) // 2

    # Calculate patch boundaries
    half_size = patch_size // 2
    starts = np.maximum(center - half_size, 0)
    ends = np.minimum(center + half_size, volume.shape)

    # Extract patch
    patch = volume[starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    # Pad if necessary to ensure (patch_size, patch_size, patch_size)
    if patch.shape != (patch_size, patch_size, patch_size):
        padded = np.zeros((patch_size, patch_size, patch_size), dtype=volume.dtype)

        # Calculate padding offsets
        pad_start = (patch_size - np.array(patch.shape)) // 2
        pad_start = np.maximum(pad_start, 0)

        # Place patch in center of padded array
        padded[
            pad_start[0]:pad_start[0] + patch.shape[0],
            pad_start[1]:pad_start[1] + patch.shape[1],
            pad_start[2]:pad_start[2] + patch.shape[2]
        ] = patch

        patch = padded

    return patch, center


def normalize_patch(patch: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to patch.

    Normalization: (x - mean) / std

    This standardizes patch intensities across different scans,
    accounting for variations in:
    - Scanner protocols
    - Contrast injection timing
    - Patient-specific factors

    Args:
        patch: 3D patch array

    Returns:
        Normalized patch (zero mean, unit variance)
    """
    mean = patch.mean()
    std = patch.std()

    if std < 1e-6:  # Avoid division by zero for empty patches
        return patch - mean

    return (patch - mean) / std


def process_volume(
    nifti_path: Path,
    patch_size: int,
    window_min: int,
    window_max: int,
    logger: logging.Logger
) -> Tuple[np.ndarray, Dict]:
    """
    Process single NIfTI volume and extract ROI patch.

    Args:
        nifti_path: Path to NIfTI file
        patch_size: Patch edge length
        window_min: Minimum HU for windowing
        window_max: Maximum HU for windowing
        logger: Logger instance

    Returns:
        patch: Normalized ROI patch
        metadata: Dictionary with processing metadata

    Raises:
        ValueError: If volume loading or processing fails
    """
    # Load NIfTI volume
    nii = nib.load(str(nifti_path))
    volume = nii.get_fdata().astype(np.float32)

    original_shape = volume.shape
    logger.debug(f"  Loaded volume: {original_shape}")

    # Apply CTA windowing
    volume_windowed = apply_cta_windowing(volume, window_min, window_max)

    # Compute brain mask
    brain_mask = compute_brain_mask(volume_windowed)
    brain_voxels = int(brain_mask.sum())
    logger.debug(f"  Brain mask: {brain_voxels} voxels")

    if brain_voxels < 1000:
        logger.warning(f"  Very small brain mask: {brain_voxels} voxels")

    # Extract ROI patch
    patch, center = extract_roi_patch(volume_windowed, patch_size, brain_mask)

    # Normalize patch
    patch_normalized = normalize_patch(patch)

    # Collect metadata
    metadata = {
        'original_shape': original_shape,
        'patch_size': patch_size,
        'center': center.tolist(),
        'brain_voxels': brain_voxels,
        'patch_mean': float(patch.mean()),
        'patch_std': float(patch.std()),
        'patch_min': float(patch.min()),
        'patch_max': float(patch.max())
    }

    return patch_normalized, metadata


def save_patch(patch: np.ndarray, output_path: Path, metadata: Dict) -> None:
    """
    Save patch as compressed NPZ file with metadata.

    NPZ format advantages:
    - Native numpy format (fast loading)
    - Compression (smaller file size)
    - Multiple arrays per file (patch + metadata)

    Args:
        patch: Normalized patch array
        output_path: Output file path (.npz)
        metadata: Processing metadata dictionary
    """
    np.savez_compressed(
        output_path,
        patch=patch,
        **metadata
    )


def main():
    """Main patch extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract 64^3 voxel ROI patches from NIfTI volumes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--nifti-dir',
        type=str,
        default='data/volumes_nifti',
        help='Directory containing NIfTI volumes'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/patches_roi',
        help='Output directory for patch NPZ files'
    )
    parser.add_argument(
        '--labels-csv',
        type=str,
        default=None,
        help='Optional labels CSV for filtering volumes'
    )
    parser.add_argument(
        '--patch-size',
        type=int,
        default=64,
        help='Patch edge length in voxels'
    )
    parser.add_argument(
        '--window-min',
        type=int,
        default=-100,
        help='Minimum HU for CTA windowing'
    )
    parser.add_argument(
        '--window-max',
        type=int,
        default=300,
        help='Maximum HU for CTA windowing'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of volumes (for testing)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing patch files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup paths
    nifti_dir = Path(args.nifti_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Log configuration
    logger.info("="*80)
    logger.info("ROI Patch Extraction - Step 2")
    logger.info("="*80)
    logger.info(f"Input directory: {nifti_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Patch size: {args.patch_size}^3 voxels")
    logger.info(f"HU window: [{args.window_min}, {args.window_max}]")

    # Validate input
    if not nifti_dir.exists():
        logger.error(f"Input directory does not exist: {nifti_dir}")
        sys.exit(1)

    # Find NIfTI files
    nifti_files = sorted(list(nifti_dir.glob("*.nii.gz")) + list(nifti_dir.glob("*.nii")))

    if args.limit:
        nifti_files = nifti_files[:args.limit]
        logger.info(f"Limited to {args.limit} volumes for testing")

    logger.info(f"Found {len(nifti_files)} NIfTI files")

    if len(nifti_files) == 0:
        logger.error("No NIfTI files found")
        sys.exit(1)

    # Load labels if provided (for filtering or metadata)
    labels_df = None
    if args.labels_csv:
        labels_path = Path(args.labels_csv)
        if labels_path.exists():
            labels_df = pd.read_csv(labels_path)
            logger.info(f"Loaded labels: {len(labels_df)} rows")

    # Process volumes
    start_time = datetime.now()
    success_count = 0
    skipped_count = 0
    failed_list = []
    metadata_records = []

    for nifti_path in tqdm(nifti_files, desc="Extracting patches"):
        series_uid = nifti_path.stem.replace('.nii', '')  # Remove .nii.gz or .nii
        output_path = output_dir / f"{series_uid}.npz"

        # Skip if exists
        if output_path.exists() and not args.overwrite:
            skipped_count += 1
            logger.debug(f"Skipping {series_uid} (already exists)")
            continue

        try:
            # Extract patch
            patch, metadata = process_volume(
                nifti_path,
                args.patch_size,
                args.window_min,
                args.window_max,
                logger
            )

            # Save patch
            save_patch(patch, output_path, metadata)

            # Record metadata
            metadata_records.append({
                'series_uid': series_uid,
                'nifti_file': nifti_path.name,
                'patch_file': output_path.name,
                **metadata
            })

            success_count += 1
            logger.debug(f"Success: {series_uid}")

        except Exception as e:
            error_msg = str(e)
            failed_list.append((series_uid, error_msg))
            logger.error(f"Failed {series_uid}: {error_msg}")
            continue

    # Calculate duration
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Print summary
    logger.info("")
    logger.info("="*80)
    logger.info("EXTRACTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total volumes:          {len(nifti_files)}")
    logger.info(f"Successfully extracted: {success_count}")
    logger.info(f"Skipped (existing):     {skipped_count}")
    logger.info(f"Failed:                 {len(failed_list)}")
    logger.info(f"Success rate:           {100.0 * success_count / len(nifti_files):.2f}%")
    logger.info(f"Processing time:        {duration:.2f} seconds")
    logger.info(f"Average per volume:     {duration / len(nifti_files):.2f} seconds")

    if failed_list:
        logger.info("")
        logger.info(f"Failed volumes (first 10):")
        for series_uid, error in failed_list[:10]:
            logger.info(f"  - {series_uid}: {error}")
        if len(failed_list) > 10:
            logger.info(f"  ... and {len(failed_list) - 10} more")

    # Verify output
    patch_files = list(output_dir.glob('*.npz'))
    logger.info("")
    logger.info(f"Total patch files in output directory: {len(patch_files)}")

    # Save metadata CSV
    if metadata_records:
        metadata_csv = output_dir.parent / "patch_metadata.csv"
        metadata_df = pd.DataFrame(metadata_records)
        metadata_df.to_csv(metadata_csv, index=False)
        logger.info(f"Metadata saved: {metadata_csv}")

        # Print statistics
        logger.info("")
        logger.info("Patch Statistics:")
        logger.info(f"  Mean patch size: {args.patch_size}^3 voxels")
        logger.info(f"  Brain voxels (mean): {metadata_df['brain_voxels'].mean():.0f}")
        logger.info(f"  Brain voxels (std):  {metadata_df['brain_voxels'].std():.0f}")

    logger.info("="*80)

    # Exit with error if too many failures
    failure_rate = len(failed_list) / len(nifti_files)
    if failure_rate > 0.1:
        logger.error(f"High failure rate: {failure_rate*100:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
