#!/usr/bin/env python3
"""
Step 1: DICOM to NIfTI Volume Conversion
==========================================

Convert raw DICOM series from the RSNA 2025 Intracranial Aneurysm Detection
competition to standardized NIfTI format volumes.

This script:
1. Reads DICOM series from organized directories
2. Sorts slices by spatial position (ImagePositionPatient)
3. Converts to Hounsfield Units (HU)
4. Preserves spatial metadata (spacing, affine transform)
5. Saves as compressed NIfTI (.nii.gz) for efficient storage

Input:
    - Raw DICOM series directories (default: data/raw/series/)
    - Each series in format: series/{series_uid}/*.dcm

Output:
    - NIfTI volumes: data/volumes_nifti/{series_uid}.nii.gz
    - Conversion log: logs/01_dicom_conversion.log
    - Success/failure report

Competition: RSNA 2025 Intracranial Aneurysm Detection
Link: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
Author: Eric Yebladja
Date: 2025-10-17
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import argparse
from datetime import datetime
import json

import numpy as np
import pydicom
import nibabel as nib
from tqdm import tqdm


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

    log_file = log_dir / "01_dicom_conversion.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def sort_dicom_slices(slices: List[pydicom.Dataset]) -> List[pydicom.Dataset]:
    """
    Sort DICOM slices by ImagePositionPatient Z-coordinate.

    In CT angiography, slices are acquired sequentially through the patient.
    The ImagePositionPatient[2] (Z-coordinate) indicates the slice position
    along the superior-inferior axis.

    Args:
        slices: List of pydicom Dataset objects

    Returns:
        Sorted list of DICOM slices (inferior to superior)

    Raises:
        ValueError: If no slices have valid ImagePositionPatient
    """
    slices_with_pos = []

    for slice_ds in slices:
        try:
            # Extract Z position (superior-inferior axis)
            z_pos = float(slice_ds.ImagePositionPatient[2])
            slices_with_pos.append((z_pos, slice_ds))
        except (AttributeError, IndexError, ValueError):
            # Skip slices without valid position metadata
            continue

    if not slices_with_pos:
        raise ValueError("No DICOM slices contain valid ImagePositionPatient")

    # Sort by Z position (inferior to superior)
    slices_with_pos.sort(key=lambda x: x[0])

    return [slice_ds for _, slice_ds in slices_with_pos]


def process_dicom_series(
    series_dir: Path,
    logger: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load DICOM series and convert to 3D volume with spatial metadata.

    Processing steps:
    1. Read all .dcm files in directory
    2. Sort by spatial position
    3. Stack into 3D numpy array
    4. Apply RescaleSlope and RescaleIntercept (convert to HU)
    5. Extract voxel spacing and create affine transform

    Args:
        series_dir: Path to DICOM series directory
        logger: Logger instance

    Returns:
        volume: 3D numpy array (H, W, D) in Hounsfield Units
        affine: 4x4 affine transformation matrix (voxel to mm)

    Raises:
        ValueError: If no valid DICOM files found or sorting fails
    """
    # Find all DICOM files
    dicom_files = list(series_dir.glob("*.dcm"))

    if not dicom_files:
        raise ValueError(f"No .dcm files found in {series_dir}")

    # Read DICOM slices
    slices = []
    read_errors = 0

    for dcm_file in dicom_files:
        try:
            slices.append(pydicom.dcmread(str(dcm_file), force=True))
        except Exception as e:
            read_errors += 1
            logger.debug(f"Could not read {dcm_file.name}: {e}")
            continue

    if not slices:
        raise ValueError(f"No valid DICOM slices (read errors: {read_errors})")

    if read_errors > 0:
        logger.debug(f"  Read errors: {read_errors}/{len(dicom_files)} files")

    # Sort slices by spatial position
    try:
        slices = sort_dicom_slices(slices)
    except ValueError as e:
        raise ValueError(f"Failed to sort slices: {e}")

    # Stack pixel arrays into 3D volume
    volume = np.stack([s.pixel_array for s in slices], axis=-1)
    logger.debug(f"  Volume shape: {volume.shape}")

    # Convert to Hounsfield Units (HU)
    # HU = RescaleSlope * pixel_value + RescaleIntercept
    try:
        slope = float(slices[0].RescaleSlope)
        intercept = float(slices[0].RescaleIntercept)
        volume = volume.astype(np.float32) * slope + intercept
        logger.debug(f"  Applied HU conversion: slope={slope}, intercept={intercept}")
    except (AttributeError, ValueError) as e:
        logger.warning(f"  No RescaleSlope/Intercept, keeping raw values: {e}")

    # Extract voxel spacing and create affine transformation
    try:
        # In-plane pixel spacing [row_spacing, col_spacing] in mm
        pixel_spacing = slices[0].PixelSpacing
        px_x, px_y = float(pixel_spacing[0]), float(pixel_spacing[1])

        # Slice spacing (Z-axis) in mm
        if len(slices) > 1:
            # Calculate from consecutive slice positions
            z1 = float(slices[0].ImagePositionPatient[2])
            z2 = float(slices[1].ImagePositionPatient[2])
            px_z = abs(z2 - z1)
        else:
            # Fallback to SliceThickness attribute
            px_z = float(slices[0].get('SliceThickness', 1.0))

        # Create affine transformation (voxel to physical mm space)
        affine = np.diag([px_x, px_y, px_z, 1.0])
        logger.debug(f"  Voxel spacing: ({px_x:.2f}, {px_y:.2f}, {px_z:.2f}) mm")

    except (AttributeError, IndexError, ValueError) as e:
        logger.warning(f"  Could not extract spacing, using identity: {e}")
        affine = np.eye(4)

    return volume, affine


def save_conversion_report(
    output_file: Path,
    total: int,
    success: int,
    skipped: int,
    failed: List[Tuple[str, str]],
    duration_sec: float
) -> None:
    """
    Save conversion summary to JSON file.

    Args:
        output_file: Path to JSON report file
        total: Total series processed
        success: Successfully converted
        skipped: Already existed
        failed: List of (series_uid, error_message)
        duration_sec: Processing duration in seconds
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "statistics": {
            "total_series": total,
            "successful": success,
            "skipped_existing": skipped,
            "failed": len(failed)
        },
        "success_rate": f"{100.0 * success / total:.2f}%" if total > 0 else "N/A",
        "processing_time_sec": round(duration_sec, 2),
        "failed_series": [
            {"series_uid": uid, "error": error}
            for uid, error in failed[:50]  # Limit to first 50 failures
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)


def main():
    """Main conversion pipeline."""
    parser = argparse.ArgumentParser(
        description="Convert DICOM series to NIfTI volumes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--series-dir',
        type=str,
        default='data/raw/series',
        help='Directory containing DICOM series folders'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/volumes_nifti',
        help='Output directory for NIfTI volumes'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of series (for testing)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing NIfTI files'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )

    args = parser.parse_args()

    # Setup paths
    series_dir = Path(args.series_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(output_dir)
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Log configuration
    logger.info("="*80)
    logger.info("DICOM to NIfTI Conversion - Step 1")
    logger.info("="*80)
    logger.info(f"Input directory: {series_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Overwrite existing: {args.overwrite}")

    # Validate input directory
    if not series_dir.exists():
        logger.error(f"Input directory does not exist: {series_dir}")
        sys.exit(1)

    # Find all series directories
    series_folders = sorted([d for d in series_dir.iterdir() if d.is_dir()])

    if args.limit:
        series_folders = series_folders[:args.limit]
        logger.info(f"Limited to {args.limit} series for testing")

    logger.info(f"Found {len(series_folders)} series directories")

    if len(series_folders) == 0:
        logger.error("No series directories found")
        sys.exit(1)

    # Process each series
    start_time = datetime.now()
    success_count = 0
    skipped_count = 0
    failed_list = []

    for series_folder in tqdm(series_folders, desc="Converting DICOM to NIfTI"):
        series_uid = series_folder.name
        output_path = output_dir / f"{series_uid}.nii.gz"

        # Skip existing files unless overwrite enabled
        if output_path.exists() and not args.overwrite:
            skipped_count += 1
            logger.debug(f"Skipping {series_uid} (already exists)")
            continue

        try:
            # Convert DICOM series to volume
            volume, affine = process_dicom_series(series_folder, logger)

            # Save as NIfTI
            nifti_img = nib.Nifti1Image(volume, affine)
            nib.save(nifti_img, str(output_path))

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
    logger.info("CONVERSION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total series:          {len(series_folders)}")
    logger.info(f"Successfully converted: {success_count}")
    logger.info(f"Skipped (existing):    {skipped_count}")
    logger.info(f"Failed:                {len(failed_list)}")
    logger.info(f"Success rate:          {100.0 * success_count / len(series_folders):.2f}%")
    logger.info(f"Processing time:       {duration:.2f} seconds")
    logger.info(f"Average per series:    {duration / len(series_folders):.2f} seconds")

    if failed_list:
        logger.info("")
        logger.info(f"Failed series (first 10):")
        for series_uid, error in failed_list[:10]:
            logger.info(f"  - {series_uid}: {error}")
        if len(failed_list) > 10:
            logger.info(f"  ... and {len(failed_list) - 10} more (see log file)")

    # Verify output
    nifti_files = list(output_dir.glob('*.nii.gz'))
    logger.info("")
    logger.info(f"Total NIfTI files in output directory: {len(nifti_files)}")

    # Save detailed report
    report_file = output_dir.parent / "conversion_report.json"
    save_conversion_report(
        report_file,
        len(series_folders),
        success_count,
        skipped_count,
        failed_list,
        duration
    )
    logger.info(f"Detailed report saved: {report_file}")
    logger.info("="*80)

    # Exit with error code if too many failures
    failure_rate = len(failed_list) / len(series_folders)
    if failure_rate > 0.1:  # More than 10% failures
        logger.error(f"High failure rate: {failure_rate*100:.1f}%")
        sys.exit(1)


if __name__ == "__main__":
    main()
