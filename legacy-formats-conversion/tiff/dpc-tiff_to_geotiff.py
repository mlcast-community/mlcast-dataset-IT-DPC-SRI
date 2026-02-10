#!/usr/bin/env python3

import os
import glob
import unicodedata
from osgeo import gdal
from pathlib import Path
import fire
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


def _display_width(s):
    """Display width of a string, accounting for wide/emoji characters."""
    w = 0
    for c in s:
        if unicodedata.category(c) in ("Mn", "Cf", "Sk"):
            continue
        if unicodedata.east_asian_width(c) in ("W", "F") or unicodedata.category(c) == "So":
            w += 2
        else:
            w += 1
    return w


def _lpad(s, width):
    """Left-align string s to a given display width."""
    return s + " " * (width - _display_width(s))


# Metadata template matching the ODIM_H5 convention used in the target GeoTIFFs.
# Corner coordinates are fixed for the Italian SRI composite grid.
METADATA = {
    "Conventions": "ODIM_H5/V2_1",
    "dataset1_data1_quality1_what_gain": "0.393701",
    "dataset1_data1_quality1_what_nodata": "0",
    "dataset1_data1_quality1_what_offset": "0",
    "dataset1_data1_quality1_what_quantity": "QIND",
    "dataset1_data1_quality1_what_undetect": "-1",
    "dataset1_data1_what_nodata": "-9999",
    "dataset1_data1_what_quantity": "RATE",
    "dataset1_what_nodata": "-9999",
    "dataset1_what_product": "SURF",
    "dataset1_what_quantity": "RATE",
    "what_object": "COMP",
    "what_version": "H5rad 2.1",
    "where_LL_lat": "35.062255859375",
    "where_LL_lon": "5.92472267150879",
    "where_LR_lat": "35.062255859375",
    "where_LR_lon": "19.0752773284912",
    "where_projdef": "+proj=tmerc +lat_0=42.0 +lon_0=12.5 +ellps=WGS84",
    "where_UL_lat": "47.5729560852051",
    "where_UL_lon": "4.51987266540527",
    "where_UR_lat": "47.5729560852051",
    "where_UR_lon": "20.4801273345947",
    "where_xscale": "1000",
    "where_xsize": "1200",
    "where_yscale": "1000",
    "where_ysize": "1400",
}


def convert_tiff(tiff_path: str, output_dir: str, skip_existing: bool = True) -> tuple[bool, str]:
    """
    Convert a single DPC TIFF to the target GeoTIFF format.

    Source TIFFs are Float32, LZW-compressed, strip-based, with -9999 nodata.
    Target GeoTIFFs are Float16, ZSTD-compressed, tiled, with -1 nodata,
    and ODIM_H5 metadata.

    Args:
        tiff_path: Path to input DPC TIFF file
        output_dir: Base directory for output GeoTIFF files

    Returns:
        tuple: (status: str, input_path: str) where status is "converted", "skipped", or "failed"
    """
    try:
        # Parse date from filename: DD-MM-YYYY-HH-MM.tif
        filename = os.path.basename(tiff_path)
        stem = Path(filename).stem
        day, month, year, hour, minute = map(int, stem.split("-"))

        # Build output path: YYYY/MM/YYYY-MM-DD-HH-MM_ROMA_SRI.tif
        output_filename = f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}_ROMA_SRI.tif"
        relative_output_dir = os.path.join(f"{year:04d}", f"{month:02d}")
        output_path = os.path.join(output_dir, relative_output_dir, output_filename)

        if skip_existing and os.path.exists(output_path):
            return "skipped", tiff_path

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Build time-dependent metadata
        metadata = dict(METADATA)
        metadata["what_date"] = f"{year:04d}{month:02d}{day:02d}"
        metadata["what_time"] = f"{hour:02d}{minute:02d}00"

        creation_options = [
            "COMPRESS=ZSTD",
            "PREDICTOR=2",
            "TILED=YES",
            "NBITS=16",
        ]

        warp_options = gdal.WarpOptions(
            format="GTiff",
            xRes=1000,
            yRes=1000,
            targetAlignedPixels=True,
            creationOptions=creation_options,
            srcNodata=-9999.0,
            dstNodata=-1.0,
            multithread=True,
            resampleAlg=gdal.GRA_NearestNeighbour,
        )

        src_ds = gdal.Open(tiff_path)
        if src_ds is None:
            raise RuntimeError(f"Could not open: {tiff_path}")

        dst_ds = gdal.Warp(output_path, src_ds, options=warp_options)
        if dst_ds is None:
            raise RuntimeError(f"Warp failed for: {tiff_path}")

        # Write ODIM metadata
        dst_ds.SetMetadata(metadata)
        dst_ds.FlushCache()
        dst_ds = None

        return "converted", tiff_path

    except Exception as e:
        # Remove partial output so skip_existing won't treat it as done
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"Error converting {tiff_path}: {e}")
        return "failed", tiff_path


def process_files(
    input_dir: str,
    output_dir: str,
    pattern: str = "**/*.tif",
    skip_existing: bool = True,
    num_workers: int = None,
) -> dict:
    """
    Convert all DPC TIFFs in input_dir to GeoTIFF format.

    Args:
        input_dir: Directory containing source DPC TIFF files
        output_dir: Base directory for output GeoTIFF files
        pattern: Glob pattern for finding TIFF files (default: '**/*.tif')
        num_workers: Number of parallel processes (default: CPU count)

    Returns:
        dict: Summary with success/failed counts
    """
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")

    input_path = os.path.join(input_dir, pattern)
    tiff_files = sorted(glob.glob(input_path, recursive=True))

    if not tiff_files:
        print(f"No TIFF files found in {input_dir} using pattern {pattern}")
        return {"success": 0, "failed": 0, "total": 0}

    if num_workers is None:
        num_workers = mp.cpu_count()

    convert_partial = partial(convert_tiff, output_dir=output_dir, skip_existing=skip_existing)

    print(f"Processing {len(tiff_files)} files using {num_workers} workers...")

    with mp.Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(convert_partial, tiff_files),
                total=len(tiff_files),
                desc="Converting files",
                unit="file",
            )
        )

    converted_count = 0
    skipped_count = 0
    failed_count = 0
    failed_files = []

    for status, filepath in results:
        if status == "converted":
            converted_count += 1
        elif status == "skipped":
            skipped_count += 1
        else:
            failed_count += 1
            failed_files.append(filepath)

    total = converted_count + skipped_count + failed_count

    rows = [
        ("✅ Converted", converted_count),
        ("⏩ Skipped", skipped_count),
        ("❌ Failed", failed_count),
    ]
    val_width = max(len(str(v)) for _, v in rows)
    col1_w, col2_w = 14, max(val_width, 5)
    sep = f"╠{'═' * (col1_w + 2)}╬{'═' * (col2_w + 2)}╣"

    print(f"\n📡🌧️  Conversion Report")
    print(f"╔{'═' * (col1_w + 2)}╦{'═' * (col2_w + 2)}╗")
    print(f"║ {'Status':<{col1_w}} ║ {'Count':>{col2_w}} ║")
    print(sep)
    for label, count in rows:
        print(f"║ {_lpad(label, col1_w)} ║ {count:>{col2_w}} ║")
    print(sep)
    print(f"║ {_lpad('📊 Total', col1_w)} ║ {total:>{col2_w}} ║")
    print(f"╚{'═' * (col1_w + 2)}╩{'═' * (col2_w + 2)}╝")

    if failed_files:
        print(f"\n💀 Failed files ({failed_count}):")
        for filepath in failed_files:
            print(f"   ↳ {filepath}")

    return {
        "converted": converted_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "total": total,
        "failed_files": failed_files,
    }


def main(
    input_dir: str,
    output_dir: str,
    pattern: str = "**/*.tif",
    skip_existing: bool = True,
    num_workers: int = None,
):
    """
    Convert DPC FTP radar TIFFs to GeoTIFF format matching the mlcast dataset spec.

    Args:
        input_dir: Directory containing source DPC TIFF files
        output_dir: Base directory for output GeoTIFF files
        pattern: Glob pattern for finding TIFF files (default: '**/*.tif')
        num_workers: Number of parallel processes (default: CPU count)
    """
    return process_files(input_dir, output_dir, pattern, skip_existing, num_workers)


if __name__ == "__main__":
    gdal.UseExceptions()
    fire.Fire(main)
