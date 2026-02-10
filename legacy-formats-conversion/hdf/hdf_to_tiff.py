#!/usr/bin/env python3

import os
import glob
from osgeo import gdal
import datetime
from pathlib import Path
import fire
import multiprocessing as mp
from tqdm import tqdm
from functools import partial


def read_hdf(hdf_path):
    """ Return gdal.Dataset class of the correct radar band"""
    hdf_dataset = gdal.Open(hdf_path)
    if hdf_dataset is None:
        raise Exception(f"Could not open HDF file: {hdf_path}")
        
    subdatasets = hdf_dataset.GetSubDatasets()
    if not subdatasets:
        raise Exception(f"No subdatasets found in HDF file: {hdf_path}")
        
    # Open the first valid subdataset 
    for i in range(len(subdatasets)):
        dataset_desc = subdatasets[i][1]
        if ("1400x1200" in dataset_desc) and ("32-bit floating-point" in dataset_desc):
            subdataset = subdatasets[i][0]
            found = True
            break
    if not found:
        raise Exception(f"No valid subdataset with matching size and dtype in HDF file: {hdf_path}")

    return gdal.Open(subdataset)


def get_reference_attrs(reference_hdf_path):
    """ Return all necessary attributes to be used in corrutped files """
    subdataset = read_hdf(reference_hdf_path)
    return {
        "proj4": subdataset.GetProjection(),
        "geotransform": subdataset.GetGeoTransform(),
    }


def convert_hdf_to_tiff(hdf_path: str,
                        output_dir: str,
                        reference_attrs: dict = None,
                        ) -> bool:
    """
    Convert a single HDF file to GeoTIFF format using gdal.Warp with 1km resolution.
    
    Args:
        hdf_path: Path to input HDF file
        output_dir: Base directory for output TIFF files
        reference_attrs: attributes from uncorrupted hdf 
    
    Returns:
        tuple: (bool: True if successful, str: input file path)
    """
    try:
        # Extract date components from filename
        filename = os.path.basename(hdf_path)
        date_part = filename.split('_')[1].split('.')[0]  # Get 31-12-2022-23-50
        day, month, year, hour, minute = map(int, date_part.split('-'))
        
        # Create output filename
        output_filename = f"{year:04d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}_ROMA_SRI.tif"
        
        # Custom name for 2025 gap filling
        # From 2025 onwards .tif filenames seems to have been modified
        # Reoridering reference date into DD-MM-YYYY-hh-mm
        # output_filename = f"{day:02d}-{month:02d}-{year:04d}-{hour:02d}-{minute:02d}.tif"

        relative_output_dir = os.path.join(f"{year:04d}", f"{month:02d}")
        output_path = os.path.join(output_dir, relative_output_dir, output_filename)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Read HDF path
        subdataset = read_hdf(hdf_path)
    
        # Check for missing proj4
        if subdataset.GetProjection() == "":
            subdataset.SetProjection(reference_attrs["proj4"])

        # Check for missing GeoTransform
        if subdataset.GetGeoTransform() != reference_attrs["geotransform"]:
            subdataset.SetGeoTransform(reference_attrs["geotransform"])

        # Set creation options for the output GeoTIFF
        creation_options = [
            'COMPRESS=ZSTD',    # ZSTD compression
            'PREDICTOR=2',      # Horizontal predictor
            'TILED=YES',        # Enable tiling
            'NBITS=16',         # 16-bit data
        ]
        
        # Set warp options
        warp_options = gdal.WarpOptions(
            format='GTiff',
            xRes=1000,          # X resolution in meters
            yRes=1000,          # Y resolution in meters
            targetAlignedPixels=True,  # Align pixels with the target dataset
            creationOptions=creation_options,
            srcNodata=-9999.,    # Source nodata value
            dstNodata=-1.,       # Destination nodata value (alternatively: -9999.)
            multithread=True,   # Enable multithreading for warp operation
            resampleAlg=gdal.GRA_NearestNeighbour  # Use nearest neighbor resampling
        )
        
        # Perform warping operation
        gdal.Warp(
            output_path,
            subdataset,
            options=warp_options
        )
        
        return True, hdf_path
                
    except Exception as e:
        print(f"Error converting {hdf_path}: {str(e)}")
        return False, hdf_path


def process_files(input_dir: str = "hdf", 
                 output_dir: str = "tiff", 
                 pattern: str = "**/*.hdf",
                 reference_file: str = "/disks/fast/2023/09/14/SRI_14-09-2023-09-45.hdf",
                 num_workers: int = None) -> dict:
    """
    Process all HDF files in the input directory and convert them to GeoTIFF.
    
    Args:
        input_dir: Base directory containing HDF files (default: 'hdf')
        output_dir: Base directory for output TIFF files (default: 'tiff')
        pattern: Glob pattern for finding HDF files (default: '**/*.hdf')
        reference_file: example of uncorrupted hdf
        num_workers: Number of parallel processes (default: CPU count)
        
    Returns:
        dict: Summary of processing results
    """
    # Ensure input directory exists
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Use glob to find all HDF files
    input_path = os.path.join(input_dir, pattern)
    hdf_files = glob.glob(input_path, recursive=True)
    hdf_files.sort()
    
    if not hdf_files:
        print(f"No HDF files found in {input_dir} using pattern {pattern}")
        return {"success": 0, "failed": 0, "total": 0}
    
    # Read reference attributes in order to tackle missing attrs
    print(f"Reading reference file: {reference_file}")
    reference_attrs = get_reference_attrs(reference_file)
    print(reference_attrs)

    # Set number of workers
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Create partial function with fixed output_dir and reference_attrs
    convert_partial = partial(
        convert_hdf_to_tiff, output_dir=output_dir, reference_attrs=reference_attrs
        )
    
    # Initialize counters
    success_count = 0
    failed_count = 0
    failed_files = []
    
    # Process files in parallel with progress bar
    print(f"\nProcessing {len(hdf_files)} files using {num_workers} workers...")
    
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(convert_partial, hdf_files),
            total=len(hdf_files),
            desc="Converting files",
            unit="file"
        ))
    
    # Count results
    for success, filepath in results:
        if success:
            success_count += 1
        else:
            failed_count += 1
            failed_files.append(filepath)
    
    # Print summary
    total = success_count + failed_count
    print(f"\nConversion Summary:")
    print(f"Total files processed: {total}")
    print(f"Successfully converted: {success_count}")
    print(f"Failed conversions: {failed_count}")
    
    if failed_files:
        print("\nFailed files:")
        for filepath in failed_files:
            print(f"- {filepath}")
    
    return {
        "success": success_count,
        "failed": failed_count,
        "total": total,
        "failed_files": failed_files
    }

def main(input_dir: str, 
         output_dir: str, 
         pattern: str = "**/*.hdf",
         reference_file: str = "/disks/fast/2023/09/14/SRI_14-09-2023-09-45.hdf",
         num_workers: int = None):
    """
    Command-line tool to convert HDF files to GeoTIFF format.
    
    Args:
        input_dir: Base directory containing HDF files (default: 'hdf')
        output_dir: Base directory for output TIFF files (default: 'tiff')
        pattern: Glob pattern for finding HDF files (default: '**/*.hdf')
        num_workers: Number of parallel processes (default: CPU count)
    """
    return process_files(input_dir, output_dir, pattern, reference_file, num_workers)

if __name__ == "__main__":
    # Enable GDAL exceptions
    gdal.UseExceptions()
    fire.Fire(main)