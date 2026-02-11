#!/usr/bin/env python3
"""
Convert a directory of GeoTIFF radar files into a single mlcast-compliant Zarr v2 store.

Produces a zarr that passes:
    mlcast.validate_dataset source_data radar_precipitation <zarr_path>
"""

import os
import glob
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from datetime import datetime, timezone
from functools import partial
from osgeo import gdal
import multiprocessing as mp
from numcodecs import Blosc
from tqdm import tqdm
from loguru import logger
from fire import Fire


# ---------------------------------------------------------------------------
# CRS constants (Italian DPC SRI composite, Transverse Mercator)
# ---------------------------------------------------------------------------
CRS_WKT = (
    'PROJCRS["unknown",'
    'BASEGEOGCRS["unknown",'
    'DATUM["Unknown based on WGS 84 ellipsoid",'
    'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1],ID["EPSG",7030]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],'
    'CONVERSION["Transverse Mercator",'
    'METHOD["Transverse Mercator",ID["EPSG",9807]],'
    'PARAMETER["Latitude of natural origin",42,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8801]],'
    'PARAMETER["Longitude of natural origin",12.5,ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8802]],'
    'PARAMETER["Scale factor at natural origin",1,SCALEUNIT["unity",1],ID["EPSG",8805]],'
    'PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],'
    'PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],'
    'CS[Cartesian,2],'
    'AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],'
    'AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]],'
    'BBOX[35.062255859375,4.51987266540527,47.5729560852051,20.4801273345947]]'
)
PROJ4 = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs"


# ---------------------------------------------------------------------------
# CF time encoding
# ---------------------------------------------------------------------------
CF_TIME_UNITS = "minutes since 2010-01-01"
CF_TIME_CALENDAR = "proleptic_gregorian"
_CF_EPOCH = np.datetime64("2010-01-01", "ns")
_CF_STEP = np.timedelta64(1, "m")


def _cf_encode_times(dt_array: np.ndarray) -> np.ndarray:
    """Encode datetime64[ns] array to int64 minutes since CF_EPOCH."""
    return ((dt_array - _CF_EPOCH) / _CF_STEP).astype(np.int64)


# ---------------------------------------------------------------------------
# Base frequency bands (specific to IT-DPC-SRI dataset)
# Each entry: (frequency_minutes, band_start, band_end_or_None)
# None means the band extends to the dataset end_date.
# ---------------------------------------------------------------------------
BASE_FREQUENCIES = [
    (15, "2010-01-01T00:00", "2014-06-25T09:00"),
    (10, "2014-06-25T09:00", "2020-06-30T00:00"),
    (5,  "2020-06-30T00:00", None),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_timestamp(filepath: str) -> np.datetime64:
    """Parse 'YYYY-MM-DD-HH-MM_ROMA_SRI.tif' into np.datetime64[ns]."""
    basename = os.path.basename(filepath).split("_")[0]
    year, month, day, hour, minute = basename.split("-")
    return np.datetime64(f"{year}-{month}-{day}T{hour}:{minute}", "ns")


def read_tif(filepath: str, fillvalue: float = -1.0) -> np.ndarray:
    """Read a single GeoTIFF band, replacing fillvalue with NaN. Returns (1, y, x)."""
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    arr = ds.GetRasterBand(1).ReadAsArray()
    arr = np.where(arr == fillvalue, np.nan, arr)
    return np.expand_dims(arr, axis=0)


def _write_chunk(idx, tif_paths, zarr_path, time_indices, batch_size):
    """Write a batch of TIFs into the zarr RR array at the correct time positions."""
    try:
        batch_paths = tif_paths[idx : idx + batch_size]
        batch_indices = time_indices[idx : idx + batch_size]
        data = np.concatenate([read_tif(p) for p in batch_paths], axis=0)
        root = zarr.open_group(zarr_path, mode="r+")
        for i, time_idx in enumerate(batch_indices):
            root["RR"][time_idx, ...] = data[i]
    except Exception as e:
        paths = tif_paths[idx : idx + batch_size]
        print(f"Error writing {paths}: {e}")


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------
class ZarrConverter:
    """Create and fill a mlcast-compliant zarr store from GeoTIFF files."""

    def __init__(
        self,
        data_path: str,
        save_path: str,
        reference_tif_file: str,
        grid_coords: str,
        start_date: str = "2010-01-01T00:00",
        end_date: str = "2026-01-01T00:00",
        pattern: str = "**/*.tif",
        num_workers: int = 32,
        batch_size: int = 1,
        timerange_split: int = 1000,
        compression_level: int = 9,
        shuffle: int = 0,
    ):
        self.data_path = os.path.abspath(data_path)
        self.save_path = os.path.abspath(save_path)
        self.pattern = pattern
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.timerange_split = timerange_split
        self.start_date = np.datetime64(start_date, "ns")
        self.end_date = np.datetime64(end_date, "ns")
        self.compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=shuffle)

        # Read reference TIF for coordinate grids
        logger.info(f"Loading reference TIF: {reference_tif_file}")
        self.ref_ds = xr.open_dataset(reference_tif_file, engine="rasterio")
        logger.info(f"Loading grid coords: {grid_coords}")
        self.grid = gpd.read_file(grid_coords)

        # Run pipeline
        self.discover_tifs()
        self.build_time_array()
        self.init_zarr()
        self.fill_data()
        self.finalize()

    def discover_tifs(self):
        """Find and sort all TIF files, parse their timestamps."""
        search = os.path.join(self.data_path, self.pattern)
        self.tif_paths = sorted(glob.glob(search, recursive=True))
        logger.info(f"Found {len(self.tif_paths)} TIF files")

        if not self.tif_paths:
            raise FileNotFoundError(f"No TIF files at {search}")

        self.tif_timestamps = np.array(
            [parse_timestamp(p) for p in self.tif_paths], dtype="datetime64[ns]"
        )

        # Build lookup: timestamp -> list index in tif_paths
        self.tif_lookup = {}
        for i, ts in enumerate(self.tif_timestamps):
            self.tif_lookup[ts] = i

    def build_time_array(self):
        """Build the time array (data-only) and compute missing_times.

        The time coordinate contains ONLY timestamps that have actual TIF data
        within [start_date, end_date).

        missing_times lists expected-grid timestamps (across all frequency
        bands defined in BASE_FREQUENCIES) that are ABSENT from the time
        coordinate.
        """
        tif_ts_set = set(self.tif_timestamps.tolist())

        # Time coord = all TIF timestamps within [start_date, end_date)
        mask = (self.tif_timestamps >= self.start_date) & (
            self.tif_timestamps < self.end_date
        )
        self.full_time = self.tif_timestamps[mask]
        logger.info(f"Time array: {len(self.full_time)} timesteps (data-only)")
        logger.info(f"Time range: {self.full_time[0]} to {self.full_time[-1]}")

        # Compute missing_times across all frequency bands
        all_missing = []
        for freq_min, band_start_str, band_end_str in BASE_FREQUENCIES:
            band_start = np.datetime64(band_start_str, "ns")
            band_end = (
                np.datetime64(band_end_str, "ns") if band_end_str else self.end_date
            )

            # Clip to requested [start_date, end_date)
            eff_start = max(band_start, self.start_date)
            eff_end = min(band_end, self.end_date)
            if eff_start >= eff_end:
                continue

            freq = np.timedelta64(freq_min, "m")
            expected = np.arange(eff_start, eff_end, freq).astype("datetime64[ns]")
            expected_set = set(expected.tolist())
            missing_in_band = expected_set - tif_ts_set
            all_missing.extend(missing_in_band)

            n_present = len(expected_set) - len(missing_in_band)
            logger.info(
                f"  {freq_min}min band [{eff_start} -> {eff_end}]: "
                f"{len(expected)} expected, {n_present} present, "
                f"{len(missing_in_band)} missing"
            )

        self.missing_times = np.array(sorted(all_missing), dtype="datetime64[ns]")
        logger.info(f"Total missing timestamps: {len(self.missing_times)}")

        # Build mapping: TIF path -> index in self.full_time
        time_to_idx = {ts: i for i, ts in enumerate(self.full_time.tolist())}
        self.tif_write_paths = []
        self.tif_write_indices = []
        for path, ts in zip(self.tif_paths, self.tif_timestamps):
            ts_val = ts.item()
            if ts_val in time_to_idx:
                self.tif_write_paths.append(path)
                self.tif_write_indices.append(time_to_idx[ts_val])

        logger.info(f"TIFs to write: {len(self.tif_write_paths)}")

    def init_zarr(self):
        """Create the zarr v2 group with all coordinates, variables, and metadata."""
        logger.info("Initialising zarr store")

        n_y = self.grid["y"].unique().shape[0]
        n_x = self.grid["x"].unique().shape[0]
        n_time = len(self.full_time)
        out_lat = self.grid["geometry"].y.values.reshape(n_y, n_x)
        out_lon = self.grid["geometry"].x.values.reshape(n_y, n_x)

        store = zarr.DirectoryStore(self.save_path)
        self.root = zarr.open_group(store, mode="w")

        # x, y coordinates
        for coord in ("x", "y"):
            self.root.create_dataset(coord, data=self.ref_ds[coord].values)
            self.root[coord].attrs["_ARRAY_DIMENSIONS"] = [coord]
            self.root[coord].attrs["units"] = "m"

        # lat, lon (2D)
        for name, arr, std, ln, units in [
            ("lat", out_lat, "latitude", "Latitude", "degrees_north"),
            ("lon", out_lon, "longitude", "Longitude", "degrees_east"),
        ]:
            self.root.create_dataset(name, data=arr, chunks=(n_y, n_x))
            self.root[name].attrs.update({
                "_ARRAY_DIMENSIONS": ["y", "x"],
                "grid_mapping": "crs",
                "long_name": ln,
                "standard_name": std,
                "units": units,
            })

        # time coordinate (CF-encoded as int64 minutes)
        # fill_value=None avoids 0 colliding with the epoch (2010-01-01T00:00)
        time_encoded = _cf_encode_times(self.full_time)
        self.root.create_dataset(
            "time", data=time_encoded, chunks=(n_time,), fill_value=None,
        )
        self.root["time"].attrs.update({
            "_ARRAY_DIMENSIONS": ["time"],
            "long_name": "Time",
            "standard_name": "time",
            "units": CF_TIME_UNITS,
            "calendar": CF_TIME_CALENDAR,
        })

        # missing_times (CF-encoded as int64 minutes)
        mt_encoded = _cf_encode_times(self.missing_times)
        mt_len = max(len(mt_encoded), 1)  # zarr needs chunks >= 1
        self.root.create_dataset(
            "missing_times", data=mt_encoded, chunks=(mt_len,), fill_value=None,
        )
        self.root["missing_times"].attrs.update({
            "_ARRAY_DIMENSIONS": ["missing_times"],
            "units": CF_TIME_UNITS,
            "calendar": CF_TIME_CALENDAR,
        })

        # RR data variable
        self.root.create_dataset(
            "RR",
            shape=(n_time, n_y, n_x),
            compressor=self.compressor,
            dtype="float32",
            chunks=(1, n_y, n_x),
            fill_value=np.nan,
            write_empty_chunks=False,
        )
        self.root["RR"].attrs.update({
            "_ARRAY_DIMENSIONS": ["time", "y", "x"],
            "grid_mapping": "crs",
            "long_name": "Total precipitation rate",
            "standard_name": "rainfall_flux",
            "units": "kg m-2 h-1",
        })

        # CRS variable
        crs_attrs = self._build_crs_attrs()
        self.root.create_dataset("crs", data=np.array(np.nan, dtype=np.float32))
        self.root["crs"].attrs.update({"_ARRAY_DIMENSIONS": [], **crs_attrs})

        # Global attributes
        curr_date = (
            datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        )
        # consistent_timestep_start = start of the finest (last) frequency band
        consistent_ts_start = BASE_FREQUENCIES[-1][1]
        # base_frequencies as plain string (netCDF-safe, immune to JSON auto-decode)
        # Format: "15min:2010-01-01T00:00/2014-06-25T09:00;10min:..."
        base_freq_parts = []
        for freq_min, bstart, bend in BASE_FREQUENCIES:
            end_val = bend if bend else str(self.end_date)
            base_freq_parts.append(f"{freq_min}min:{bstart}/{end_val}")
        base_freq_str = ";".join(base_freq_parts)

        self.root.attrs.update({
            "Author": "Gabriele Franch",
            "history": f"Created at {curr_date}",
            "Copyright": "Dipartimento Protezione Civile (DPC) Nazionale rete RADAR",
            "license": "CC-BY-SA-4.0",
            "Processed by": "Fondazione Bruno Kessler",
            "title": "Italian Radar DPC Surface Rainfall Intensity (SRI) Archive",
            "consistent_timestep_start": consistent_ts_start,
            "base_frequencies": base_freq_str,
            "mlcast_created_on": curr_date,
            "mlcast_created_by": "Gabriele Franch <franch@fbk.eu>",
            "mlcast_created_with": "https://github.com/mlcast-community/mlcast-dataset-IT-DPC-SRI@0.1.0",
            "mlcast_dataset_version": "0.1.0",
            "mlcast_dataset_identifier": "IT-DPC-SRI",
            "coordinates": "lat lon",
        })

        zarr.consolidate_metadata(zarr.DirectoryStore(self.save_path))
        logger.info(f"Zarr initialised: {n_time} timesteps, {n_y}x{n_x} grid")

    def _build_crs_attrs(self) -> dict:
        """Extract CRS attributes from the reference TIF and override projection info."""
        general_attrs = dict(self.ref_ds.spatial_ref.attrs)
        band_attrs = dict(
            iter(self.ref_ds._attr_sources).__next__()["band_data"].attrs
        )
        band_attrs.update(general_attrs)

        # Remove ODIM metadata keys
        for k in list(band_attrs.keys()):
            if k.startswith("what"):
                del band_attrs[k]
        band_attrs.pop("where_projdef", None)

        # Override with correct projection strings
        band_attrs["proj4"] = PROJ4
        band_attrs["crs_wkt"] = CRS_WKT
        band_attrs["spatial_ref"] = CRS_WKT

        return band_attrs

    def fill_data(self):
        """Write TIF data into the zarr RR array using multiprocessing."""
        n = len(self.tif_write_paths)
        if n == 0:
            logger.warning("No TIF data to write")
            return

        split = self.timerange_split
        slices = list(range(0, n, split))

        for i, start in enumerate(slices):
            end = min(start + split, n)
            batch_paths = self.tif_write_paths[start:end]
            batch_indices = self.tif_write_indices[start:end]

            logger.info(
                f"Part {i + 1}/{len(slices)}: writing TIFs {start}-{end} "
                f"({batch_paths[0]} ... {batch_paths[-1]})"
            )

            write_fn = partial(
                _write_chunk,
                tif_paths=batch_paths,
                zarr_path=self.save_path,
                time_indices=batch_indices,
                batch_size=self.batch_size,
            )

            idxs = list(range(0, len(batch_paths), self.batch_size))
            with mp.Pool(self.num_workers) as pool:
                list(tqdm(
                    pool.imap(write_fn, idxs),
                    total=len(idxs),
                    desc="Writing",
                    unit="batch",
                ))

    def finalize(self):
        """Consolidate zarr metadata."""
        logger.info("Consolidating metadata")
        store = zarr.DirectoryStore(self.save_path)
        zarr.consolidate_metadata(store)
        logger.info(f"Done. Zarr stored at: {self.save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(
    data_path: str,
    save_path: str,
    reference_tif_file: str = "data/2023-01-01-00-00_ROMA_SRI.tif",
    grid_coords: str = "data/reprojected_radar_dpc_grid_yx.geojson",
    start_date: str = "2010-01-01T00:00",
    end_date: str = "2026-01-01T00:00",
    pattern: str = "**/*.tif",
    num_workers: int = 32,
    batch_size: int = 1,
    timerange_split: int = 1000,
    compression_level: int = 9,
    shuffle: int = 0,
):
    """
    Convert a directory of GeoTIFF radar files into a mlcast-compliant Zarr v2 store.

    Frequency bands for missing-timestamp detection are defined in
    BASE_FREQUENCIES (module constant).

    Args:
        data_path: Directory containing TIF files
        save_path: Output zarr path
        reference_tif_file: Reference TIF for CRS/coordinate extraction
        grid_coords: GeoJSON file with the lat/lon grid
        start_date: Start of the time array (ISO 8601)
        end_date: End of the time array (ISO 8601), can extend past last TIF
        pattern: Glob pattern for finding TIF files
        num_workers: Number of parallel workers
        batch_size: TIFs to read per batch
        timerange_split: Split large writes into chunks of this size
        compression_level: Blosc ZSTD compression level (0-9)
        shuffle: Blosc shuffle filter (0=none, 1=byte, 2=bit)
    """
    t0 = time.time()
    ZarrConverter(
        data_path=data_path,
        save_path=save_path,
        reference_tif_file=reference_tif_file,
        grid_coords=grid_coords,
        start_date=start_date,
        end_date=end_date,
        pattern=pattern,
        num_workers=num_workers,
        batch_size=batch_size,
        timerange_split=timerange_split,
        compression_level=compression_level,
        shuffle=shuffle,
    )
    elapsed = (time.time() - t0) / 60
    logger.info(f"Total time: {elapsed:.1f} min")


if __name__ == "__main__":
    gdal.UseExceptions()
    Fire(main)
