import os
import glob
import itertools
import zarr
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

from zarr.storage import LocalStore 
from datetime import datetime, timezone
from tqdm import tqdm
from osgeo import gdal
from pyproj import CRS
from numcodecs import Blosc as Codec 
from pyproj import Transformer
from loguru import logger
from fire import Fire
from typing import Dict, List, Any


class InitZarr:
    """Init Empty zarr storage."""

    def __init__(
        self,
        save_path: str,
        reference_tif_file: str,
        grid_coords: str,
        data_path: str = None,
        chunk_size: int = 1,
        pattern: str = "**/*.tif",
        crs_wkt: str = None,
        proj4: str = None,
        start_date: str = None,
        end_date: str = None,
        time_freq_minutes: int = 5,
        compression_level: int = 9,
        shuffle: int = 0,
        overwrite: bool = False,
        parse_year_first: bool = False,
        ) -> None:
        """
        Initialize the class and automatically call the to_zarr function.

        Time Range Initialisatoin:
            1. (Build user-defined time ranges):
                - Provide a valid time range using 'start_date', 'end_date', 'time_freq_minutes'
                - 'data_path' must be None
            2. (Initialise time range respecting raw data frequency found in 'data_path'):
                - Provide valid 'data_path' and 'pattern'. 
                - Provide 'end_date' and 'time_freq_minutes'.

        CRS attrs Initialisation:
            1. Initialise crs attrs from reference tif file:
                'crs_wkt' & 'proj4' must be None
            2. Force overwriting with user provided crs attrs:
                Provide 'crs_wkt' & 'proj4'           

        Args:
            save_path: (mandatory)  s(str) output zarr storage file to path.
            reference_tif_file: (mandatory)  (str) example .tif file from where to extract crs attrs.
            grid_coords: (mandatory)  (str) path to .geojson grid coords.
            data_path: (str) path to .tif files.
            chunk_size: (int) amount of timesteps to be unified in a single time-wise chunk.
            pattern: (str) sting pattern to match tif file.
            crs_wkt: (str) custom projection info. If not None, 
                force overwriting out crs attribute (default: None).
            proj4: (str) custom projection info. If not None, 
                force overwriting out crs attribute (default: None).
            start_date: (str) 
            end_date: (str)
            time_freq_minutes: (int)
            compression_level: (int) Blosc clevel (default: 9)
            shuffle: (int) Blosc shuffle (default: 0)
            overwrite: (bool) force to overwrite output zarr.
            parse_year_first: (bool) if reading from data_path check if 
                filename starts with YYYY-MM-DD vs DD-MM-YYYY
        """
        self.save_path = save_path
        
        logger.info("Loading reference .tif file.")
        self.reference_tif_file = reference_tif_file
        self.reference_tif_ds = xr.open_dataset(
            self.reference_tif_file,
            engine="rasterio",
            )

        logger.info("Loading reference grid coords file.")
        self.grid_coords = grid_coords
        self.grid = gpd.read_file(
            self.grid_coords
        )

        self.data_path = data_path
        self.chunk_size = chunk_size
        self.pattern = pattern
        self.crs_wkt = crs_wkt            
        self.proj4 = proj4
        
        self.start_date = start_date
        self.end_date = end_date
        self.time_freq_minutes = time_freq_minutes
        self.overwrite = overwrite
        self.parse_year_first = parse_year_first

        self.compressor = Codec(
            cname="zstd",
            clevel=compression_level,
            shuffle=shuffle,
            )        

        # External attributes (Radar Only)
        self.radar_attributes = {
            "grid_mapping" : "crs",
            "long_name" : "Total precipitation rate", 
            "standard_name" : "tprate",
            "units" : "kg m-2 s-1",
        }
        
        # Global attributes 
        curr_date = datetime\
            .now(timezone.utc)\
            .astimezone()\
            .isoformat(timespec="seconds")

        self.global_attributes = {
            "Author": "Gabriele Franch",
            "Conventions": "",
            "history": f"Created at {curr_date}",
            "Copyright": "Dipartimento Protezione Civile Nazionale rete RADAR",
            "Licence": "CC BY-SA 4.0",
            "Processed by": "Fondazione Bruno Kessler",
            "title": "Italian Radar DPC SRI Archive",
            "zarr_creation": f"Created on {curr_date} by Gabriele Franch (franch@fbk.eu)",
            "zarr_dataset_version": f"0.1.0",
        }
    
        # Initialise the Store object
        self.store = LocalStore(self.save_path)

        # Initialise empty zarr
        self.get_metadata()
        self.generate_timestep_array()
        self.init_zarr()
        
   
    def parse_timestamp(
        self,
        data_path: str,
        ) -> np.datetime64:
        """
        (Hardcoded to current use case).
        Utils function to parse tif filenames into valid datetime object
        
        if self.parse_year_first:
            expected filename format: 'YYYY-MM-DD-hh-mm_ROMA_SRI.tif'
        otherwise:
            expected filename format: 'DD-MM-YYYY-hh-mm.tif'

        Args:
            data_path: (str) path to .tif file
                Example input name: 'YYYY-MM-DD-hh-MM_ROMA_SRI.tif'
                Corresponding timesteps: 'YYYY-MM-DD-hh-MM' 
        Return:
            np.datetime64 (ns)
        """
        if self.parse_year_first:
            basename = os.path.basename(
                data_path.strip(".tif")
                )\
                .split("_")[0]\
                .split("-")
            year, month, day, hours, minutes = basename
        else:
            basename = os.path.basename(
                data_path.strip(".tif")
                ).split("-")
            day, month, year, hours, minutes = basename

        return np.datetime64(
                f"{year}-{month}-{day} {hours}:{minutes}",
                ).astype("datetime64[ns]")        

    def get_metadata(self) -> None:
        """Initialise complete list of tif files.
        """
        logger.info("Init Metadata.")

        # Check if data_path has been provided
        if self.data_path is None:
            self.metadata = []
        else:
            search_path = os.path.join(self.data_path, self.pattern)
            self.metadata = sorted(glob.glob(search_path))

            # If provided and not empty, save metadata to csv
            if len(self.metadata) > 0 :
                pd.DataFrame(self.metadata, columns=["path"])\
                    .to_csv(
                        os.path.join(self.save_path, "metadata.csv"),
                        index=False
                        )
        logger.info(
            f"Found: {len(self.metadata)} files."
        )

    def generate_timestep_array(self) -> None:
        """Generate the full date array including up to end_date
        by reading all timesteps belonging to the collected metadata.
        """
        logger.info("Generating datarray")
        
        # Case in which we generate the full timestep array a priori
        # using start and end date.
        if (self.data_path is None) and\
            (self.start_date is not None) and\
            (self.end_date is not None):

            # Check for date inconsistencies
            assert \
                (np.datetime64(self.start_date) < np.datetime64(self.end_date)),\
                "Start/End date mismatch."
            assert \
                (self.time_freq_minutes is not None),\
                "Time freq param must be provided."

            logger.info("Generating empty timestep array from start/end date")
            self.full_datetime_array = np.arange(
                np.datetime64(self.start_date),
                np.datetime64(self.end_date),
                np.timedelta64(self.time_freq_minutes, "m"),
                )\
                .astype("datetime64[ns]")

        # Case in which we sample timestep frequency from files.
        elif (self.data_path is not None) and\
            len(self.metadata) > 0:

            logger.info("Generating timestep array reading from metadata")

            # Check for date inconsistencies
            assert \
                (self.time_freq_minutes is not None),\
                "Time freq param must be provided."

            # Collect all previously parsed timesteps
            date_array = np.array([
                self.parse_timestamp(filename)\
                for filename in self.metadata
                ])

            # Additional up to self.end_date by self.time_freq_minutes
            if self.end_date:
                
                future_dates = np.arange(
                    date_array[-1] + np.timedelta64(self.time_freq_minutes, "m"),
                    np.datetime64(self.end_date),
                    np.timedelta64(self.time_freq_minutes, "m"),
                    )

                date_array = np.concat([
                    date_array,
                    future_dates,
                ])

            self.full_datetime_array = \
                date_array\
                .astype("datetime64[ns]")

        # Handle empty file list
        elif (self.data_path is not None) and\
            len(self.metadata) == 0:
                raise FileNotFoundError(
                    f"No timesteps found at {os.path.join(self.data_path, self.pattern)}"
                    ) 


    def get_additional_attrs(self) -> Dict[str, Any]:
        """ Read and preprocess additional attributes with correct projection info
        Args:
            ds: (xr.Dataset) of a reference tif file
        Return:
            Dict mapping crs attribute with the reference tif values.
        """
        # Collect attrs by reading from the reference tif file
        # and merge all attrs into a unified structure
        general_attrs = self.reference_tif_ds\
            .spatial_ref\
            .attrs
        crs_attrs = iter(self.reference_tif_ds._attr_sources)\
            .__next__()["band_data"]\
            .attrs
        crs_attrs.update(general_attrs)

        # Remove keys
        [crs_attrs.pop(k) \
            for k in crs_attrs.copy().keys()\
            if k.startswith("what")
        ]

        if not self.proj4 is None:
            logger.info("Overwriting 'proj4'")
            crs_attrs["proj4"] = self.proj4
        else:
            crs_attrs["proj4"] = crs_attrs.pop("where_projdef")

        # Force overwriting if params are provided        
        if not self.crs_wkt is None:
            logger.info("Overwriting ('crs_wkt', 'spatial_ref')")
            crs_attrs["crs_wkt"] = self.crs_wkt
            crs_attrs["spatial_ref"] = self.crs_wkt

        return crs_attrs


    def init_zarr(self):
        """Initialise zarr hierarchy.
        """
        logger.info(f"Initialise empty Zarr Storage.")

        # Generate lat, lon arrays
        n_y      = self.grid["y"].unique().shape[0]
        n_x      = self.grid["x"].unique().shape[0]
        n_time   = self.full_datetime_array.shape[0]
        out_lat  = self.grid["geometry"].y.values.reshape(n_y, n_x)
        out_lon  = self.grid["geometry"].x.values.reshape(n_y, n_x)

        # Init lat, lon related info
        lat_lon = [
            {
                "name": "lat",
                "long_name": "Latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "array": out_lat,
            },
            {
                "name": "lon",
                "long_name": "Longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "array": out_lon,
            }
        ]

        # Init Root Group
        self.root = zarr.group(
            store=self.store,
            zarr_version=2,
            overwrite=self.overwrite,
        )

        # Create coords (x, y)
        for coord in ("x", "y"):
            self.root.create_array(
                name=coord,
                data=self.reference_tif_ds[coord].values,
                attributes={
                    "_ARRAY_DIMENSIONS": [coord],
                    "units": "m",
                }
            )
        
        # Create lat/lon
        for coord in lat_lon:
            self.root.create_array(
                name=coord.pop("name"), 
                data=coord.pop("array"), 
                chunks=(n_y, n_x),
                attributes={
                    "_ARRAY_DIMENSIONS": ["y", "x"],
                    "grid_mapping": "crs",
                    **coord,
                    }
                )
            
        # Create empty timesteps dimension to be filled later on
        # Integer numbers will allow the zarr group to store the
        # default 1970-01-01 fill date (not NaT).
        # This does not solve the issue though 
        self.root.create_array(
            name="time", 
            data=self.full_datetime_array,
            chunks=(n_time,),
            compressor=self.compressor,
            attributes={
                "_ARRAY_DIMENSIONS": ["time"],
                "long_name": "Time",
                "standard_name": "time",
                }
            )

        # Create empty Radar dataset
        self.root.create_array(
            name="RR",
            shape=(n_time, n_y, n_x),
            compressor=self.compressor,
            dtype="float32",
            chunks=(self.chunk_size, n_y, n_x),
            fill_value=np.nan,
            config={"write_empty_chunks": False},
            attributes={
                "_ARRAY_DIMENSIONS": ["time", "y", "x"],
                **self.radar_attributes,
            }
            )
  
        # DataArray of CRS projection information
        self.root.create_array(
            name="crs",
            data=np.array(np.nan, dtype=np.float32),
            attributes={
                "_ARRAY_DIMENSIONS": list(),
                **self.get_additional_attrs(),
                }
            )

        # Add meta info about the zarr dataset creation (global attrs) and consolidate
        self.root.attrs.update(self.global_attributes)
        logger.info("Consolidating metadata.")
        zarr.consolidate_metadata(self.store, zarr_format=2)
        logger.info(f"Zarr stored in: {self.store.__str__()}")


def main(
    save_path: str,
    reference_tif_file: str = "data/2023-01-01-00-00_ROMA_SRI.tif",
    grid_coords: str = "data/reprojected_radar_dpc_grid_yx.geojson",
    data_path: str = None,
    chunk_size: int = 1,
    pattern: str = "**/*.tif",
    crs_wkt: str = None,
    proj4: str = None,
    start_date: str = "2010-01-01 00:00",
    end_date: str = "2050-01-01 00:00",
    time_freq_minutes: int = 5,
    compression_level: int = 9,
    shuffle: int = 0,
    overwrite: bool = False,
    parse_year_first: bool = False,
    ) -> None:
    
    # For reference purpose
    reference_crs_wkt = \
        "PROJCRS[\"unknown\",BASEGEOGCRS[\"unknown\",DATUM[\"Unknown based on WGS 84 ellipsoid\",ELLIPSOID[\"WGS 84\",6378137,298.257223563,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",7030]]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433,ID[\"EPSG\",9122]]]],CONVERSION[\"Transverse Mercator\",METHOD[\"Transverse Mercator\",ID[\"EPSG\",9807]],PARAMETER[\"Latitude of natural origin\",42,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8801]],PARAMETER[\"Longitude of natural origin\",12.5,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8802]],PARAMETER[\"Scale factor at natural origin\",1,SCALEUNIT[\"unity\",1],ID[\"EPSG\",8805]],PARAMETER[\"False easting\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8806]],PARAMETER[\"False northing\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8807]]],CS[Cartesian,2],AXIS[\"easting\",east,ORDER[1],LENGTHUNIT[\"metre\",1,ID[\"EPSG\",9001]]],AXIS[\"northing\",north,ORDER[2],LENGTHUNIT[\"metre\",1,ID[\"EPSG\",9001]]]]"

    # For reference purpose
    reference_proj4 = \
        "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs"
    
    InitZarr(
        save_path,
        reference_tif_file,
        grid_coords,
        data_path,
        chunk_size,
        pattern,
        crs_wkt,
        proj4,
        start_date,
        end_date,
        time_freq_minutes,
        compression_level,
        shuffle,
        overwrite,
        parse_year_first,
    )


if __name__=="__main__":
    Fire(main)
