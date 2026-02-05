import os
import glob
import itertools
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import zarr
from datetime import datetime, timezone
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from tqdm import tqdm
from functools import partial
import time
from osgeo import gdal

from pyproj import CRS

# from zarr import DirectoryStore #(zarr v2)
from zarr.storage import LocalStore # which replace the DirectoryStore #(zarr v3?)

from numcodecs import Blosc as Codec #(zarr v2) (apparently accepted also in v3)
# from zarr.codecs import BloscCodec  as Codec #(zarr v3?)

from pyproj import Transformer
from tqdm import tqdm
from loguru import logger
from fire import Fire


def parse_timestamp(data_path) -> str:
    """
    Example input name: 'YYYY-MM-DD-HH-MM_ROMA_SRI.tif'
    Example input timesteps: 'YYYY-MM-DD-HH-MM' 
    Return np.datetime
    """
    basename = os.path.basename(data_path).split("_")[0]
    year, month, day, hours, minutes = basename.split("-")
    return np.datetime64(f"{year}-{month}-{day} {hours}:{minutes}") 


def get_sigle_data(item, fillvalue=-1.):
    """ Utility function to be passed in the append_to_zarr """
    # If reading tif files, rely on Gdal instead of xarray
    ds = gdal.Open(item, gdal.GA_ReadOnly)
    array = ds.GetRasterBand(1).ReadAsArray()
    array = np.where(array==fillvalue, np.nan, array)
    return  np.expand_dims(array, axis=0)


def get_data(item):
    return np.concat(
        [get_sigle_data(curr_item) for curr_item in item], axis=0
    )


def append_to_zarr(idx, shift_idx, iterable, group, array_name, batch_size) -> None:
    """Specific function to append into zarr group for MP
    Args:
        idx: (int) iteration idx
        shift_idx: (int) how many idx to shift while saving
        iterable: (list) list of complete filepaths
        group: zarr.LocalStore group
        array_name: (str) zarr group array name
        batch_size: (int) amount of items to be loaded in batch
    """
    try:
        # Extract the individual item from the iterable obj
        # It can be either datapath or np.datetime64
        
        # Shift the curret idx [0-len(iterable)] by a shifting factor
        item = iterable[idx : (idx+batch_size)]

        # Return final data to be stored
        data = get_data(item)

        # Store in Zarr group according to the correct shift idx terms
        group[array_name][(idx+shift_idx) : (idx+shift_idx+batch_size), ...] = data
        # return True, item
    
    # To tackle the missing object exception while performing multiprocessing
    except Exception as e:
        print(f"Error while saving {item}: {str(e)}")
        # return False, item


class ToZarr:
    """Convert tif files to a single zarr store."""

    def __init__(
        self,
        data_path: str,
        save_path: str = None,
        group_name: str = None,
        chunk_size: int = 1,
        pattern: str = "**/*.tif",
        grid_coords: str = None,
        num_workers: int = 8,
        batch_size: int = 1, 
        timerange_split: int = None,
        reference_tif_file: str = None,
        compression_level: int = 5,
        shuffle: int = 1, 
        end_date: str = "2050-01-01 00:00",
    ) -> None:
        """
        Initialize the class and automatically call the to_zarr function.
        """
        self.data_path = os.path.abspath(data_path)
        self.save_path = (
            os.path.abspath(save_path) if save_path is not None else None
        )
        self.compressor = Codec(cname="zstd", clevel=compression_level, shuffle=shuffle)
        self.group_name = group_name
        self.chunk_size = chunk_size
        self.pattern = pattern
        self.grid_coords = grid_coords
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.reference_tif_file = reference_tif_file
        self.end_date = end_date

        # Copied from a valid tiff file:
        # Xarray based attrs reding tends to omit some attrs: resulting 
        # in poor reproducible results.
        self.crs_wkt = "PROJCRS[\"unknown\",BASEGEOGCRS[\"unknown\",DATUM[\"Unknown based on WGS 84 ellipsoid\",ELLIPSOID[\"WGS 84\",6378137,298.257223563,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",7030]]],PRIMEM[\"Greenwich\",0,ANGLEUNIT[\"degree\",0.0174532925199433,ID[\"EPSG\",9122]]]],CONVERSION[\"Transverse Mercator\",METHOD[\"Transverse Mercator\",ID[\"EPSG\",9807]],PARAMETER[\"Latitude of natural origin\",42,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8801]],PARAMETER[\"Longitude of natural origin\",12.5,ANGLEUNIT[\"degree\",0.0174532925199433],ID[\"EPSG\",8802]],PARAMETER[\"Scale factor at natural origin\",1,SCALEUNIT[\"unity\",1],ID[\"EPSG\",8805]],PARAMETER[\"False easting\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8806]],PARAMETER[\"False northing\",0,LENGTHUNIT[\"metre\",1],ID[\"EPSG\",8807]]],CS[Cartesian,2],AXIS[\"easting\",east,ORDER[1],LENGTHUNIT[\"metre\",1,ID[\"EPSG\",9001]]],AXIS[\"northing\",north,ORDER[2],LENGTHUNIT[\"metre\",1,ID[\"EPSG\",9001]]]]"

        self.proj4 = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs +type=crs"

        # To be savend in the _CRS attributes
        # self.projjson = CRS.from_proj4(self.proj4).to_json()
        
        # External attributes (Radar Only)
        self.radar_attributes = {
            "grid_mapping" : "crs",
            "long_name" : "Total precipitation rate", 
            "standard_name" : "tprate",
            "units" : "kg m-2 s-1",
        }
        
        # Global attributes 
        curr_date = datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')
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
        
        if save_path is None:
            self.store = LocalStore(
                os.path.join(self.data_path, self.group_name)
            )
        else:
            self.store = LocalStore(
                os.path.join(self.save_path, self.group_name)
            )

        # Steps
        self.get_metadata()
        self.generate_timestep_array()
        self.init_zarr()

        # Zarr v3 datatime64 is not supported yet (while indexing the dataset)
        # but is seems to work fine with the append method (although suboptimal)
        timerange_split = len(self.metadata) if timerange_split is None\
            else timerange_split

        timerange_slices = list(range(0, len(self.metadata), timerange_split)) 

        # Split large iterator on smaller subtask
        for i, idx in enumerate(timerange_slices):
            start_idx = idx
            end_idx   = idx + timerange_split

            filtered_metadata = self.metadata[start_idx : end_idx]

            # Execute it on the sliced metadata
            logger.info(f"Part: {i+1}/{len(timerange_slices)}  Current starting idx: {start_idx}-{end_idx}  Shift idx {start_idx}")
            logger.info(f"Storing from: {filtered_metadata[0]}")
            logger.info(f"To:           {filtered_metadata[-1]}")

            self.to_zarr(
                iterable=filtered_metadata,
                array_name="RR",
                shift_idx=start_idx, 
                )

        logger.info("Consolidate metadata.")
        zarr.consolidate_metadata(self.store, zarr_format=2)
        logger.info(f"Zarr Group stored in: {self.store.__str__()}")
   

    def get_metadata(self) -> None:
        """ Initialise complete list of tif files"""
        logger.info("Init Metadata.")
        search_path = os.path.join(self.data_path, self.pattern)
        self.metadata = glob.glob(search_path)
        self.metadata.sort()
        print(f"Found: {len(self.metadata)} files in {search_path}")

        # Saving Metadata
        pd.DataFrame(self.metadata, columns=["path"])\
            .to_csv(os.path.join(self.save_path, "metadata.csv"), index=False)


    def generate_timestep_array(self):
        """ Generate the full date array including up to end_date"""
        logger.info("Generating datarray")
        # Into np.datetime64
        date_array = np.array([parse_timestamp(data_path) for data_path in self.metadata])

        if len(date_array) == 0:
            raise FileNotFoundError(f"No timesteps found at {os.path.join(self.data_path, self.pattern)}") 

        # Additional up to 2050 by 5 mins
        future_dates = np.arange(
            date_array[-1] + np.timedelta64(5, "m"),
            np.datetime64(self.end_date),
            np.timedelta64(5, "m")
        ).astype("datetime64[ns]")

        self.full_datetime_array = np.concat([date_array, future_dates])\
            .astype("datetime64[ns]")


    def get_additional_attrs(self, ds) -> dict:
        """ Read and preprocess additional attributes with correct projection info """
        
        # Collect attrs from geotif
        general_attrs = ds.spatial_ref.attrs
        proj_attrs = iter(ds._attr_sources).__next__()["band_data"].attrs

        # Merge
        proj_attrs.update(general_attrs)

        # Remove keys
        [proj_attrs.pop(k) for k in proj_attrs.copy().keys() if k.startswith("what")]
        _ = proj_attrs.pop("where_projdef") if "where_projdef" in proj_attrs else None

        # Assign correct projection info
        proj_attrs["proj4"] = self.proj4
        proj_attrs["crs_wkt"] = self.crs_wkt
        proj_attrs["spatial_ref"] = self.crs_wkt

        return proj_attrs


    def init_zarr(self):
        """ Initialise zarr group """
    
        # Read the first dataobject
        # ds = xr.open_dataset(self.metadata[0], engine="rasterio")

        # Read from reference, user-defined, tif file
        ds = xr.open_dataset(self.reference_tif_file)

        # Generate lat, lon arrays
        logger.info("Reading grid coords file ...")
        grid = gpd.read_file(self.grid_coords)
        n_y, n_x = grid["y"].unique().shape[0], grid["x"].unique().shape[0]
        n_time = self.full_datetime_array.shape[0]
        out_lat  = grid["geometry"].y.values.reshape(n_y, n_x)
        out_lon  = grid["geometry"].x.values.reshape(n_y, n_x)

        logger.info(f"Initialise Zarr group.")

        # Init Group
        self.group = zarr.group(store=self.store, zarr_version=2, overwrite=True)

        # Create coords (x, y)
        for coord in ("x", "y"):
            self.group.create_array(
                coord,
                data=ds[coord].values
                )
            self.group[coord].attrs["_ARRAY_DIMENSIONS"] = [coord]
            self.group[coord].attrs["units"] = "m"

        # Create coords (lat, lon)
        lat_lon = [
            {
                "name": "lat",
                "long_name": "Latitude",
                "standard_name": "latitude",
                "units": "degrees_north",
                "array": out_lat
            },
            {
                "name": "lon",
                "long_name": "Longitude",
                "standard_name": "longitude",
                "units": "degrees_east",
                "array": out_lon
            }
        ]
        for coord in lat_lon:
            cname, latlonarray = coord.pop("name"), coord.pop("array")
            self.group.create_array(
                cname, 
                data=latlonarray, 
                chunks=(n_y, n_x)
                )
            self.group[cname].attrs["_ARRAY_DIMENSIONS"] = ["y", "x"]
            self.group[cname].attrs["grid_mapping"] = "crs"
            # self.group[cname].attrs["_CRS"] = {"projjson": self.projjson}
            self.group[cname].attrs.update(coord)

        # Create empty timesteps dimension to be filled later on
        # Integer numbers will allow the zarr group to store the
        # default 1970-01-01 fill date (not NaT). This does not solve the issue though 
        self.group.create_array(
            "time", 
            data=self.full_datetime_array,
            chunks=(len(self.full_datetime_array),),
            )
        self.group["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]
        self.group["time"].attrs["long_name"] = "Time"
        self.group["time"].attrs["standard_name"] = "time"
        
        # Create empty Radar dataset
        self.group.create_array(
            "RR",
            shape=(n_time, n_y, n_x),
            compressor=self.compressor,
            dtype="float32",
            chunks=(self.chunk_size, n_y, n_x),
            fill_value=np.nan,
            config={"write_empty_chunks": False}
            )
        self.group["RR"].attrs["_ARRAY_DIMENSIONS"] = ["time", "y", "x"]
        # self.group["RR"].attrs["_CRS"] = {"projjson": self.projjson}
        self.group["RR"].attrs.update(self.radar_attributes)

        # DataArray of CRS projection information
        self.group.create_array(
            "crs",
            data=np.array(np.nan, dtype=np.float32),
            )
        self.group["crs"].attrs["_ARRAY_DIMENSIONS"] = list()
        self.group["crs"].attrs.update(self.get_additional_attrs(ds))

        # Add meta info about the zarr dataset creation (global attrs)
        self.group.attrs.update(self.global_attributes)


    def to_zarr(self, iterable, array_name, shift_idx) -> None:
        """Recursively read from datapath and store into a single zarr storage"""
        logger.info(f"Now storing files into group var:  ({array_name})")       

        # Generate iterable idxs (starting idxs)
        iterable_idxs = list(range(0, len(iterable), self.batch_size))

        # success_count, failed_count, failed_files = 0, 0, []

        # Initialise a partial function (keeping all except the iterative idx fixed)
        append_partial = partial(
            append_to_zarr,
            shift_idx=shift_idx,
            iterable=iterable, 
            group=self.group, 
            array_name=array_name, 
            batch_size=self.batch_size
        )

        # while imap shows slightly better performance 
        with mp.Pool(self.num_workers) as pool:
            results = list(tqdm(
                pool.imap(append_partial, iterable_idxs),
                total=len(iterable_idxs),
                desc="Storing files", 
                unit="file"
            ))
            pool.close()
            pool.join()


        # # Count results
        # for success, filepath in results:
        #     if success:
        #         success_count += 1
        #     else:
        #         failed_count += 1
        #         failed_files.append(filepath)
        
        # # Logging failed conversions
        # if failed_count: 
        #     logger.info(f"Completed with errors:")
        #     out_failed_path = os.path.join(self.save_path, "failed_conversion_files.csv")
        #     print(f"Failed: {failed_count}/{len(results)} ({(failed_count/len(results))*100:.3f}%)")
        #     print(f"Saving failed files list into: {out_failed_path}")
        #     pd.DataFrame(failed_files, columns="path")\
        #         .to_csv(out_failed_path, index=False)
        # else:
        #     logger.info("Done!")
       
            
        ### Plain single worker single append 
        # for i, data_path in tqdm(enumerate(self.metadata), total=len(self.metadata)): 
        #     band_data = xr.open_dataset(data_path)["band_data"].values

        #     # ts = np.array((parse_timestamp(data_path),))
        #     self.group["RR"][i, :, :] = (band_data.squeeze())
        #     # self.group["time"].append(ts)

        #     if i == 2:
        #         break

        ### Multi file reading, single appending
            




def main(
        data_path: str,
        save_path: str = None,
        group_name: str = None,
        chunk_size: int = 1,
        pattern: str = "**/*.tif",
        grid_coords: str = "./data/reprojected_radar_dpc_grid_yx.geojson",
        num_workers: int = 8,
        batch_size: int = 1,
        timerange_split: int = 1000,
        reference_tif_file: str = "./data/radar_dpc_geotif/2023/01/2023-01-01-00-00_ROMA_SRI.tif",
        compression_level: int = 5,
        shuffle: int = 1, 
        end_date: str = "2050-01-01 00:00",
        ):
    
    t0 = time.time()
    ToZarr(
        data_path,
        save_path,
        group_name,
        chunk_size,
        pattern,
        grid_coords,
        num_workers,
        batch_size,
        timerange_split,
        reference_tif_file,
        compression_level,
        shuffle,
        end_date, 
    )
    t1 = time.time()
    logger.info(f"Time elapsed: {((t1-t0)/60):.3f} min")


if __name__=="__main__":
    Fire(main)

    ### Full run 
    # nohup .venv/bin/python3 italy-dpc/zarr_converter.py --data_path /disks/quattro/mlcast/radar_dpc_geotif/ --save_path /disks/quattro/mlcast/  --group_name italian_radar_dpc_sri.zarr --pattern 20*/**/*.tif --batch_size 1 --num_workers 32 --timerange_split 1000 --compression_level 9 --shuffle 0 --end_date "2050-01-01 00:00" >> /disks/quattro/mlcast/full_conversion_run_logs_clevel9_shuffle0_attempt3_correctingtimestepsupto2050_noduplicates.txt &
