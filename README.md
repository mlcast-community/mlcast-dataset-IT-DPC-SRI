# mlcast-dataset-IT-DPC-SRI
Code to convert the TIFF files from the Italian radar composite dataset to Zarr format.

Obtain (ha! easier said than done) and put all the tiff of the radar composite in a folder. Then run:

```shell
cd zarr-converter

uv run python zarr_converter.py \
    --data_path=/path/to/geotiff/dir \
    --save_path=/path/to/output.zarr \
    --reference_tif_file=../data/2023-01-01-00-00_ROMA_SRI.tif \
    --grid_coords=../data/reprojected_radar_dpc_grid_yx.geojson \
    --start_date=2010-01-01T00:00 \
    --end_date=2026-01-01T00:00 \
    --num_workers=32
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `data_path` | *(required)* | Directory containing GeoTIFF files |
| `save_path` | *(required)* | Output zarr path |
| `reference_tif_file` | `data/2023-01-01-00-00_ROMA_SRI.tif` | Reference TIF for CRS/coordinate extraction |
| `grid_coords` | `data/reprojected_radar_dpc_grid_yx.geojson` | GeoJSON with the reprojected lat/lon grid |
| `start_date` | `2010-01-01T00:00` | Start of the time range (ISO 8601) |
| `end_date` | `2026-01-01T00:00` | End of the time range (ISO 8601) |
| `pattern` | `**/*.tif` | Glob pattern for finding TIF files |
| `num_workers` | `32` | Number of parallel write workers |
| `batch_size` | `1` | TIFs per write batch |
| `timerange_split` | `1000` | Split write jobs into chunks of this size |
| `compression_level` | `9` | Blosc ZSTD compression level (0-9) |
| `shuffle` | `0` | Blosc shuffle filter (0=none, 1=byte, 2=bit) |

### Base frequencies

The DPC SRI archive changed its temporal resolution over the years. The converter uses these frequency bands to detect missing timestamps across the full time range:

| Period | Frequency |
|---|---|
| 2010-01-01 to 2014-06-25 09:00 | 15 min |
| 2014-06-25 09:00 to 2020-06-30 | 10 min |
| 2020-06-30 onwards | 5 min |

These are defined in `BASE_FREQUENCIES` in `zarr_converter.py` and stored as global metadata in the output zarr.