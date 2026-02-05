# mlcast-dataset-IT-DPC-SRI
Code to convert the TIFF files from the Italian radar composite dataset to Zarr format.

Obtain (ha! easier said than done) and put all the tiff of the radar composite in a folder. Then run:

```shell
uv run python zarr_converter.py \
    --data_path  /path/to/geotiff/dir/ \
    --save_path  /save/path/ \
    --group_name italian_radar_dpc_sri.zarr \
    --pattern  20*/**/*.tif \
    --batch_size  1 \
    --num_workers  32 \
    --timerange_split  1000 \
    --compression_level  9 \
    --shuffle  0 \
    --end_date  "2025-12-31 23:55" 
```
a (somewhat) mlcast-compliant zarr should appear.