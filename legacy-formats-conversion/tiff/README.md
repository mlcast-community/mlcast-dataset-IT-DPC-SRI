# DPC TIFF to GeoTIFF

Converts raw radar TIFFs from the DPC FTP (2025 onwards) to the GeoTIFF format used to build the mlcast zarr datasets.

The source TIFFs are Float32, LZW-compressed, strip-based, with `-9999` nodata. The output GeoTIFFs are Float16, ZSTD-compressed, tiled (256x256), with `-1` nodata and ODIM_H5/V2_1 metadata.

### Usage

```bash
uv sync
uv run python dpc-tiff_to_geotiff.py --input_dir /disks/fast/SRI --output_dir /disks/fast/tiff
```

By default already-converted files are skipped (`--skip_existing=True`). Pass `--noskip_existing` to force re-conversion.
