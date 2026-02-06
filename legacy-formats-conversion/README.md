# IT-DPC-SRI Legacy format conversion
Here you can find some scripts that kept us busy for a LOONG time trying to convert the older part of the Italian radar composite into a usable format:
- From `OPERA BUFR` to `Geotiff`/`NetCDF` (period from 2010 - mid 2020): decbufr.py is the main script, while decbufr.tgz contains a pre-compiled binary utility from OPERA to decode BUFR that is invoked by Python to perform the conversion. variables.py defines the radar variable types used during decoding. Part of the code has been borrowed from the clever people at ARPAE SIMC that have written [simcradarlib](https://github.com/ARPA-SIMC/simcradarlib).
- From `HDF` to `Geotiff` (period from mid 2020 - 2024): hdf_to_tiff.py handles this conversion with multiprocessing support. After dealing with BUFR, converting from HDF feels like a walk in the park.

### HIC SVNT LEONES
These scripts were written to be used exactly once and have served their purpose. Comments and docstrings are in Italian.

The pre-compiled `decbufr` binary included in decbufr.tgz was built from a version of the [OPERA BUFR library](https://www.eumetnet.eu) by EUMETNET (a copy of what may be the same library can be found at [baltrad/bbufr](https://github.com/baltrad/bbufr)). The compilation was done on Ubuntu 24.04, so the binary may not work on other systems. Due to this platform dependency and the one-off nature of these setup, no Python environment file (pyproject.toml, requirements.txt) is provided.
