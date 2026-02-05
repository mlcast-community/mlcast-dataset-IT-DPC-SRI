# IT-DPC-SRI Legacy format conversion
Here you can find some scripts that kept us busy for a LOONG time trying to convert the older part of the Italian radar composite into a usable format:
- From `OPERA BUFR` to `Geotiff` (period from 2010 - mid 2020): (decbufr.py is the main script, while decbufr.tgz contains a pre-compiled binary utility from OPERA to decode bufr that is invoked by python to perform the conversion). Part of the code has been borrowed from the clever people at ARPAE SIMC that have written simcradarlib https://github.com/ARPA-SIMC/simcradarlib
- From `HDF` to `Geotiff` (period from mid 2020 - 2024): (hdf_to_tiff.py) after dealing with BUFR, converting from HDF feels like a walk in the park. 

### HIC SVNT LEONES
Countless hours have been spent putting together these scripts, to use them EXACTLY ONCE and not look at them again. I advise the reader to do the same. Also, comments and docstrings are in Italian.
