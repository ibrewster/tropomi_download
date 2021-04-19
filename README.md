# README

download_tropomi.py is a script to handle downloading TROPOMI data files from s5phub.copernicus.eu, and generating VolcView images for any VolcView sectors specified in the config.py file.

# INSTALL

This sofware was written and tested using python 3.8 Slightly newer/older versions of python3 may work as well, however they are not tested.

It is recommended that you install this software using a venv or the like to avoid potential module version interferance with other python software.

All required python modules can be installed via pip using the requirments.txt file:

```
pip install -r requirements.txt
```

You will need to copy the `config-dist.py` file to `config.py` and edit the relevant entries prior to running the script. Otherwise, you will get a `ModuleNotFoundError: No module named 'config'` error.

# USAGE

To use the script, simply run `python download_tropomi.py`. This will download any new TROPOMI data files, generate VolcView images for any VolcView sectors covered by the data file, and upload said images to any defined VolcView servers.

# NOTES

- The sector definitions for the download and the sector definitions for VolcView DO NOT need to be the same. Often you can specify a larger area for download that covers multiple VolcView sectors. Only VolcView sectors for which there is data in any given data file will have images generated.
- The download process is designed to be robust. If the connection drops, it will attempt to resume the download from where it left off. If a file is corrupted on download, or in the case of a total download failure, it will download the file again at the next run.
- Downloaded files are checked to ensure that they actually contain data within one or more of the download sectors. Often files will be included in the download that only have NaN values within the area of interest. Such files are discarded and a placeholder (.skipped file) created to prevent future download attempts of said file.
- The download script keeps track of the last *succesfully completed* run. The next time the script it run it will look for any data files dated back to the last succesfull run, or 48 before runtime, _whichever is **older**_. This means that if a new file is added *after* the last run it will be downloaded even if it is timestamped up to 48 hours *before* the last run (unless the last run was more than 48 hours ago).