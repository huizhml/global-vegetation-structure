# Global Vegetation Structure Modeling - Data Downloading

![Project Status](https://img.shields.io/badge/status-active-success.svg)
<!-- ![License](https://img.shields.io/badge/license-MIT-blue.svg) -->

## 📖 Overview

This project aims to facilitate the downloading and processing of data for global vegetation structure modeling. It includes scripts for downloading data from GEDI and Sentinel-2.


## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

- earthengine-api
- planetary-computer
- PySTAC
- stackstac
- xarray
- dask
- geopandas


## 📚 Code Structure

The codebase is organized as follows:

- `run.py`: This is the main script that coordinates the downloading and processing of data.
- `download/`: This directory contains scripts for downloading data from various sources.
    - `gedi.py`: This script downloads data from Source 1.
    - `s2.py`: This script downloads data from Source 2.
- `analyze/`: This directory contains scripts for performing basic analyses on the processed data.
    - `analyze.py`: This script performs the analyses and generates output files.

Each script is designed to be run independently, but `run.py` can be used to run the entire pipeline from start to finish.


## 🏃‍♀️ Running the Code

To download GEDI data
```bash
python -m download.gedi task=download 
```

1. Open a terminal in the root directory of the project.
2. Run the following command:

```bash
python analyze.py