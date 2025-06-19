# UQ4ML_WaterTemp

## Cold Stunning Models
![ColdStunNet Overview](images/Fig5OG_Sub12Legend.png)

## Publications

[Your publication citation will go here]

Citation:
```
@article{yourname2024coldstunnet,
    title={ColdStunNet: Deep learning approach for predicting cold stunning events},
    author={Your Name and Collaborators},
    journal={Journal Name},
    pages={page_numbers},
    year={2024},
    publisher={Publisher}
}
```



## Data Sources

**Environmental Data:**
- Sea surface temperature (SST) 
- Air temperature from weather stations
- Water temperature

**Time horizon (hours/days):** How far ahead is the prediction? Options: 6H, 12H, 24H, 48H, 72H, 96H, 120H (Operational)

**Temperature threshold (°C):** Critical temperature threshold for cold stunning risk for sea turtles 8°C, for fisheries 4.5°C,

**Data Availability:**
- [Laguna Madre Water Air Temp Data Cleaner](https://github.com/conrad-blucher-institute/LagunaMadreWaterAirTempDataCleaner)
- [NOAA Sea Surface Temperature](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/)
- [Weather station data](https://www.ncei.noaa.gov/data/)
- [tccoon?](placeholder_url)


## Installation (Windows 10)
we provide a .yaml
then the steps


### Conda Environment Setup
```bash
# Create environment
conda env create -f environment.yml

conda env install tensorflow=2.15

# Activate environment
conda activate coldstun

# Install pip specific packages
pip install lib1 lib2 lib3 lib4 etc
```


## Download & Format Data

You can either:

1. Get data from original sources
2. Use data [Laguna Madre Water Air Temp Data Cleaner](https://github.com/conrad-blucher-institute/LagunaMadreWaterAirTempDataCleaner)
3. use prebuilt datasets, file location is data\June_May_Datasets (to replicate published results)

In this README, we refer to your dataset directory as `$DATASETS`
Change the location of data to whereever you are stowing the data you want to use

## Quickstart

Activate your environment:
```bash
conda activate coldstun
```



**Train MSE model from scratch**
```bash
python src/train.py 
```

**Train CRPS model from scratch**
```bash
python src/train.py 
```

**Train PNN off a config file in models/, use existing config or make your own**
**Train PNN model**
```bash
python src/driver/pnn_driver.py @models/PNN_12.txt
python src/driver/pnn_driver.py @models/PNN_48.txt
python src/driver/pnn_driver.py @models/PNN_96.txt

python src/driver/mse_driver.py @models/MSE_96.txt

python src/driver/crps_driver.py @models/CRPS_96.txt
```


**Make predictions with pre-trained model**

The `trained_model` subdirectory includes outputs from MSE, CRPS, & PNN.

```bash
# Generate predictions
# idea below, not actually implemented
python src/driver/cmd_ai_builder.py \
    --flag1 False \
    --flag2 False \
    --flag3 False \
    --flag4 False
```


## Data Format

Example dataset structure:
```
$DATASETS/
├── 24HOURS/
│   ├── INPUT/
│   │   ├── SST_CUBE_2022_24H.npz
│   │   ├── WEATHER_CUBE_2022_24H.npz
│   │   ├── BATHYMETRY_CUBE_2022.npz
│   │   └── TIDAL_CUBE_2022_24H.npz
│   ├── TARGET/
│   │   ├── cold_stunning_events_2022_24H.csv
│   │   └── species_locations_2022.csv
│   └── METADATA/
│       ├── station_locations.csv
│       └── data_quality_flags.csv
```


## Contributing
[Add contribution guidelines]

## License
[Add license information]

## Acknowledgments
[Add acknowledgments for data sources, funding, collaborators]

## Contact
[Your contact information]