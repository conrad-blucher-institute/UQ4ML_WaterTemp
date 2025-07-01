# UQ4ML_WaterTemp
Uncertainty in reference to machine learning (ML) systems are notable across a wide span of scientific domains due to their influence on critical decision-making processes. But further attention is needed when using ML-uncertainty quantification (UQ) approaches for rare and often impactful environmental events.  

This repository contains several sets of code for implementing, evaluating, and visualizing three ML-UQ multi-model ensemble (MME) approaches using various loss functions (_mean squared error [MSE], negative log-likelihood [NLL, i.e., PNN], continuous ranked probability score [CRPS]_) discussed in its companion paper cited below. This was done to understand which methods best estimate uncertainty of predictive water temperature information at various lead times (12-, 48-, 96-hr) for improved cold-stunning event advisory in south Texas (or when water temperatures reach threatening levels for wildlife). A 10-fold cross-validation framework was used to compare deterministic and probabilistic skill among all UQ methods during cold seasons and sub-12°C cases. Two case studies representing the most recent event and most impactful event were also used to assess model performance on unseen data, under rare, high-impact cold conditions. 

## Cold-Stunning Models Time Series
![ColdStunNet Overview](images/image10.png)

## Publication
White, M. C., et al. (to be submitted) "Machine Learning Uncertainty Quantifications for Extreme Cold Events." _Artificial Intelligence for the Earth Systems._

## Data Sources

**Data Availability:**
We use water and air temperature observation measurements provided by Texas Coastal Ocean Observation Network (TCOON; https://tidesandcurrents.noaa.gov/tcoon.html), publicly available within the NOAA Tides and Current website (https://tidesandcurrents.noaa.gov/map/index.shtml?region=Texas). This data was then cleaned using an imputation framework described in White et al. (2024), publicly available in the Github repository (https://github.com/conrad-blucher-institute/LagunaMadreWaterAirTempDataCleaner).

**Environmental Data:**
- Water temperature observations
- Air temperature observations
- Air temperature predictions (_perfect prognosis_)

**Leadtime (hours):** 12-hr, 48-hr, 96-hr

**Temperature threshold (°C):** Critical water temperature threshold for cold-stunning event risk for sea turtles: 8°C (Shaver et al., 2017); for fisheries: 4.5°C (Texas Parks and Wildlife Department, 2021)


# Setting up the environment
**Simple Fix**
This repository was made with specific versions of libraries. Most notably Tensorflow version 2.15.0
If you have the exact libraries specified in the file src/setup/UQ4ML_2025.yaml everything in this repository will work. 


**default values, change to your preference**
independent_year variable determines if the models train normally or if the users wishes to test on independent testing years, Set this to be '2021' or '2024', or for Regular testing on rolling origin rotation structure set to "cycle".
independent_year = "cycle"
cycle_list = [0,1,2,3,4,5,6,7,8,9] 
start_iteration = 1
end_iteration = 100

**Train MSE model from scratch**
go into the python script and change this variable
model_name = "MSE"
```bash
python -m src.driver.operational_mse_crps_driver 
```

**Train CRPS model from scratch**
go into the python script and change this variable
model_name = "CRPS" 
```bash
python -m src.driver.operational_mse_crps_driver 
```

**Train PNN off a config file in models/, use existing config or make your own**
```bash
python -m src.driver.pnn_mme_driver @configs/pnn_12h.txt
python -m src.driver.pnn_mme_driver @configs/pnn_48h.txt
python -m src.driver.pnn_mme_driver @configs/pnn_96h.txt

python -m src.helper.pnn_to_csv @configs/pnn_12h.txt
python -m src.helper.pnn_to_csv @configs/pnn_48h.txt
python -m src.helper.pnn_to_csv @configs/pnn_96h.txt
```

**How to Evaluate and Visualize the Results**
inside visualization_driver there is comments explaining how to change the variables to fit what you need.
```bash
python -m src/driver/visualization_driver.py
```

**After training models, follow the instructions and run visualization_driver.py while changing the relevant field. Evaluation and visualization are reliant on one another. Please follow the instructions carefully in the comments within the visualization_driver file.
If you wish to make changes to the visuals, calculations, or other parts of this process please note you will have to refactor the code in their correspondng reference files.s**

**Note:** This file assumes that a main folder called "results" exists in the src directory that contains the corresponding model runs.
```
This is an example of what the results directory should look like, and where it should be located within the repository directory.
$src/
└── results/
    ├── mse_Results/
    │   ├── 12h/
    │   │   └──  mse-1_layers-leaky_relu-64_neurons-cycle_1-iteration_1/
    │   │        ├── val_datetime_obsv_predictions.csv
    │   │        ├── train_datetime_obsv_predictions.csv
    │   │        └── test_datetime_obsv_predictions.csv
    │   ├── 48h/...
    │   └── 96h/...
    ├── pnn_Results/...
    └── crps_Results/...
```

## Contact
Miranda White: [MWhite20@islander.tamucc.edu](mailto:mwhite20@islander.tamucc.edu) <br>
Dr. Philippe Tissot: [Philippe.Tissot@tamucc.edu](mailto:Philippe.Tissot@tamucc.edu)<br>
Son Nguyen: [Son.Nguyen@tamucc.edu](mailto:Son.Nguyen@tamucc.edu) <br>
Hector Marrero-Colominas: [Hector.MarreroColominas@tamucc.edu](mailto:Hector.MarreroColominas@tamucc.edu) <br>
Jarett Woodall: [jarett.woodall@tamucc.edu](mailto:jarett.woodall@tamucc.edu) <br>
