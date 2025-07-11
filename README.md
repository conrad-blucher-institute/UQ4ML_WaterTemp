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
**Quick Start**
There are a few ways you can set up the environment to get this repo working. Here's one approach that worked for us—your setup might differ depending on your hardware and preferences.

This repository was made using Windows 11 
We used specific versions of certain Python libraries. Most notably Tensorflow version 2.15.0
If you have the exact libraries specified in the file src/setup/UQ4ML_2025.yml, everything in this repository will work. 

1. Install Anaconda Distribution GUI:
 - Download from:
    * https://repo.anaconda.com/archive/Anaconda3-2025.06-0-Windows-x86_64.exe
Follow the installation wizard using all default options.
No need to make an account.

2. Create the Environment:

 - Open Anaconda Navigator
 - Go to the Environments tab (left side)
    * Might need to wait for it to load
 - Click Import at the bottom
 - Use the file browser to navigate to src/setup/UQ4ML_2025.yml in the repo
    * You may have to extract the contents of the zip folder that has the repo
 - Click Import

3. Activate the Environment:
After the environment finishes installing, press the green play button next to environment name, and select 'open terminal'

 - If succeded near your shell prompt will be:
```bash
(UQ4ML_2025)
``` 

Now you're ready to follow the rest of the guide! Dont forgot to change the directory into the folder containing the repo you downloaded


# Preparing Models for Training
**Variable descriptions for training (MSE and CRPS):**
* independent_year: 
    * Variable that controls model testing mode:
        * Set to **"2021"** or **"2024"** to have the models test on the independent hold-out years.
        * Set to **"cycle"** for rolling origin cross-validation (Default).
     
* cycle_list:
    * Variable that controls the rotations (cycles) included in training:
        * Set to **[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]** to test on rolling origin cross-validation (Default).
     
* start_iteration:
    * Variable that controls where the number of trials start:
        * Set to **1** to have the trainer begin at number 1 (Default).
        
* end_iteration:
    * Variable that controls where the trials end:
        * Set to **100** to have the trainer stop running trials once it reaches 100 (Default).
        * **Note:** The default value for this means that the trainer will run 100 times.
        * To run fewer then this, change this variable to a number that will yield you the number of trials you would like.
     
* model_name:
    * Variable that controls whether CRPS or MSE is trained.
        * Set to **MSE** to run the trainer to train MSE.
        * Set to **CRPS** to run the trainer to train CRPS. 


**Default values, change to your preference:** 
* independent_year = "cycle"
* cycle_list = [0,1,2,3,4,5,6,7,8,9] 
* start_iteration = 1 
* end_iteration = 100 

**Variable descriptions for training (PNN):**

To change the variables for the PNN model, you will have to navigate to the configs folder in the repository.
Within the folder there are currently three files one for 12-hr, 48-hr, and 96-hr.
* repetitions:
    * Variable that controls the number of times the model runs:
        * Set to **100** to have 100 runs execute (Default).

* cycle:
    * Variable that determines the cycles that the model will train on:
        * Numbers 0 - 9 are placed on new lines below **--cycle** within the config file (Default).
        
* leadtime:
     * Variable that controls what the leadtime the model will predict:
        * Set to either 12, 48, 96, or a number a user wishes to test on. 

**Train MSE model from scratch**
To train the MSE model, go into **operational_mse_crps_driver** and change the **model_name** variable to equal **"MSE"** <br>
model_name = "MSE"
```bash
python -m src.driver.operational_mse_crps_driver 
```

**Train CRPS model from scratch**
To train the CRPS model, go into **operational_mse_crps_driver** and change the **model_name** variable to equal **"CRPS"**<br>
model_name = "CRPS" 
```bash
python -m src.driver.operational_mse_crps_driver 
```

**Train PNN off a config file in models/, you can use existing config or make your own**

```bash
python -m src.driver.pnn_mme_driver @configs/pnn_12h.txt
python -m src.driver.pnn_mme_driver @configs/pnn_48h.txt
python -m src.driver.pnn_mme_driver @configs/pnn_96h.txt
```
**This line takes the output of pnn_mme_driver and reformats it into something the visualization_driver can use:**
```bash
python -m src.helper.pnn_to_csv @configs/pnn_12h.txt
python -m src.helper.pnn_to_csv @configs/pnn_48h.txt
python -m src.helper.pnn_to_csv @configs/pnn_96h.txt
```

**If you want to run the independent testing years as well, both 2021 and 2024, you run the command like so:**
```bash
python -m src.helper.pnn_to_csv @configs/pnn_12h.txt -I
python -m src.helper.pnn_to_csv @configs/pnn_48h.txt -I
python -m src.helper.pnn_to_csv @configs/pnn_96h.txt -I
```


# Model Evaluation and Visualization Creation
**How to Evaluate and Visualize the Results**<br>
Inside the visualization_driver.py file there are comments explaining how to change the variables to fit what you need.
```bash
python -m src.driver.visualization_driver
```
**After training models, follow the instructions and run visualization_driver.py while changing the relevant field. Evaluation and visualization are reliant on one another. Please follow the instructions carefully in the comments within the visualization_driver file.
If you wish to make changes to the visuals, calculations, or other parts of this process please note you will have to refactor the code in their correspondng reference files.**

**Note:** The visualization_driver.py assumes that a main folder called "results" exists in the src directory that contains the corresponding model runs. In other words, evaluation and visualization should only be made after the training of models is complete. 
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
* Miranda White: [MWhite20@islander.tamucc.edu](mailto:mwhite20@islander.tamucc.edu) <br>
* Dr. Philippe Tissot: [Philippe.Tissot@tamucc.edu](mailto:Philippe.Tissot@tamucc.edu)<br>
* Son Nguyen: [Son.Nguyen@tamucc.edu](mailto:Son.Nguyen@tamucc.edu) <br>
* Hector Marrero-Colominas: [Hector.MarreroColominas@tamucc.edu](mailto:Hector.MarreroColominas@tamucc.edu) <br>
* Jarett Woodall: [jarett.woodall@tamucc.edu](mailto:jarett.woodall@tamucc.edu) <br>
