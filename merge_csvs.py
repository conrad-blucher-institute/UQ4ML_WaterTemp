import pandas as pd
import glob 

csv_files = glob.glob("results/esb_tuner_scaled/*progress*.csv")

for file in csv_files:
    df = pd.read_csv(file)
    df.to_csv("results/esb_tuner_scaled/merged_results.csv", mode='a', index=False, header=not pd.io.common.file_exists("results/esb_tuner_scaled/merged_results.csv"))
