from pathlib import Path

from datetime import datetime

import pandas as pd

from evaluations.cross_validation_visuals_paper import model_selection_conditional

def file_retriever(leadTime, cycle, architecture, obsVsPred, start, end, fileLocation = "TWC_hourly_UQ_Files"):

    # Calles function to retrieve model list based on leadTime and architecture
    model_list = model_selection_conditional(leadTime, architecture)
                
    # Loop over each hyperparameter combo.
    for model in model_list:

        start_dt = datetime.strptime(start, '%m/%d/%Y %H:%M')
        end_dt = datetime.strptime(end, '%m/%d/%Y %H:%M')

        # Format for filename or variable name
        results_path = f"TWC_results_{start_dt.strftime('%Y%m%d_%H%M')}_to_{end_dt.strftime('%Y%m%d_%H%M')}"

        # Utilizes Path for cross compatability regardless of macOs or Windows
        input_path = Path("src") / results_path / fileLocation / f"{obsVsPred}_{leadTime}h_{architecture}_Cycle_{cycle}_Model_{model}.csv"
        df = pd.read_csv(input_path)

        df['date_time'] = pd.to_datetime(df["date_time"])

        return df
# END: def file_retriever()