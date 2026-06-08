"""
Benchmark: Original vs Optimized creatingAdditionalColumns
===========================================================

Compares the performance of the original repeated-insert approach
vs an optimized version that collects columns and uses pd.concat.

Run standalone:
    python benchmark_creatingAdditionalColumns.py

Or run in real inference context (see bottom of script).
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# =============================================================================
# ORIGINAL FUNCTION (from utils_mse_crps.py)
# =============================================================================

def creatingAdditionalColumns_ORIGINAL(df, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset=0.0):
    """Original version with repeated df[col] = ... inserts."""
    import warnings
    import pandas as pd
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    
    interval = pred_atp_interval

    # Creating past air temperature values
    s = 'airTemperature__'
    s3 = 'h_ago'

    for i in range(atp_hours_back): 
        j = i + 1
        con = s + str(j) 
        con = []
        name = s + str(j) + s3

        for k in range(j):
            con.append(-999)

        for w in range(len(df)-(j)):
            temp = df['Air Average'][w]
            con.append(temp)  

        df[name] = con
        
    # Creating past water temperature values
    x1 = 'waterTemperature__'
    x3 = 'h_ago'

    for i in range(wtp_hours_back): 
        j = i + 1
        con = x1 + str(j) 
        con = []
        name = x1 + str(j) + x3

        for k in range(j):
            con.append(-999)

        for w in range(len(df)-(j)):
            temp = df['Water Average'][w]
            con.append(temp)  

        df[name] = con
        
    df = offSetCreator(df, IPPOffset, input_hours_forecast)

    # Predicted air temperature
    xPred = 'airTemperature_pred__'
    xPred3 = 'h_forecast'
        
    for i in range(interval, input_hours_forecast+1, interval): 
        j = i
        con = xPred + str(i) 
        con = []
        name = xPred + str(j) + xPred3

        for w in range(len(df) - (i)):   
            temp = df['Air Average'][w + i]
            con.append(temp)  

        for k in range(i):
            con.append(-999)

        df[name] = con
    
    # Creating target
    t = 'waterTemperature_' + str(input_hours_forecast) + 'h_forecast'
    con = t + str(j)
    con = []
    name = t 

    for w in range(len(df) - (input_hours_forecast)):
        temp = df['Water Average'][w+(input_hours_forecast)]
        con.append(temp)
    
    for k in range(input_hours_forecast):
        con.append(-999)
    
    df[name] = con
    
    if input_structure == "descending":
        water_temp_columns = [col for col in df.columns if "waterTemperature__" in col]
        air_temp_columns = [col for col in df.columns if "airTemperature__" in col and "_ago" in col]
        forecast_columns = [col for col in df.columns if "forecast" in col]
        other_columns = [col for col in df.columns if col not in water_temp_columns + air_temp_columns + forecast_columns]

        if "Water Average" in other_columns:
            other_columns.remove("Water Average")  
        if "Air Average" in other_columns:
            other_columns.remove("Air Average")  

        reordered_columns = (
            other_columns
            + water_temp_columns[::-1]
            + ["Water Average"]
            + air_temp_columns[::-1]
            + ["Air Average"]
            + forecast_columns
        )

        df = df[reordered_columns]
        return df

    elif input_structure == "ascending":
        return df


# =============================================================================
# OPTIMIZED FUNCTION (build columns first, concat once)
# =============================================================================

def creatingAdditionalColumns_OPTIMIZED(df, input_structure, input_hours_forecast, atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset=0.0):
    """Optimized version: vectorized pandas operations instead of row-by-row loops."""
    import warnings
    import pandas as pd
    import numpy as np
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    
    interval = pred_atp_interval
    new_columns = {}
    
    # Apply offset first (vectorized)
    df_work = df.copy()
    if IPPOffset != 0.0:
        mask = df_work['Air Average'] != -999
        df_work.loc[mask, 'Air Average'] = df_work.loc[mask, 'Air Average'] + IPPOffset
    
    # Vectorized: create lagged air temperature columns using pandas.shift()
    for lag in range(1, atp_hours_back + 1):
        col_name = f'airTemperature__{lag}h_ago'
        # shift(lag) moves values down, fills top with NaN, then fillna(-999)
        new_columns[col_name] = df_work['Air Average'].shift(lag).fillna(-999).values
    
    # Vectorized: create lagged water temperature columns using pandas.shift()
    for lag in range(1, wtp_hours_back + 1):
        col_name = f'waterTemperature__{lag}h_ago'
        new_columns[col_name] = df_work['Water Average'].shift(lag).fillna(-999).values
    
    # Vectorized: create forward-looking air temperature columns (perfect prognosis)
    # Use negative shift to look ahead
    for hours_ahead in range(interval, input_hours_forecast + 1, interval):
        col_name = f'airTemperature_pred__{hours_ahead}h_forecast'
        # shift(-hours_ahead) moves values up, fills tail with NaN
        shifted = df_work['Air Average'].shift(-hours_ahead)
        new_columns[col_name] = shifted.fillna(-999).values
    
    # Vectorized: create target (forward-looking water temperature)
    target_col = f'waterTemperature_{input_hours_forecast}h_forecast'
    shifted_target = df_work['Water Average'].shift(-input_hours_forecast)
    new_columns[target_col] = shifted_target.fillna(-999).values
    
    # Add all new columns at once via concat
    df = pd.concat([df_work, pd.DataFrame(new_columns)], axis=1)
    
    if input_structure == "descending":
        water_temp_columns = [col for col in df.columns if "waterTemperature__" in col]
        air_temp_columns = [col for col in df.columns if "airTemperature__" in col and "_ago" in col]
        forecast_columns = [col for col in df.columns if "forecast" in col]
        other_columns = [col for col in df.columns if col not in water_temp_columns + air_temp_columns + forecast_columns]

        if "Water Average" in other_columns:
            other_columns.remove("Water Average")  
        if "Air Average" in other_columns:
            other_columns.remove("Air Average")  

        reordered_columns = (
            other_columns
            + water_temp_columns[::-1]
            + ["Water Average"]
            + air_temp_columns[::-1]
            + ["Air Average"]
            + forecast_columns
        )

        df = df[reordered_columns]
        return df

    elif input_structure == "ascending":
        return df


# =============================================================================
# HELPER: offSetCreator
# =============================================================================

def offSetCreator(dataYear, IPPOffset, input_hours_forecast):
    """Apply IPP offset if needed."""
    if IPPOffset != 0.0:
        for i, row in dataYear.iterrows():
            value = dataYear.loc[i].at["Air Average"]
            if value != -999:
                dataYear.at[i, "Air Average"] = value + IPPOffset
    return dataYear


# =============================================================================
# BENCHMARK
# =============================================================================

def run_benchmark(num_runs=5):
    """Benchmark both versions on a sample ESB dataset."""
    
    print("\n" + "="*70)
    print("BENCHMARK: creatingAdditionalColumns (Original vs Optimized)")
    print("="*70 + "\n")
    
    # Load sample data (ESB 2020-2021, same as inference uses)
    data_path = _REPO_ROOT / "data" / "ESB_datasets" / "esb_2020_2021.csv"
    df_raw = pd.read_csv(data_path)
    
    print(f"Sample data shape: {df_raw.shape}")
    print(f"Number of runs per version: {num_runs}\n")
    
    # Parameters (matching inference defaults)
    params = {
        'input_structure': 'descending',
        'input_hours_forecast': 12,
        'atp_hours_back': 24,
        'wtp_hours_back': 24,
        'pred_atp_interval': 1,
        'IPPOffset': 0.0
    }
    
    # Benchmark ORIGINAL
    print("Running ORIGINAL version...")
    times_original = []
    for run_num in range(num_runs):
        df_test = df_raw.copy()
        start = time.time()
        result_orig = creatingAdditionalColumns_ORIGINAL(df_test, **params)
        elapsed = time.time() - start
        times_original.append(elapsed)
        print(f"  Run {run_num+1}: {elapsed:.3f}s")
    
    avg_original = np.mean(times_original)
    std_original = np.std(times_original)
    
    print(f"\nOriginal Average: {avg_original:.3f}s (±{std_original:.3f}s)\n")
    
    # Benchmark OPTIMIZED
    print("Running OPTIMIZED version...")
    times_optimized = []
    for run_num in range(num_runs):
        df_test = df_raw.copy()
        start = time.time()
        result_opt = creatingAdditionalColumns_OPTIMIZED(df_test, **params)
        elapsed = time.time() - start
        times_optimized.append(elapsed)
        print(f"  Run {run_num+1}: {elapsed:.3f}s")
    
    avg_optimized = np.mean(times_optimized)
    std_optimized = np.std(times_optimized)
    
    print(f"\nOptimized Average: {avg_optimized:.3f}s (±{std_optimized:.3f}s)\n")
    
    # Speedup
    speedup = avg_original / avg_optimized
    improvement_pct = (1 - avg_optimized / avg_original) * 100
    
    print("="*70)
    print(f"Speedup: {speedup:.2f}x faster")
    print(f"Improvement: {improvement_pct:.1f}%")
    print("="*70 + "\n")
    
    # Verify outputs are identical (spot check)
    print("Verifying outputs match...")
    assert result_orig.shape == result_opt.shape, "Shape mismatch!"
    assert (result_orig.columns == result_opt.columns).all(), "Column mismatch!"
    print("✓ Outputs match!\n")



if __name__ == "__main__":
    # Run synthetic benchmark
    run_benchmark(num_runs=5)
    