import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
import warnings
from datetime import datetime


# ---------------------------------------------------------------------------
# Column name configuration — change these if your CSVs use different headers
# ---------------------------------------------------------------------------
AIR_COL = "Air Average"
WATER_COL = "Water Average"
DATE_COL = "date"

# Number of rows to trim from beginning/end of each year after feature
# engineering.  These rows have incomplete lag/forecast features filled with
# -999.  Set to 0 to skip trimming (the old commit commented this out).
TRIM_ROWS = 120


# ---------------------------------------------------------------------------
#                           TrainingLogger callback
# ---------------------------------------------------------------------------
class TrainingLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file):
        super().__init__()
        self.log_file = log_file

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_message = (
            f"Epoch {epoch + 1}: "
            + ", ".join([f"{key}={value:.4f}" for key, value in logs.items()])
            + "\n"
        )
        print(log_message, end="")
        with open(self.log_file, "a") as f:
            f.write(log_message)


# ---------------------------------------------------------------------------
#                             readingData
# ---------------------------------------------------------------------------
def readingData(path_to_data):
    """Read all CSVs from *path_to_data*, sorted alphabetically.

    Returns
    -------
    csv_files : list[str]
        Sorted list of file paths (needed by callers for independent-year lookup).
    data_list : list[pd.DataFrame]
        One DataFrame per CSV, in the same order as *csv_files*.
    """
    csv_files = sorted(glob.glob(os.path.join(path_to_data, "*.csv")))
    print(f"Found {len(csv_files)} CSV files.")
    print("Files:", [os.path.basename(f) for f in csv_files])

    data_list = [pd.read_csv(f) for f in csv_files]
    return csv_files, data_list


# ---------------------------------------------------------------------------
#                       creatingAdditionalColumns
# ---------------------------------------------------------------------------
def creatingAdditionalColumns(
    df,
    input_structure,
    input_hours_forecast,
    atp_hours_back,
    wtp_hours_back,
    pred_atp_interval,
    IPPOffset=0.0,
    trim_rows=TRIM_ROWS,
):
    """Vectorized feature engineering using pandas.shift().

    Creates lagged air/water temperature columns, forward-looking perfect-
    prognosis air temperature columns, and the target water temperature column.

    Parameters
    ----------
    trim_rows : int
        Number of rows to drop from the start and end of the DataFrame to
        remove incomplete lag/forecast padding.  Pass 0 to skip.
    """
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    interval = pred_atp_interval
    new_columns = {}

    df_work = df.copy()

    # Apply IPP offset vectorized
    if IPPOffset != 0.0:
        mask = df_work[AIR_COL] != -999
        df_work.loc[mask, AIR_COL] = df_work.loc[mask, AIR_COL] + IPPOffset

    # Lagged air temperature columns
    for lag in range(1, atp_hours_back + 1):
        col_name = f"airTemperature__{lag}h_ago"
        new_columns[col_name] = df_work[AIR_COL].shift(lag).fillna(-999).values

    # Lagged water temperature columns
    for lag in range(1, wtp_hours_back + 1):
        col_name = f"waterTemperature__{lag}h_ago"
        new_columns[col_name] = df_work[WATER_COL].shift(lag).fillna(-999).values

    # Forward-looking air temperature columns (perfect prognosis)
    for hours_ahead in range(interval, input_hours_forecast + 1, interval):
        col_name = f"airTemperature_pred__{hours_ahead}h_forecast"
        new_columns[col_name] = df_work[AIR_COL].shift(-hours_ahead).fillna(-999).values

    # Target: forward-looking water temperature
    target_col = f"waterTemperature_{input_hours_forecast}h_forecast"
    new_columns[target_col] = df_work[WATER_COL].shift(-input_hours_forecast).fillna(-999).values

    df = pd.concat([df_work, pd.DataFrame(new_columns)], axis=1)

    # Trim incomplete rows from edges
    if trim_rows > 0:
        df = df.iloc[trim_rows:]
        df = df.iloc[:-trim_rows]

    # Reorder columns for descending input structure
    if input_structure == "descending":
        water_temp_columns = [c for c in df.columns if "waterTemperature__" in c]
        air_temp_columns = [c for c in df.columns if "airTemperature__" in c and "_ago" in c]
        forecast_columns = [c for c in df.columns if "forecast" in c]
        other_columns = [
            c for c in df.columns
            if c not in water_temp_columns + air_temp_columns + forecast_columns
        ]
        if WATER_COL in other_columns:
            other_columns.remove(WATER_COL)
        if AIR_COL in other_columns:
            other_columns.remove(AIR_COL)

        reordered_columns = (
            other_columns
            + water_temp_columns[::-1]
            + [WATER_COL]
            + air_temp_columns[::-1]
            + [AIR_COL]
            + forecast_columns
        )
        df = df[reordered_columns]

    return df


# ---------------------------------------------------------------------------
#                            splittingData
# ---------------------------------------------------------------------------
def splittingData(year_list, cycle, independent_year=None):
    """Split N years into training / validation / testing via cyclic rotation.

    Parameters
    ----------
    year_list : list[pd.DataFrame]
        Engineered DataFrames for each year (any length >= 3).
    cycle : int
        Which rotation to use (0-indexed).
    independent_year : pd.DataFrame or None
        If provided, this DataFrame is used as the testing set instead of the
        last year in the rotation.

    Returns
    -------
    training, testing, validation : pd.DataFrame
    """
    years = list(year_list)  # shallow copy so rotation doesn't mutate caller

    for j in range(cycle + 1):
        for i in range(len(years)):
            if i > 0:
                years = [years[-1]] + years[:-1]

        training = pd.concat(years[:-2], ignore_index=True)
        validation = years[-2]

        if independent_year is not None:
            print("USING INDEPENDENT TEST YEAR")
            testing = independent_year
        else:
            print("USING REGULAR CYCLE TESTING")
            testing = years[-1]

        if j == cycle:
            return training, testing, validation

        # reset for next rotation iteration (only reached if cycle > 0)


# ---------------------------------------------------------------------------
#                          countingMissingValues
# ---------------------------------------------------------------------------
def countingMissingValues(df):
    """Count rows with at least one missing value (NaN or -999)."""
    if len(df) == 0:
        return 0, 0.0

    missing_standard = df.isna().any(axis=1)
    missing_custom = df.isin([-999]).any(axis=1)
    num_missing = (missing_standard | missing_custom).sum()
    pct_missing = (num_missing / len(df)) * 100
    return int(num_missing), pct_missing


# ---------------------------------------------------------------------------
#                         deletingMissingValues
# ---------------------------------------------------------------------------
def deletingMissingValues(df):
    """Drop rows where any non-date column is -999 or NaN."""
    cols_to_check = [c for c in df.columns if c != DATE_COL]
    mask = df[cols_to_check].isin([-999]).any(axis=1) | df[cols_to_check].isna().any(axis=1)
    df = df[~mask]
    df = df.dropna(subset=cols_to_check)
    return df


# ---------------------------------------------------------------------------
#                              reshaping
# ---------------------------------------------------------------------------
def reshaping(input_structure, training, testing, validation, model):
    """Extract numpy arrays (X, y) from each split DataFrame."""
    input_column_start = 1 if input_structure == "descending" else 3

    def _extract(df):
        if df.empty:
            return np.empty((0, 0)), np.empty((0,))
        X = df.iloc[:, input_column_start:-1].values.astype(float)
        y = df.iloc[:, -1].values.astype(float)
        return X, y

    x_train, y_train = _extract(training)
    x_val, y_val = _extract(validation)
    x_test, y_test = _extract(testing)

    if model == "LSTM":
        x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        x_val = x_val.reshape((x_val.shape[0], 1, x_val.shape[1]))
        y_train = np.expand_dims(y_train, axis=-1)
        y_test = np.expand_dims(y_test, axis=-1)
        y_val = np.expand_dims(y_val, axis=-1)

    return x_train, y_train, x_val, y_val, x_test, y_test


# ---------------------------------------------------------------------------
#                          dateTimeRetriever
# ---------------------------------------------------------------------------
def dateTimeRetriever(dataset, input_hours_forecast):
    """Parse the date column and shift forward by lead time hours."""
    dataset[DATE_COL] = (
        pd.to_datetime(dataset[DATE_COL], utc=True)
        + pd.DateOffset(hours=input_hours_forecast)
    )
    return dataset[DATE_COL].tolist()


# ---------------------------------------------------------------------------
#                        prepare_independent_year
# ---------------------------------------------------------------------------
def prepare_independent_year(
    csv_path,
    input_structure,
    lead_time,
    atp_hours_back,
    wtp_hours_back,
    pred_atp_interval=1,
    IPPOffset=0.0,
):
    """Prepare a single independent-year CSV for evaluation.

    Runs the same feature-engineering pipeline as preparingData() but on one
    CSV that readingData() intentionally skips (index 0).

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    dates : list
    """
    df_raw = pd.read_csv(csv_path)
    df_features = creatingAdditionalColumns(
        df=df_raw,
        input_structure=input_structure,
        input_hours_forecast=lead_time,
        atp_hours_back=atp_hours_back,
        wtp_hours_back=wtp_hours_back,
        pred_atp_interval=pred_atp_interval,
        IPPOffset=IPPOffset,
    )
    df_clean = deletingMissingValues(df_features)
    dates = dateTimeRetriever(df_clean.copy(), lead_time)

    col_start = 1 if input_structure == "descending" else 3
    X = df_clean.iloc[:, col_start:-1].values.astype(float)
    y = df_clean.iloc[:, -1].values.astype(float)
    return X, y, dates


# ---------------------------------------------------------------------------
#                            offSetCreator
# ---------------------------------------------------------------------------
def offSetCreator(dataYear, IPPOffset, input_hours_forecast):
    """Apply IPP offset to air temperature column (vectorized)."""
    if IPPOffset != 0.0:
        mask = dataYear[AIR_COL] != -999
        dataYear.loc[mask, AIR_COL] = dataYear.loc[mask, AIR_COL] + IPPOffset
    return dataYear


# ---------------------------------------------------------------------------
#                          dataframe_checker
# ---------------------------------------------------------------------------
def dataframe_checker(checkNum, dfList):
    """Raise ValueError if any non-date numeric column has a value < checkNum."""
    for i, df in enumerate(dfList):
        if df.empty:
            print(f"Warning: DataFrame {i} is empty, skipping check.")
            continue
        numeric_cols = df.drop(columns=[df.columns[0]])
        if (numeric_cols < checkNum).any().any():
            raise ValueError(
                f"DataFrame {i} contains numbers lower than {checkNum}"
            )


# ---------------------------------------------------------------------------
#                           preparingData
# ---------------------------------------------------------------------------
def preparingData(
    path_to_data,
    input_structure,
    independent_year,
    input_hours_forecast,
    atp_hours_back,
    wtp_hours_back,
    pred_atp_interval,
    IPPOffset=0.0,
    cycle=0,
    model="MLP",
    verbose=0,
):
    """Main entry point: read CSVs, engineer features, split, reshape.

    Works with any number of year-CSVs found in *path_to_data*.
    The first file (alphabetically) is reserved as the independent test year
    and is skipped during normal training.

    Parameters
    ----------
    independent_year : str or 'cycle'
        If 'cycle', testing comes from the rotation. Otherwise pass a year
        identifier (e.g. '2021') — the first sorted CSV is loaded and
        feature-engineered as the independent test set.
    """
    csv_files, all_data = readingData(path_to_data)

    # First file is reserved for independent testing; rest are training pool
    independent_csv = csv_files[0]
    training_data_raw = all_data[1:]
    n_years = len(training_data_raw)
    print(f"Using {n_years} years for train/val/test rotation.")

    if verbose >= 3:
        for i, v in enumerate(training_data_raw):
            print(f"  Year {i} columns: {list(v.columns)}")

    # Feature engineering on each year
    start_time = datetime.now()
    engineered_years = [
        creatingAdditionalColumns(
            df, input_structure, input_hours_forecast,
            atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset,
        )
        for df in training_data_raw
    ]
    print("finished input construction")
    end_time = datetime.now()

    if verbose >= 3:
        for i, v in enumerate(engineered_years):
            print(f"  Engineered year {i}: {v.shape}, cols: {list(v.columns)[:5]}...{list(v.columns)[-3:]}")

    # Handle independent year
    year_independent = None
    if independent_year != "cycle":
        year_independent = creatingAdditionalColumns(
            pd.read_csv(independent_csv),
            input_structure, input_hours_forecast,
            atp_hours_back, wtp_hours_back, pred_atp_interval, IPPOffset,
        )

    # Split into train / test / validation
    training_data, testing_data, validation_data = splittingData(
        engineered_years, cycle, independent_year=year_independent,
    )

    print("finished splitting the data")
    print(f"Training data shape: {training_data.shape}")
    print(f"Testing data shape:  {testing_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")

    # Missing value report
    for label, data in [("Training", training_data), ("Testing", testing_data), ("Validation", validation_data)]:
        num_miss, pct_miss = countingMissingValues(data)
        print(f"\n{label} Missing Values: {num_miss}")
        print(f"{label} Percentage of Missing Values: {round(pct_miss, 4)} %")

    dataframe_checker(-999, [training_data, testing_data, validation_data])

    # Delete missing values
    training = deletingMissingValues(training_data)
    testing = deletingMissingValues(testing_data)
    validation = deletingMissingValues(validation_data)

    dataframe_checker(-100, [training, testing, validation])

    if verbose >= 2:
        print(f"\nAfter deleting missing: train={training.shape}, test={testing.shape}, val={validation.shape}")

    # Extract dates and air temps
    training_dates = dateTimeRetriever(training.copy(), input_hours_forecast) if not training.empty else []
    validation_dates = dateTimeRetriever(validation.copy(), input_hours_forecast) if not validation.empty else []
    if not testing.empty:
        testingDates = dateTimeRetriever(testing.copy(), input_hours_forecast)
        testingAirTemps = testing[AIR_COL].tolist()
    else:
        testingDates = []
        testingAirTemps = []

    # Reshape for model input
    x_train, y_train, x_val, y_val, x_test, y_test = reshaping(
        input_structure, training, testing, validation, model,
    )

    if verbose >= 2:
        print(f"\nFinal shapes: x_train={x_train.shape}, y_train={y_train.shape}")
        print(f"  x_val={x_val.shape}, y_val={y_val.shape}")
        print(f"  x_test={x_test.shape}, y_test={y_test.shape}")

    if verbose >= 3:
        print(f"NaNs in x_train: {np.isnan(x_train).sum()}")
        print(f"NaNs in y_train: {np.isnan(y_train).sum()}")
        print(f"Infs in x_train: {np.isinf(x_train).sum()}")
        print(f"Infs in y_train: {np.isinf(y_train).sum()}")

    return (
        x_train, y_train, x_val, y_val, x_test, y_test,
        training_dates, validation_dates, testingDates, testingAirTemps,
    )


# ===========================================================================
#                        METRICS / LOSS FUNCTIONS
# ===========================================================================

def _ensemble_mean(y_pred):
    """If y_pred has multiple outputs, return the ensemble mean."""
    if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        return mean_pred
    return y_pred


# ---------------------------------------------------------------------------
#  CRPS — the REAL implementation (Gneiting & Raftery 2005)
#    CRPS(F, x) = E|y_pred - y_true| - 0.5 * E|y_pred_i - y_pred_j|
# ---------------------------------------------------------------------------
def crps(y_true, y_pred):
    """Continuous Ranked Probability Score for finite ensemble members.

    From Ryan Lagerquist, based on:
        Gneiting & Raftery (2005) — Strictly proper scoring rules.
        Hersbach (2000) — Decomposition of CRPS for ensemble prediction.

    References
    ----------
    https://www.stat.washington.edu/research/reports/2004/tr463R.pdf
    https://doi.org/10.1175/1520-0434(2000)015<0559:DOTCRP>2.0.CO;2
    """
    term_one = tf.reduce_mean(tf.abs(tf.subtract(y_true, y_pred)), axis=-1)

    term_two = tf.reduce_mean(
        tf.abs(
            tf.subtract(
                tf.expand_dims(y_pred, -1),
                tf.expand_dims(y_pred, -2),
            )
        ),
        axis=(-2, -1),
    )

    half = tf.constant(-0.5, dtype=term_two.dtype)
    score = tf.add(term_one, tf.multiply(half, term_two))
    return tf.reduce_mean(score)


def crps_loss(y_true, y_pred):
    """CRPS as a Keras-compatible loss function (same as crps())."""
    return crps(y_true, y_pred)


# ---------------------------------------------------------------------------
#  Standard metrics (all ensemble-aware)
# ---------------------------------------------------------------------------
def mae(y_true, y_pred):
    mean_pred = _ensemble_mean(y_pred)
    return tf.reduce_mean(tf.abs(tf.subtract(y_true, mean_pred))).numpy()


def mae12(y_true, y_pred):
    """MAE for observations below 12 C (cold stunning threshold)."""
    if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
        mean_pred = tf.reduce_mean(y_pred, axis=-1, keepdims=True)
        true_val = y_true[:, :1]
    else:
        mean_pred = y_pred
        true_val = y_true

    mask = true_val < 12
    filtered_true = tf.boolean_mask(true_val, mask)
    filtered_pred = tf.boolean_mask(mean_pred, mask)

    if tf.size(filtered_true) == 0:
        return 0.0
    return tf.reduce_mean(tf.abs(filtered_true - filtered_pred)).numpy()


def mse(y_true, y_pred):
    mean_pred = _ensemble_mean(y_pred)
    return tf.reduce_mean(tf.square(tf.subtract(y_true, mean_pred))).numpy()


def rmse(y_true, y_pred):
    mean_pred = _ensemble_mean(y_pred)
    per_sample = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_true, mean_pred)), axis=-1))
    return tf.reduce_mean(per_sample).numpy()


# Alias used by ssrat_avg — identical to rmse
rmse_avg = rmse


def me(y_true, y_pred):
    """Mean Error (bias)."""
    mean_pred = _ensemble_mean(y_pred)
    return tf.reduce_mean(tf.subtract(y_true, mean_pred)).numpy()


def me12(y_true, y_pred):
    """Mean Error for observations below 12 C.

    Author: Hector M. Marrero-Colominas
    """
    meanErrBelow12List = []
    for i in range(len(y_true)):
        if y_true[i] < 12:
            meanErrBelow12List.append(float(y_pred[i] - y_true[i]))
    if not meanErrBelow12List:
        return 0.0
    return float(np.mean(meanErrBelow12List))


def y_pred_std(y_true, y_pred):
    return tf.math.reduce_mean(tf.math.reduce_std(y_pred, axis=-1)).numpy()


# ---------------------------------------------------------------------------
#  Spread-Skill metrics (UQ)
# ---------------------------------------------------------------------------
def ssrat(y_true, y_pred):
    """Spread Skill Ratio (SSRAT)."""
    std = tf.math.reduce_std(y_pred, axis=-1)
    return (tf.math.reduce_mean(std) / rmse(y_true, y_pred)).numpy()


def ssrat_avg(y_true, y_pred, y_std):
    return (tf.math.reduce_mean(y_std) / rmse_avg(y_true, y_pred)).numpy()


def ryan_ssrel(y_true, y_pred, y_std=None):
    """Spread-Skill Reliability (Ryan Lagerquist's formulation)."""
    y_true = np.array(tf.expand_dims(y_true, axis=-1))
    y_pred = np.array(y_pred)
    if y_std is None:
        y_std = np.std(y_pred, axis=-1)
    else:
        y_std = np.array(y_std)

    nPts = y_true.shape[0]
    minBin = min(0.0, y_std.min())
    maxBin = np.ceil(max(rmse(y_true, y_pred), y_std.max()))

    nBins = 10
    bins = np.linspace(minBin, maxBin, nBins + 1)
    ssRel = 0.0

    for i in range(nBins):
        refs = np.logical_and(y_std >= bins[i], y_std < bins[i + 1])
        nPtsBin = np.count_nonzero(refs)
        if nPtsBin > 0:
            error_i = rmse(y_true[refs], y_pred[refs])
            spread_i = np.mean(y_std[refs])
            ssRel += (nPtsBin / nPts) * np.abs(error_i - spread_i)

    return ssRel


# ---------------------------------------------------------------------------
#  PITD: Probability Integral Transform Distance
# ---------------------------------------------------------------------------
def pitd(y_true, y_pred):
    y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
    y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    pit_bins = np.arange(0.0, 1.1, 0.1)
    nEns = y_pred.shape[-1]
    nSamples = y_true.shape[0]

    ytrueT = y_true.reshape(-1)
    ypredTS = np.sort(y_pred.reshape((nSamples, nEns)), axis=1)

    ytrueTE = np.repeat(ytrueT[..., np.newaxis], nEns, axis=-1)
    pred_diff = np.abs(ytrueTE - ypredTS)
    pit_values = np.argmin(pred_diff, axis=-1) / nEns
    weights = np.ones_like(pit_values) / nSamples

    pit_counts, _ = np.histogram(pit_values, bins=pit_bins, weights=weights)

    # D-value
    nBins = pit_counts.shape[0]
    pit_freq = pit_counts / np.sum(pit_counts)
    uniform = 1.0 / nBins
    dvalue = np.sqrt(np.mean((pit_freq - uniform) ** 2))
    return dvalue


# ---------------------------------------------------------------------------
#  Cold-water error metrics
# ---------------------------------------------------------------------------
def errorBelow12c(y_true, y_pred):
    """Mean error and MAE for observations below 12 C.

    Returns
    -------
    meanErrorBelow12 : float
    maeBelow12 : float
    maeBelow12List : list[float]
    """
    meanErrList = []
    maeList = []
    for i in range(len(y_true)):
        if y_true[i] < 12:
            residual = float(y_pred[i] - y_true[i])
            meanErrList.append(residual)
            maeList.append(abs(residual))

    if not meanErrList:
        return 0.0, 0.0, []
    return float(np.mean(meanErrList)), float(np.mean(maeList)), maeList


def max10PercentError(y_true, y_pred):
    """Mean of the top 10% largest absolute errors.

    Author: Hector M. Marrero-Colominas
    """
    abs_residuals = np.abs(np.array(y_pred) - np.array(y_true))
    n_top = max(1, int(len(abs_residuals) * 0.1))
    top10 = np.sort(abs_residuals)[::-1][:n_top]
    return float(np.round(np.mean(top10), 4))


# ---------------------------------------------------------------------------
#                              __main__
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("utils_mse_crps_optimized — smoke test")
    x_train, y_train, x_val, y_val, x_test, y_test, *_ = preparingData(
        "data/ESB_datasets",
        "descending",
        "cycle",
        12,
        24,
        24,
        1,
        cycle=1,
        model="MSE",
        verbose=0,
    )
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
