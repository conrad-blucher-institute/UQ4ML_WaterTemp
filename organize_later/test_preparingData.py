from src.helper.utils_mse_crps import preparingData

x_train, y_train, x_val, y_val, x_test, y_test, training_dates, validation_dates, testingDates, testingAir = preparingData(
    "data/ESB_datasets",
    "descending",
    "2021",  # This will use the independent year
    96,      # leadtime
    24,      # atp_hours_back
    24,      # wtp_hours_back
    1,       # pred_atp_interval
    cycle=1,
    model="MSE",
    verbose=0
)