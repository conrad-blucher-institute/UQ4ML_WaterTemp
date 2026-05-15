# inference_driver's job:
# for every saved (model, scaler) pair under a results directory,
# run predictions on a target csv and save them to a parallel output directory

from pathlib import Path
import keras
import joblib
import pandas as pd

from src.helper.utils_mse_crps import prepare_independent_year
from src.helper.my_parser import create_parser


def run_inference(args):
    # find  model directorries -> loop over directories -> load model n scaler -> prep data  -> predict -> save predictions
    leadtime_dir = Path(args.results_folder) / f"{args.c_leadtime}h"
    model_dirs = sorted(leadtime_dir.iterdir())

    predictions_column_names = [f"pred_{k+1}" for k in range(args.num_output_neurons)]

    for model_dir in model_dirs:
        keras_files = list(model_dir.glob("*.keras"))
        scaler_files = list(model_dir.glob("*.joblib"))

        if not keras_files or not scaler_files:
            print(f"Skipping {model_dir} - missing model or scaler.")
            continue
    
        model_path = keras_files[0]
        scaler_path = scaler_files[0]

        model = keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        X, y, dates = prepare_independent_year(
            csv_path=args.inference_csv,
            input_structure=args.input_structure,
            lead_time=args.c_leadtime,
            atp_hours_back=args.atp_hours_back,
            wtp_hours_back=args.wtp_hours_back,
            pred_atp_interval=args.pred_atp_interval,
            scaler=scaler,
            column_map=None
        )

        predictions = model.predict(X)

        preds_df = pd.DataFrame(columns=predictions_column_names, data=predictions)
        preds_df.insert(0, "date_time", dates)
        preds_df.insert(1, "target", y)

        print(f" [{model_dir.name}] predicted {len(predictions)} samples.")

        # build the output path of the inference results
        output_dir = Path(args.inference_output_folder) / f"{args.c_leadtime}h" / model_dir.name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "predictions.csv"
        preds_df.to_csv(output_path, index=False)



if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run_inference(args)