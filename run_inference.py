import subprocess
import sys
from datetime import datetime, timedelta

from src.helper.my_parser import create_parser

if __name__ == "__main__":
    # add whatever models you want to run to this list
    configs = [
        "configs/mape/mape_12h.txt",
        "configs/mape/mape_48h.txt",
        "configs/mape/mape_96h.txt",
        "configs/mape/mape_120h.txt"
    ]

    if args.inference_csv is None or args.inference_output_folder is None:
        raise ValueError("Inference requires --inference_csv and --inference_output_folder in the config")

    run_summary = [] # contains each config file name; if it succeeded; and time it took to train
    failed_runs = [] # contains failed model runs, if any

    for cfg in configs:
        start = datetime.now()

        result = subprocess.run([sys.executable, "-m", "src.driver.inference_driver", f"@{cfg}"])

        end = datetime.now()
        elapsed = end - start

        run_summary.append({"config": cfg, "returncode": result.returncode, "elapsed":elapsed})

        status = "succeeded" if result.returncode == 0 else "failed"
        print(f"[{cfg}] {status} in {elapsed}")

    # total amount of time it took to run all the models in the list above
    total_elapsed = sum((r["elapsed"] for r in run_summary), start=timedelta())

    for r in run_summary:
        if r["returncode"] != 0:
            failed_runs.append(r)

    print("-------------------- RUN SUMMARY --------------------")

    print(f"{len(run_summary) - len(failed_runs)}/{len(run_summary)} configs succeeded.")
    print(f"Total elapsed time: {total_elapsed}")

    for fail in failed_runs:
        print(f"Failed runs: {fail['config']} | returncode = {fail['returncode']} | elapsed time = {str(fail['elapsed']).split('.')[0]}")
    
    if failed_runs:
        sys.exit(1)