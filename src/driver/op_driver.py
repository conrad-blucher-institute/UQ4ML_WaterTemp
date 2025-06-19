# add documentation here -hector 6-11
import subprocess

subprocess.run("conda env update -f environment.yml", shell=True, check=True)

if __name__ == "__main__":

    # Parse incoming command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    
    if args.model == 'pnn':
        # call cmd_ai_builder.py
        subprocess.run(f"python pnn_mme_driver.py @c/{}")
        pass

    elif args.model == 'mse':
        # call mme_runner.py
        subprocess.run(f"python mse_mme_driver.py @c/{}")
        pass

    elif args.model == 'crps':
        # call mme_runner.py
        subprocess.run(f"python crps_mme_driver.py @c/{args.}")
        pass