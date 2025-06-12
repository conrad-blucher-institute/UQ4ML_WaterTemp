import subprocess

subprocess.run("conda env update -f environment.yml", shell=True, check=True)
subprocess.run("pip install -r requirements.txt", shell=True, check=True)

print("Environment setup complete.")

subprocess.run("export PYTHONPATH=$PYTHONPATH:$(pwd)/src/driver", shell=True, check=True)
subprocess.run("ln -s configs c", shell=True, check=True)

print("Exports and Symlink created.")

# python pnn_driver.py @c/pnn_12.txt
