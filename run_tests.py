# run_tests.py
import subprocess

print("Starting test script...")

# Run test1.py
# subprocess.run(["python", "test1.py"], check=True)
subprocess.run(["python", "-m", "src.driver.operational_mse_crps_driver"], check=True)


# Run test2.py
# subprocess.run(["python", "test2.py"], check=True)
subprocess.run(["python", "-m", "src.driver.visualization_driver"], check=True)


print("Test script completed.")
