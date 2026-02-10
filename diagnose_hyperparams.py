#!/usr/bin/env python3
"""
Diagnostic script to trace hyperparameter assignments in operational_mse_crps_driver.py
Shows which hyperparameters are assigned for each lead_time and compares against
existing result directories.
"""

from pathlib import Path
import json

model_name = "MAPE"
lead_time_list = [12, 48, 96, 120]

# Replicate the hyperparameter logic from operational_mse_crps_driver.py
hyperparams_by_leadtime = {}

print("=" * 80)
print("HYPERPARAMETER MAPPINGS IN OPERATIONAL DRIVER")
print("=" * 80)

for lead_time in lead_time_list:
    if lead_time == 12:
        if model_name == "MAPE":
            num_layers = 1
            act_func = 'leaky_relu'
            neurons = 100
    elif lead_time == 48:
        if model_name == "MAPE":
            num_layers = 3
            act_func = 'relu'
            neurons = 32
    elif lead_time == 96:
        if model_name == "MAPE":
            num_layers = 1
            act_func = 'leaky_relu'
            neurons = 256
    elif lead_time == 120:
        if model_name == "MAPE":
            num_layers = 3
            act_func = 'relu'
            neurons = 256
    
    combo_name = f"{model_name.lower()}-{num_layers}_layers-{act_func}-{neurons}_neurons"
    hyperparams_by_leadtime[lead_time] = {
        "num_layers": num_layers,
        "act_func": act_func,
        "neurons": neurons,
        "combo_name": combo_name
    }
    
    print(f"\n{lead_time}h:")
    print(f"  → Layers: {num_layers}, Activation: {act_func}, Neurons: {neurons}")
    print(f"  → Expected folder pattern: mape-{num_layers}_layers-{act_func}-{neurons}_neurons-cycle_X-iteration_Y")

print("\n" + "=" * 80)
print("ACTUAL DIRECTORIES IN src/results/mape_results/")
print("=" * 80)

base_path = Path("src/results/mape_results")
if not base_path.exists():
    print("ERROR: src/results/mape_results/ does not exist!")
else:
    for lead_dir in sorted(base_path.iterdir()):
        if lead_dir.is_dir():
            lead_str = lead_dir.name
            print(f"\n{lead_str}:")
            combos = list(lead_dir.iterdir())
            
            # Group by unique combo name
            unique_combos = {}
            for combo_dir in sorted(combos):
                if combo_dir.is_dir():
                    combo_name = "-".join(combo_dir.name.split("-")[:-2])  # Remove cycle and iteration
                    if combo_name not in unique_combos:
                        unique_combos[combo_name] = []
                    unique_combos[combo_name].append(combo_dir.name)
            
            for combo, instances in sorted(unique_combos.items()):
                print(f"  {combo}")
                print(f"    Found {len(instances)} instance(s): {instances[0]} ... (showing first)")
                
                # Extract neurons from combo
                parts = combo.split("-")
                for part in parts:
                    if "neurons" in part:
                        neurons_found = part.replace("_neurons", "")
                        try:
                            lead_int = int(lead_str.replace("h", ""))
                            expected = hyperparams_by_leadtime[lead_int]["neurons"]
                            if int(neurons_found) != expected:
                                print(f"    ⚠️  MISMATCH: Found {neurons_found} neurons, expected {expected}")
                        except:
                            pass

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for lead in [12, 48, 96, 120]:
    print(f"{lead}h should have: {hyperparams_by_leadtime[lead]['neurons']} neurons")
