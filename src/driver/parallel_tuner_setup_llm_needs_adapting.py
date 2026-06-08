from itertools import product
from multiprocessing import Pool

LEADTIMES = [1, 7, 14, 30]
ROTATIONS = [0, 1, 2, 3]

def run_tuner_for_combo(combo):
    """Run GridSearch for one (leadtime, rotation) combo"""
    leadtime, rotation = combo
    
    tuner = GridSearch(
        build_model,
        objective='val_accuracy',
        max_trials=50,
        directory=f"tuner_lt{leadtime}_rot{rotation}"
    )
    
    # Get data for this specific combo
    X_train, y_train = get_data(leadtime, rotation)
    tuner.search(X_train, y_train, epochs=10)
    
    return {
        'leadtime': leadtime,
        'rotation': rotation,
        'best_hp': tuner.get_best_hyperparameters(),
        'best_score': tuner.oracle.get_best_trials(1)[0].score
    }

if __name__ == "__main__":
    # Create cartesian product of all combos
    combos = list(product(LEADTIMES, ROTATIONS))  # 16 total
    
    with Pool(processes=8) as pool:
        results = pool.map(run_tuner_for_combo, combos)
    
    # Aggregate results
    for result in results:
        print(f"LT={result['leadtime']}, Rot={result['rotation']}: {result['best_score']}")