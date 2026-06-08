"""
CRPS Hyperparameter Tuner

Grid search over activation functions with locked other hyperparameters.
"""

import sys
import argparse
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from esb_03-2026_tune_experiment.base_tuner import BaseHyperparameterTuner
from esb_03-2026_tune_experiment.tuner_utils import GridSearchConfig

# Import your model building and training logic here
# from src.driver.your_training_module import build_crps_model, train_crps_model


class CRPSTuner(BaseHyperparameterTuner):
    """CRPS-specific hyperparameter tuner."""
    
    def __init__(self, output_dir: Path, max_workers: int = 4):
        super().__init__("CRPS", output_dir, max_workers)
        
        # TODO: Load training data once (for all workers)
        # self.train_data = load_train_data()
        # self.val_data = load_val_data()
        # self.test_data = load_test_data()
    
    def get_grid_search_config(self) -> GridSearchConfig:
        """
        Define the grid search space for CRPS.
        Focus on activation functions while keeping other hyperparameters locked.
        """
        return GridSearchConfig(
            model_type="CRPS",
            lead_times=[12, 48, 96, 120],           # All ESB lead times
            cycles=[0, 1, 2, 3],                              # CRPS uses cycles 0-3
            activations=['relu', 'leaky_relu', 'selu', 'sigmoid', 'tanh'],  # NEW: more activation functions
            num_layers_range=[2, 3],                # Locked: test a few layer counts
            neurons_range=[16, 32, 64, 100, 128, 256]             # Locked: test a few neuron counts
        )
    
    def _build_model(self, config: dict):
        """
        Build a CRPS model with the given configuration.
        
        TODO: Implement this using your model building logic.
        """
        # Example structure (replace with actual implementation):
        # return build_crps_model(
        #     activation=config['activation'],
        #     num_layers=config['num_layers'],
        #     neurons=config['neurons'],
        #     lead_time=config['lead_time'],
        #     input_units=24  # or whatever your input is
        # )
        raise NotImplementedError(
            "Implement CRPS model building using your training logic. "
            "See operational_mse_crps_driver.py for reference."
        )
    
    def _train_model(self, model, config: dict) -> float:
        """
        Train the CRPS model and return the validation loss.
        
        TODO: Implement this using your training logic.
        """
        # Example structure (replace with actual implementation):
        # history = train_crps_model(
        #     model=model,
        #     train_data=self.train_data,
        #     val_data=self.val_data,
        #     epochs=50,
        #     batch_size=32
        # )
        # return history['val_loss'][-1]  # Return final validation loss
        
        raise NotImplementedError(
            "Implement CRPS model training using your training logic. "
            "See operational_mse_crps_driver.py for reference."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRPS Hyperparameter Tuner")
    parser.add_argument("--output", type=Path, default="./crps_tune_results",
                        help="Output directory for progress and timing logs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    tuner = CRPSTuner(args.output, max_workers=args.workers)
    tuner.run_tuning()
