"""
MAPE Hyperparameter Tuner

Grid search over activation functions with locked other hyperparameters.
"""

import sys
import argparse
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from base_tuner import BaseHyperparameterTuner
from tuner_utils import GridSearchConfig


class MAPETuner(BaseHyperparameterTuner):
    """MAPE-specific hyperparameter tuner."""
    
    def __init__(self, output_dir: Path, max_workers: int = 4):
        super().__init__("MAPE", output_dir, max_workers)
        
        # TODO: Load training data once (for all workers)
        # self.train_data = load_train_data()
        # self.val_data = load_val_data()
        # self.test_data = load_test_data()
    
    def get_grid_search_config(self) -> GridSearchConfig:
        """
        Define the grid search space for MAPE.
        Focus on activation functions while keeping other hyperparameters locked.
        """
        return GridSearchConfig(
            model_type="MAPE",
            lead_times=[12, 48, 96, 120],           # All ESB lead times
            cycles=[0, 1, 2, 3],                    # MAPE uses cycles 0-3
            activations=['relu', 'leaky_relu', 'selu', 'sigmoid', 'tanh'],  # NEW: more activation functions
            num_layers_range=[1, 2, 3],             # Locked: test a few layer counts
            neurons_range=[16, 32, 64, 100, 128, 256]         # Locked: test a few neuron counts
        )
    
    def _build_model(self, config: dict):
        """
        Build a MAPE model with the given configuration.
        
        TODO: Implement this using your model building logic.
        """
        # Build a simple feed-forward model following the pattern in crps_mme_runner.py
        import tensorflow as tf
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import Dense

        # require keys to be present; KeyError will surface if missing
        activation = config['activation']
        num_layers = int(config['num_layers'])
        neurons = int(config['neurons'])
        output_units = int(config['output_units'])
        output_activation = config['output_activation']
        learning_rate = float(config['learning_rate'])
        input_shape = tuple(config['input_shape'])

        model = Sequential()

        # Add explicit Input layer (matches MyHyperModel behavior)
        from tensorflow.keras.layers import Input
        model.add(Input(shape=input_shape))

        # Add hidden layers
        for _ in range(num_layers):
            model.add(Dense(units=neurons, activation=activation))

        # Output layer
        model.add(Dense(output_units, activation=output_activation))

        # Compile with Adam and MAPE loss; record MAE as additional metric
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mape', metrics=['mape', 'mae'])

        return model
    
    def _train_model(self, model, config: dict) -> float:
        """
        Train the MAPE model and return the validation loss.
        
        TODO: Implement this using your training logic.
        """
        # Example structure (replace with actual implementation):
        # history = train_mape_model(
        #     model=model,
        #     train_data=self.train_data,
        #     val_data=self.val_data,
        #     epochs=50,
        #     batch_size=32
        # )
        # return history['val_loss'][-1]  # Return final validation loss

        # Prepare kwargs for preparingData() by extracting expected keys from config
        try:
            from src.helper.utils_mse_crps import preparingData
        except Exception:
            try:
                from helper.utils_mse_crps import preparingData
            except Exception:
                raise RuntimeError('preparingData() not found in project')

        allowed_params = [
            'path_to_data', 'input_structure', 'independent_year', 'input_hours_forecast',
            'atp_hours_back', 'wtp_hours_back', 'pred_atp_interval', 'IPPOffset',
            'cycle', 'model', 'verbose'
        ]

        # First call: get a preview to infer input shape
        default_data_path = str(_REPO_ROOT / 'data' / 'ESB_datasets')
        prep_kwargs = {
            'path_to_data': config.get('path_to_data', default_data_path),
            'input_structure': config.get('input_structure', 'descending'),
            'independent_year': config.get('independent_year', 'cycle'),
            'input_hours_forecast': config.get('input_hours_forecast', config.get('lead_time')),
            'atp_hours_back': config.get('atp_hours_back', 24),
            'wtp_hours_back': config.get('wtp_hours_back', 24),
            'pred_atp_interval': config.get('pred_atp_interval', 1),
            'IPPOffset': config.get('IPPOffset', 0.0),
            'cycle': config.get('cycle', 0),
            'model': config.get('model', config.get('model_type', 'MAPE')),
            'verbose': config.get('verbose', 0)
        }

        preview = preparingData(**prep_kwargs)

        input_shape = None
        if preview is not None:
            try:
                X_preview = preview[0]
                if hasattr(X_preview, 'shape'):
                    shp = X_preview.shape
                    input_shape = tuple(shp[1:]) if len(shp) >= 2 else (shp[0],)
            except Exception:
                input_shape = None

        if input_shape is not None and 'input_shape' not in config:
            # store as list for JSON-serializable configs
            config['input_shape'] = list(input_shape)

        # Now build the model (with injected input shape) from preview
        model = self._build_model(config)

        # Get the actual training/validation data. Reuse the earlier preview
        # if available to avoid duplicate work; otherwise call preparingData().
        if preview is None:
            data = preparingData(**prep_kwargs)
        else:
            data = preview

        if data is None:
            raise RuntimeError('preparingData() not available or returned None')

        # Unpack data in common formats
        try:
            if len(data) >= 4:
                X_train, y_train, X_val, y_val = data[:4]
                validation = (getattr(X_val, 'values', X_val), getattr(y_val, 'values', y_val))
                x_train = getattr(X_train, 'values', X_train)
                y_train = getattr(y_train, 'values', y_train)
            elif len(data) == 2:
                X_train, y_train = data
                x_train = getattr(X_train, 'values', X_train)
                y_train = getattr(y_train, 'values', y_train)
                validation = None
            else:
                raise RuntimeError('preparingData() returned unexpected format')
        except Exception as e:
            raise RuntimeError(f'Error unpacking data from preparingData(): {e}')

        # Determine input shape and batch size from the actual data (ensures correctness)
        try:
            input_shape = x_train[0].shape
        except Exception:
            try:
                shp = x_train.shape
                input_shape = tuple(shp[1:]) if len(shp) >= 2 else (shp[0],)
            except Exception:
                input_shape = None

        if input_shape is not None:
            # update config and rebuild model if needed
            config['input_shape'] = list(input_shape)
            model = self._build_model(config)

        # Training hyperparameters (allow override in config)
        epochs = int(config.get('epochs'))

        # Determine batch_size: require explicit config or infer from x_train shape
        if 'batch_size' in config:
            batch_size = int(config['batch_size'])
        else:
            if hasattr(x_train, 'shape') and getattr(x_train, 'shape', (None,))[0] is not None:
                batch_size = int(x_train.shape[0])
            else:
                raise RuntimeError('batch_size not specified in config and cannot be inferred from x_train')

        # Callbacks (best-effort import)
        callbacks = []
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            callbacks = [
                EarlyStopping(monitor='val_loss', 
                            min_delta=0.001,
                            patience=25,
                            verbose=2,
                            mode='auto', restore_best_weights=True),

                ReduceLROnPlateau(monitor='val_loss',
                                min_delta=0.001,
                                factor=0.1, 
                                patience=15,
                                min_lr=0.00001, 
                                verbose=2)
            ]
        except Exception:
            callbacks = []

        # Fit the model
        fit_kwargs = dict(epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        try:
            if validation is not None:
                history = model.fit(x_train, y_train, validation_data=validation, **fit_kwargs)
            else:
                history = model.fit(x_train, y_train, validation_split=0.2, **fit_kwargs)
        except Exception as e:
            raise RuntimeError(f'Model training failed: {e}')

        # Extract final validation loss (or training loss if not available)
        hist = getattr(history, 'history', {})
        if 'val_loss' in hist and len(hist['val_loss']) > 0:
            val_loss = float(hist['val_loss'][-1])
        elif 'loss' in hist and len(hist['loss']) > 0:
            val_loss = float(hist['loss'][-1])
        else:
            val_loss = -1.0

        # Return loss and all final metrics
        metrics_dict = {k: float(v[-1]) for k, v in hist.items() if isinstance(v, (list, tuple)) and len(v) > 0}
        return (val_loss, metrics_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPE Hyperparameter Tuner")
    parser.add_argument("--output", type=Path, default="./mape_tune_results",
                        help="Output directory for progress and timing logs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers")
    args = parser.parse_args()
    
    tuner = MAPETuner(args.output, max_workers=args.workers)
    tuner.run_tuning()
