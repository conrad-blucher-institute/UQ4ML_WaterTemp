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
    
    def __init__(self, output_dir: Path, max_workers: int = 4, keras_save_dir: Path = None, verbose: int = 0):
        super().__init__("MAPE", output_dir, max_workers, keras_save_dir=keras_save_dir, verbose=verbose)
        
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

        # Dropout rate + activation-aware layer type (Stage C, design §9):
        # AlphaDropout preserves SELU self-normalization (mean AND variance);
        # plain Dropout would silently break it. dropout == 0.0 -> no layer (off).
        dropout = float(config.get('dropout', 0.0))
        from tensorflow.keras.layers import Dropout, AlphaDropout
        DropoutLayer = AlphaDropout if activation == 'selu' else Dropout

        model = Sequential()

        # Add explicit Input layer (matches MyHyperModel behavior)
        from tensorflow.keras.layers import Input
        model.add(Input(shape=input_shape))

        # Add hidden layers, each optionally followed by a dropout layer
        for _ in range(num_layers):
            model.add(Dense(units=neurons, activation=activation))
            if dropout > 0.0:
                model.add(DropoutLayer(dropout))

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

        # --- scale stage (Stage C, decision C1; opt-in via config['scale']) ----
        # OFF by default -> identity passthrough, byte-identical to the unscaled
        # path (the Stage B golden digest d80ae54 stays valid). ON -> fit a
        # StandardScaler on x_train ONLY (leakage-checked inside scale_arrays) and
        # transform train/val/test with it. The fitted scaler is persisted next to
        # the .keras (single io site) and reused for the 2021 inference below, so
        # the independent year is scaled identically to training.
        fitted_scaler = None
        if config.get('scale', False):
            from esb.contracts import Arrays
            from esb.stages.scale import scale_arrays
            if len(data) < 6:
                raise RuntimeError(
                    'scale=True requires preparingData to return the full 6 arrays'
                )
            scaled = scale_arrays(Arrays.from_preparing_data(data), enabled=True)
            fitted_scaler = scaled.scaler
            x_train = scaled.x_train
            y_train = scaled.y_train
            validation = (scaled.x_val, scaled.y_val)

        # Determine input shape and batch size from the actual data (ensures correctness)
        try:
            input_shape = x_train[0].shape
        except Exception:
            try:
                shp = x_train.shape
                input_shape = tuple(shp[1:]) if len(shp) >= 2 else (shp[0],)
            except Exception:
                input_shape = None

        print(f"  DEBUG x_train.shape={getattr(x_train, 'shape', '?')}, inferred input_shape={input_shape}")

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

        # Callback knobs (R5): read from config so they are honest/overridable.
        # Defaults equal the previous hardcoded literals, so a default run is
        # byte-identical.
        monitor = config.get('call_back_monitor', 'val_loss')
        es_patience = int(config.get('early_stop_patience', 25))
        lr_patience = int(config.get('lr_reducer_patience', 15))
        min_delta = float(config.get('min_delta', 0.001))

        # Callbacks (best-effort import)
        callbacks = []
        try:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            callbacks = [
                EarlyStopping(monitor=monitor,
                            min_delta=min_delta,
                            patience=es_patience,
                            verbose=2,
                            mode='auto', restore_best_weights=True),

                ReduceLROnPlateau(monitor=monitor,
                                min_delta=min_delta,
                                factor=0.1,
                                patience=lr_patience,
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

        # Build metrics dict from Keras history
        metrics_dict = {k: float(v[-1]) for k, v in hist.items() if isinstance(v, (list, tuple)) and len(v) > 0}

        # Import mae12 once so it is available for both val and 2021 evaluation
        try:
            from src.helper.utils import mae12 as _mae12
        except Exception:
            try:
                from helper.utils import mae12 as _mae12
            except Exception:
                _mae12 = None

        # Post-hoc mae12 on validation set (y_true <= 12°C)
        if validation is not None and _mae12 is not None:
            try:
                x_val_arr, y_val_arr = validation
                y_pred_val = model.predict(x_val_arr, verbose=0)
                metrics_dict['val_mae12'] = float(_mae12(y_val_arr, y_pred_val))
            except Exception as e:
                print(f"  WARNING: mae12 val computation failed: {e}")

        # Post-hoc 2021 independent test set evaluation
        # Uses prepare_independent_year() which runs the same feature-engineering
        # pipeline as preparingData() on the CSV that readingData() skips.
        try:
            import glob as _glob
            import os as _os
            try:
                from src.helper.utils_mse_crps import prepare_independent_year as _prep2021
            except Exception:
                from helper.utils_mse_crps import prepare_independent_year as _prep2021
            data_path = config.get('path_to_data', str(_REPO_ROOT / 'data' / 'ESB_datasets'))
            csv_files = sorted(_glob.glob(_os.path.join(data_path, '*.csv')))
            if not csv_files:
                raise RuntimeError('No CSV files found in data path for 2021 evaluation')
            # First sorted CSV = esb_2020_2021.csv (the independent test year)
            x_2021, y_2021, _ = _prep2021(
                csv_path=csv_files[0],
                input_structure=config.get('input_structure', 'descending'),
                lead_time=config.get('input_hours_forecast', config.get('lead_time')),
                atp_hours_back=config.get('atp_hours_back', 24),
                wtp_hours_back=config.get('wtp_hours_back', 24),
                pred_atp_interval=config.get('pred_atp_interval', 1),
                IPPOffset=config.get('IPPOffset', 0.0),
                # Apply the SAME training-fit scaler (None when scaling is off) so
                # the independent year is scaled exactly like training — no re-fit.
                scaler=fitted_scaler,
            )
            if x_2021.shape[0] > 0:
                eval_results = model.evaluate(x_2021, y_2021, verbose=0)
                for name, val in zip(model.metrics_names, eval_results):
                    if name == 'loss':
                        continue
                    metrics_dict[f'{name}_2021'] = float(val)
                if _mae12 is not None:
                    y_pred_2021 = model.predict(x_2021, verbose=0)
                    metrics_dict['mae12_2021'] = float(_mae12(y_2021, y_pred_2021))
        except Exception as e:
            print(f"  WARNING: 2021 evaluation failed: {e}")

        # Record how many epochs actually ran (before early stopping)
        metrics_dict['epochs_trained'] = len(hist.get('loss', []))

        # Save trained model and training history if a save directory was provided
        keras_save_dir = config.get('keras_save_dir')
        if keras_save_dir is not None:
            import json as _json
            from pathlib import Path as _Path
            save_dir = _Path(keras_save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            base_name = (
                f"{config['model_type']}_{config['lead_time']}h"
                f"_cycle{config['cycle']}_{config['activation']}"
                f"_{config['num_layers']}L_{config['neurons']}N"
                f"_run{config['run_num']}"
            )
            model.save(save_dir / f"{base_name}.keras")
            with open(save_dir / f"{base_name}_history.json", 'w') as _hf:
                _json.dump(hist, _hf, default=lambda o: float(o) if hasattr(o, 'item') else str(o))
            # Persist the fitted scaler next to its .keras (single io site). No-op
            # when scaling is off (fitted_scaler is None).
            try:
                from esb.io.results import save_scaler
                scaler_path = save_scaler(save_dir, base_name, fitted_scaler)
                if scaler_path:
                    print(f"  saved scaler -> {scaler_path}")
            except Exception as e:
                print(f"  WARNING: scaler persistence failed: {e}")

        # Free model and TF session memory to prevent OOM across iterations
        del model, history, x_train, y_train
        if 'x_2021' in dir():
            del x_2021, y_2021
        if validation is not None:
            del validation
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except Exception:
            pass
        import gc
        gc.collect()

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
