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
    
    def __init__(self, output_dir: Path, max_workers: int = 4, keras_save_dir: Path = None, verbose: int = 0, shard: tuple = None):
        super().__init__("MAPE", output_dir, max_workers, keras_save_dir=keras_save_dir, verbose=verbose, shard=shard)
        
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

        # In-graph scaling: when the scale stage fitted a scaler, its statistics
        # become a FROZEN Normalization layer here instead of being applied to the
        # arrays beforehand. The model then consumes RAW inputs and standardizes
        # them itself, so the .keras is self-contained (no _scaler.joblib for the
        # consumer to locate, pair and apply correctly).
        #
        # The layer is non-trainable and deterministic, so the forward pass AND
        # the gradients are identical to pre-transforming the arrays -- this is a
        # relocation of the transform, not a change to the training math.
        norm_mean = config.get('norm_mean')
        norm_variance = config.get('norm_variance')
        if (norm_mean is None) != (norm_variance is None):
            raise ValueError(
                'norm_mean and norm_variance must be provided together '
                f'(got mean={norm_mean is None and "None" or "set"}, '
                f'variance={norm_variance is None and "None" or "set"})'
            )
        if norm_mean is not None:
            import numpy as _np
            from tensorflow.keras.layers import Normalization
            mean_arr = _np.asarray(norm_mean, dtype='float32')
            var_arr = _np.asarray(norm_variance, dtype='float32')
            # Loud guard: a scaler fitted on a different feature count than the
            # model's input would otherwise broadcast or fail deep inside Keras.
            if mean_arr.shape[-1] != input_shape[-1]:
                raise ValueError(
                    f'scaler width {mean_arr.shape[-1]} != model input width '
                    f'{input_shape[-1]} -- the scaler does not belong to this model'
                )
            model.add(Normalization(axis=-1, mean=mean_arr, variance=var_arr))

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

        # NOTE: the model is NOT built here any more. It is built ONCE further
        # down, after the scale stage has fitted the scaler, because the frozen
        # Normalization layer needs its mean/variance at construction time.
        # (This previously built here and then rebuilt below, discarding the
        # first model.)

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
        # OFF by default -> nothing happens, byte-identical to the unscaled path
        # (the Stage B golden digest d80ae54 stays valid).
        #
        # ON -> fit a StandardScaler on x_train ONLY (leakage-checked inside
        # fit_scaler) and hand its statistics to the model as a frozen
        # Normalization layer. The ARRAYS ARE LEFT RAW on purpose: the model now
        # performs the transform in-graph, so the saved .keras is self-contained
        # and the consumer never applies a scaler by hand.
        #
        # Consequence to be aware of: the fit inputs are now raw rather than
        # standardized, so the scaled-path golden digest legitimately changes and
        # must be re-frozen deliberately.
        fitted_scaler = None
        if config.get('scale', False):
            from esb.contracts import Arrays
            from esb.stages.scale import fit_scaler, normalization_stats
            if len(data) < 6:
                raise RuntimeError(
                    'scale=True requires preparingData to return the full 6 arrays'
                )
            arrays = Arrays.from_preparing_data(data)
            fitted_scaler = fit_scaler(arrays.x_train)
            mean, variance = normalization_stats(fitted_scaler)
            # Consumed by _build_model below to construct the frozen layer.
            # Stored as plain lists, not ndarrays: config dicts get JSON-dumped
            # into provenance/progress records and numpy arrays are not
            # serializable.
            config['norm_mean'] = mean.tolist()
            config['norm_variance'] = variance.tolist()

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

        # Build the model ONCE, here: input width is known from the real arrays
        # and (when scaling is on) the Normalization statistics are in config.
        # A missing input_shape used to fall through silently and reuse a model
        # built from the preview -- which, now that scaling lives in the graph,
        # would produce an UNSCALED model that still trains and still saves.
        # Fail loudly instead.
        if input_shape is None:
            raise RuntimeError(
                'could not determine input_shape from x_train; refusing to build '
                'a model (a silently unscaled model would look identical)'
            )
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

        # Full metric suite (design §9): keras history already gives val_mae /
        # val_mape (and mae / mape); we add the below-12 and mean-error metrics so
        # the whole suite (mae, mae12, me, me12, mape) is persisted per config/rep
        # for both the validation set and the 2021 independent year. Selection
        # still uses val_mae; these are for later inspection in the viz suite.
        try:
            from src.helper.metrics import mae12 as _mae12, me as _me, me12 as _me12
        except Exception:
            try:
                from helper.metrics import mae12 as _mae12, me as _me, me12 as _me12
            except Exception:
                _mae12 = _me = _me12 = None

        def _suite(y_true, y_pred):
            """{mae12, me, me12} on flattened 1D arrays (avoids the shape-broadcast
            pitfalls in the metric fns when y_pred is (N,1) and y_true is (N,))."""
            import numpy as _np
            yt = _np.asarray(y_true).reshape(-1)
            yp = _np.asarray(y_pred).reshape(-1)
            out = {}
            if _mae12 is not None:
                out['mae12'] = float(_mae12(yt, yp))
            if _me is not None:
                out['me'] = float(_me(yt, yp))
            if _me12 is not None:
                out['me12'] = float(_me12(yt, yp))
            return out

        # Post-hoc suite on the validation set (val_mae/val_mape come from history)
        if validation is not None and _mae12 is not None:
            try:
                x_val_arr, y_val_arr = validation
                y_pred_val = model.predict(x_val_arr, verbose=0)
                for name, val in _suite(y_val_arr, y_pred_val).items():
                    metrics_dict[f'val_{name}'] = val
            except Exception as e:
                print(f"  WARNING: val metric suite computation failed: {e}")

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
                # NO scaler here, deliberately. The model now carries a frozen
                # Normalization layer and scales its own inputs, so the
                # independent year must arrive RAW — exactly like training did.
                # Passing the scaler as well would double-scale it, silently.
            )
            if x_2021.shape[0] > 0:
                eval_results = model.evaluate(x_2021, y_2021, verbose=0)
                for name, val in zip(model.metrics_names, eval_results):
                    if name == 'loss':
                        continue
                    metrics_dict[f'{name}_2021'] = float(val)
                # Add the below-12 / mean-error suite on the 2021 year (mae_2021 /
                # mape_2021 already came from model.evaluate above).
                if _mae12 is not None:
                    y_pred_2021 = model.predict(x_2021, verbose=0)
                    for name, val in _suite(y_2021, y_pred_2021).items():
                        metrics_dict[f'{name}_2021'] = val
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
            from tuner_utils import job_base_name
            base_name = job_base_name(config)
            model.save(save_dir / f"{base_name}.keras")
            # Metadata sidecar - the artifact that REPLACES _scaler.joblib. The
            # scaler's numbers now live in the model; what still has to ship
            # beside it is the input column ORDER, which the model cannot check.
            # The training history goes IN here too (no more _history.json), so
            # one job is one .keras plus one sidecar.
            #
            # This block does NOT swallow errors. A .keras whose column order was
            # never recorded is an unusable artifact that looks fine on disk, so
            # a metadata failure must fail the job loudly rather than leave a
            # silently-unusable model behind.
            from esb.io.results import save_model_metadata
            from esb.stages.features import feature_names as _feature_names
            try:
                _names = _feature_names(
                    input_structure=config.get('input_structure', 'descending'),
                    input_hours_forecast=config.get(
                        'input_hours_forecast', config.get('lead_time')),
                    atp_hours_back=config.get('atp_hours_back', 24),
                    wtp_hours_back=config.get('wtp_hours_back', 24),
                    pred_atp_interval=config.get('pred_atp_interval', 1),
                )
            except NotImplementedError as e:
                # Non-descending structure: ship metadata without a name list
                # rather than a guessed (silently wrong) one.
                print(f"  WARNING: no feature-name list written: {e}")
                _names = None
            # Width cross-check: the derived contract must agree with the
            # array the model was actually built from. Disagreement means
            # feature engineering and config have drifted apart.
            if _names is not None and len(_names) != int(input_shape[-1]):
                raise RuntimeError(
                    f'feature-name contract width {len(_names)} != model input '
                    f'width {int(input_shape[-1])} - config and feature '
                    f'engineering disagree'
                )
            # Coerce numpy scalars the same way the old _history.json dump did,
            # since json.dump inside save_model_metadata has no default= hook.
            _hist = _json.loads(_json.dumps(
                hist,
                default=lambda o: float(o) if hasattr(o, 'item') else str(o)))
            meta_path = save_model_metadata(
                save_dir, base_name,
                feature_names=_names,
                config=config,
                scaling='in_graph' if fitted_scaler is not None else 'none',
                history=_hist,
            )
            print(f"  saved metadata -> {meta_path}")

            # NOTE: no _scaler.joblib is written any more. The scaler's numbers
            # live inside the .keras as the frozen Normalization layer, and they
            # remain recoverable from it if ever needed:
            #     m, v = model.get_layer(index=0).get_weights()
            #     mean, scale = m, np.sqrt(v)
            # so nothing is lost by not persisting the sklearn object.

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
