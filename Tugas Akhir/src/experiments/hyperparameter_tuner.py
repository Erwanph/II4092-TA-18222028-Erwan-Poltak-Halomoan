"""
Hyperparameter tuning for BI-Rate prediction models.
Supports both Optuna (Bayesian optimization) and Grid Search with
Time Series Cross-Validation (expanding window).

All search spaces are scientifically justified in scenario_config.py.
"""

import warnings
import numpy as np
import pandas as pd
from itertools import product as itertools_product

warnings.filterwarnings("ignore")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.modeling.metrics import calculate_rmse


# Time Series Cross-validation
class TimeSeriesSplitter:
    """Validasi silang deret waktu expanding window: selalu latih pada masa
    lalu, validasi pada masa depan (tanpa shuffle)."""

    def __init__(self, n_splits=3, min_train_size=50):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def split(self, X):
        n = len(X)
        fold_size = (n - self.min_train_size) // self.n_splits

        for i in range(self.n_splits):
            train_end = self.min_train_size + i * fold_size
            val_end = min(train_end + fold_size, n)

            if train_end >= n or val_end > n:
                break

            train_idx = list(range(0, train_end))
            val_idx = list(range(train_end, val_end))

            if len(val_idx) > 0:
                yield train_idx, val_idx


# Grid Search With Ts-cv
class TimeSeriesGridSearch:
    """
    Exhaustive grid search with time series cross-validation.

    Unlike sklearn's GridSearchCV, this uses expanding-window splits
    to prevent temporal data leakage.
    """

    def __init__(self, n_splits=3, min_train_size=50, verbose=True):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.verbose = verbose
        self.results_ = []
        self.best_params_ = None
        self.best_score_ = float("inf")

    def search(self, X, y, model_class, param_grid, model_type="ml"):
        """Cari kombinasi grid terbaik dengan TSCV; kembalikan (best_params, hasil)."""
        splitter = TimeSeriesSplitter(self.n_splits, self.min_train_size)

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combos = list(itertools_product(*param_values))

        if self.verbose:
            print(f"Grid search: {len(all_combos)} combinations x {self.n_splits} folds")

        for combo in all_combos:
            params = dict(zip(param_names, combo))
            fold_scores = []

            for train_idx, val_idx in splitter.split(X):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                try:
                    score = self._evaluate_params(
                        model_class, params, X_train_cv, y_train_cv,
                        X_val_cv, y_val_cv, model_type
                    )
                    if score is not None:
                        fold_scores.append(score)
                except Exception as e:
                    if self.verbose:
                        print(f"  Error with {params}: {e}")
                    continue

            if fold_scores:
                mean_score = np.mean(fold_scores)
                self.results_.append({
                    "params": params,
                    "mean_rmse": mean_score,
                    "std_rmse": np.std(fold_scores),
                    "n_folds": len(fold_scores),
                })

                if mean_score < self.best_score_:
                    self.best_score_ = mean_score
                    self.best_params_ = params

        if self.verbose and self.best_params_:
            print(f"Best params: {self.best_params_} (RMSE={self.best_score_:.4f})")

        return self.best_params_, self.results_

    def _evaluate_params(self, model_class, params, X_train, y_train,
                         X_val, y_val, model_type):
        """Train with given params and return RMSE on validation set."""
        if model_type == "ml":
            model = model_class(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return calculate_rmse(y_val, preds)

        elif model_type == "dl":
            import gc
            import tensorflow.keras.backend as K
            from src.modeling.data_prep import create_sequences
            seq_len = params.pop("sequence_length", 12)
            n_features = X_train.shape[1]

            X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, seq_len)
            X_va_seq, y_va_seq = create_sequences(X_val, y_val, seq_len)

            if len(X_tr_seq) == 0 or len(X_va_seq) == 0:
                params["sequence_length"] = seq_len
                return None

            anc_tr = np.asarray(y_train, float)[seq_len - 1:-1]
            anc_va = np.asarray(y_val, float)[seq_len - 1:-1]
            model = model_class(
                sequence_length=seq_len,
                n_features=n_features,
                **params
            )
            model.fit(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq,
                      anchor_train=anc_tr, anchor_val=anc_va)
            preds = model.predict(X_va_seq, anchor=anc_va)
            params["sequence_length"] = seq_len

            rmse_score = calculate_rmse(y_va_seq, preds)

            del model
            K.clear_session()
            gc.collect()

            return rmse_score

        return None


# Optuna-based Optimization
class OptunaOptimizer:
    """Optimasi hyperparameter Bayesian (Optuna TPE)."""

    def __init__(self, n_trials=100, n_cv_splits=5, min_train_size=50, verbose=True):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install: pip install optuna")
        self.n_trials = n_trials
        self.n_cv_splits = n_cv_splits
        self.min_train_size = min_train_size
        self.verbose = verbose
        self.study = None
        self.all_results_ = []

    def optimize(self, X, y, model_name, search_space, model_type="ml"):
        """Jalankan optimasi; kembalikan (best_params, study)."""
        splitter = TimeSeriesSplitter(self.n_cv_splits, self.min_train_size)
        folds = list(splitter.split(X))

        def objective(trial):
            params = self._sample_params(trial, search_space, model_name)

            fold_scores = []
            for train_idx, val_idx in folds:
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]

                try:
                    score = self._evaluate_trial(
                        model_name, params, X_train_cv, y_train_cv,
                        X_val_cv, y_val_cv, model_type
                    )
                    if score is not None and not np.isnan(score):
                        fold_scores.append(score)
                except Exception:
                    continue

            if not fold_scores:
                return float("inf")

            mean_rmse = np.mean(fold_scores)
            self.all_results_.append({"params": params.copy(), "mean_rmse": mean_rmse})

            # Diagnostic: log memory usage every 10 trials
            trial_num = len(self.all_results_)
            if trial_num % 10 == 0:
                try:
                    import psutil
                    mem = psutil.Process().memory_info().rss / 1024 / 1024
                    print(f"    [DIAG] Trial {trial_num}: RSS={mem:.0f}MB")
                except ImportError:
                    pass

            return mean_rmse

        storage_path = f"sqlite:///optuna_{model_name}.db"
        self.study = optuna.create_study(
            direction="minimize", 
            study_name=model_name,
            storage=storage_path,
            load_if_exists=True
        )
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=self.verbose, gc_after_trial=True)

        best = self.study.best_params
        if self.verbose:
            print(f"Optuna best for {model_name}: {best} "
                  f"(RMSE={self.study.best_value:.4f})")

        return best, self.study

    def _sample_params(self, trial, search_space, model_name):
        """Sample parameters from Optuna trial based on search space config."""
        params = {}
        for pname, pconfig in search_space.items():
            ptype = pconfig["type"]

            if ptype == "int":
                params[pname] = trial.suggest_int(
                    pname, pconfig["low"], pconfig["high"],
                    step=pconfig.get("step", 1)
                )
            elif ptype == "int_or_none":
                use_none = trial.suggest_categorical(f"{pname}_none", [True, False])
                if use_none:
                    params[pname] = None
                else:
                    params[pname] = trial.suggest_int(pname, pconfig["low"], pconfig["high"])
            elif ptype == "float":
                params[pname] = trial.suggest_float(
                    pname, pconfig["low"], pconfig["high"]
                )
            elif ptype == "float_log":
                params[pname] = trial.suggest_float(
                    pname, pconfig["low"], pconfig["high"], log=True
                )
            elif ptype == "categorical":
                params[pname] = trial.suggest_categorical(pname, pconfig["choices"])

        return params

    def _evaluate_trial(self, model_name, params, X_train, y_train,
                        X_val, y_val, model_type):
        """Train + evaluate for a single Optuna trial."""
        if model_type == "ml":
            if model_name == "RF":
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
            elif model_name == "XGBoost":
                from xgboost import XGBRegressor
                model = XGBRegressor(random_state=42, verbosity=0, **params)
            else:
                return None

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return calculate_rmse(y_val, preds)

        elif model_type == "dl":
            import gc
            import tensorflow.keras.backend as K
            from src.modeling.data_prep import create_sequences

            seq_len = params.pop("sequence_length", 12)
            n_features = X_train.shape[1]

            X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, seq_len)
            X_va_seq, y_va_seq = create_sequences(X_val, y_val, seq_len)

            if len(X_tr_seq) < 5 or len(X_va_seq) < 3:
                params["sequence_length"] = seq_len
                return None

            # Anchor residual (sejajar create_sequences: anchor = y[seq-1:-1])
            anc_tr = np.asarray(y_train, float)[seq_len - 1:-1]
            anc_va = np.asarray(y_val, float)[seq_len - 1:-1]

            if model_name == "LSTM":
                from src.modeling.deep_learning import LSTMModel
                model = LSTMModel(
                    sequence_length=seq_len, n_features=n_features,
                    units=params.get("units", 64),
                    dropout=params.get("dropout", 0.2),
                    learning_rate=params.get("learning_rate", 0.001),
                    batch_size=params.get("batch_size", 16),
                    l1=params.get("l1", 0.0), l2=params.get("l2", 0.0),
                    epochs=50,
                )
            elif model_name == "BiLSTM":
                from src.modeling.deep_learning import BiLSTMModel
                model = BiLSTMModel(
                    sequence_length=seq_len, n_features=n_features,
                    units=params.get("units", 64),
                    dropout=params.get("dropout", 0.2),
                    learning_rate=params.get("learning_rate", 0.001),
                    batch_size=params.get("batch_size", 16),
                    l1=params.get("l1", 0.0), l2=params.get("l2", 0.0),
                    epochs=50,
                )
            else:
                params["sequence_length"] = seq_len
                return None

            model.fit(X_tr_seq, y_tr_seq, X_va_seq, y_va_seq,
                      anchor_train=anc_tr, anchor_val=anc_va)
            preds = model.predict(X_va_seq, anchor=anc_va)
            params["sequence_length"] = seq_len

            rmse_score = calculate_rmse(y_va_seq, preds)

            del model
            K.clear_session()
            gc.collect()

            return rmse_score

        return None

    def get_results_df(self):
        """Return all trial results as a DataFrame."""
        if not self.all_results_:
            return pd.DataFrame()
        rows = []
        for r in self.all_results_:
            row = r["params"].copy()
            row["mean_rmse"] = r["mean_rmse"]
            rows.append(row)
        return pd.DataFrame(rows).sort_values("mean_rmse")


# Auto Arima Tuner
class ARIMATuner:
    """Pemilihan order ARIMA otomatis via pmdarima.auto_arima (Box-Jenkins)."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.best_order = None
        self.best_seasonal_order = None
        self.fitted_model = None

    def fit(self, y_train, config):
        """Cari order ARIMA optimal; kembalikan (best_order, fitted_model)."""
        try:
            import pmdarima as pm
        except ImportError:
            raise ImportError("pmdarima required. Install: pip install pmdarima")

        seasonal = config.get("seasonal", False)
        m = config.get("m", 12) if seasonal else 1

        if self.verbose:
            print(f"auto_arima: searching (seasonal={seasonal}, m={m})...")

        model = pm.auto_arima(
            y_train,
            start_p=0, max_p=config.get("max_p", 5),
            start_q=0, max_q=config.get("max_q", 5),
            d=None, max_d=config.get("max_d", 2),
            seasonal=seasonal, m=m,
            start_P=0, max_P=config.get("max_P", 2),
            start_Q=0, max_Q=config.get("max_Q", 2),
            D=None, max_D=config.get("max_D", 1),
            information_criterion=config.get("information_criterion", "aic"),
            stepwise=config.get("stepwise", True),
            suppress_warnings=True,
            error_action="ignore",
            trace=self.verbose,
        )

        self.best_order = model.order
        self.best_seasonal_order = model.seasonal_order if seasonal else None
        self.fitted_model = model

        if self.verbose:
            print(f"Best ARIMA order: {self.best_order}")
            if seasonal:
                print(f"Best seasonal order: {self.best_seasonal_order}")
            print(f"AIC: {model.aic():.2f}")

        return self.best_order, model


# Var Lag Optimizer
class VARLagOptimizer:
    """
    VAR lag order selection using information criteria.

    Tests multiple lag orders and information criteria (AIC, BIC, HQIC, FPE)
    to find the optimal specification.
    """

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.results_ = []
        self.best_config_ = None

    def search(self, train_df, param_grid):
        """Cari kombinasi (maxlags, ic) VAR terbaik; kembalikan (best_config, hasil)."""
        from statsmodels.tsa.api import VAR

        numeric_df = train_df.select_dtypes(include=[np.number]).dropna()
        best_aic = float("inf")

        for maxlags in param_grid.get("maxlags", [12]):
            for ic in param_grid.get("ic", ["aic"]):
                try:
                    model = VAR(numeric_df)
                    fitted = model.fit(maxlags=maxlags, ic=ic)

                    result = {
                        "maxlags": maxlags,
                        "ic": ic,
                        "selected_lag": fitted.k_ar,
                        "aic": fitted.aic,
                        "bic": fitted.bic,
                        "hqic": fitted.hqic,
                        "fpe": fitted.fpe,
                    }
                    self.results_.append(result)

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        self.best_config_ = result

                    if self.verbose:
                        print(f"  VAR(maxlags={maxlags}, ic={ic}): "
                              f"lag={fitted.k_ar}, AIC={fitted.aic:.2f}")

                except Exception as e:
                    if self.verbose:
                        print(f"  VAR(maxlags={maxlags}, ic={ic}): Error - {e}")

        if self.verbose and self.best_config_:
            print(f"Best VAR config: lag={self.best_config_['selected_lag']}, "
                  f"AIC={self.best_config_['aic']:.2f}")

        return self.best_config_, self.results_


# Unified Tuner Interface
def tune_model(model_name, X, y, hyperparams_config, train_df=None):
    """Antarmuka tunggal penalaan semua model; kembalikan (best_params, hasil)."""
    method = hyperparams_config.get("tuning_method", "default")

    if model_name == "ARIMA":
        tuner = ARIMATuner()
        best_order, model = tuner.fit(y, hyperparams_config)
        return {"order": best_order}, {
            "method": "auto_arima",
            "best_order": best_order,
            "seasonal_order": tuner.best_seasonal_order,
            "aic": model.aic(),
        }

    elif model_name == "VAR" and train_df is not None:
        optimizer = VARLagOptimizer()
        best_config, all_results = optimizer.search(
            train_df, hyperparams_config.get("param_grid", {})
        )
        return best_config, {"method": "grid_search", "all_results": all_results}

    elif method == "optuna":
        if not OPTUNA_AVAILABLE:
            print("Optuna not available, falling back to grid search")
            method = "grid_search"
        else:
            model_type = "dl" if model_name in ("LSTM", "BiLSTM") else "ml"
            optimizer = OptunaOptimizer(
                n_trials=hyperparams_config.get("n_trials", 100)
            )
            best_params, study = optimizer.optimize(
                X, y, model_name,
                hyperparams_config["search_space"],
                model_type=model_type,
            )
            return best_params, {
                "method": "optuna",
                "n_trials": len(study.trials),
                "best_value": study.best_value,
                "results_df": optimizer.get_results_df(),
            }

    if method == "grid_search":
        model_type = "dl" if model_name in ("LSTM", "BiLSTM") else "ml"
        searcher = TimeSeriesGridSearch()

        if model_name == "RF":
            from sklearn.ensemble import RandomForestRegressor
            model_class = RandomForestRegressor
        elif model_name == "XGBoost":
            from xgboost import XGBRegressor
            model_class = XGBRegressor
        else:
            # For DL, grid search uses direct model classes
            if model_name == "LSTM":
                from src.modeling.deep_learning import LSTMModel
                model_class = LSTMModel
            elif model_name == "BiLSTM":
                from src.modeling.deep_learning import BiLSTMModel
                model_class = BiLSTMModel
            else:
                return {}, {"error": f"Unsupported model: {model_name}"}

        best_params, results = searcher.search(
            X, y, model_class,
            hyperparams_config.get("param_grid", {}),
            model_type=model_type,
        )
        return best_params, {"method": "grid_search", "all_results": results}

    return {}, {"error": "Unknown tuning method"}
