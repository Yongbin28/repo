
import re
import argparse
import sys
import time
import warnings
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

import numpy as np
import pandas as pd
import joblib
# SHAP is lazily loaded to avoid crashing if its dependency chain
# (e.g. cv2, numba) has version conflicts with numpy 2.x.
HAS_SHAP = False
try:
    import shap
    HAS_SHAP = True
except (ImportError, AttributeError, Exception):
    pass
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA

# Suppress library warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HAS_OPTUNA = False
try:
    import optuna
    HAS_OPTUNA = True
    optuna.logging.set_verbosity(optuna.logging.WARNING) # Suppress noisy trial logs
except ImportError:
    pass

# Optional models
HAS_XGB, HAS_CAT, HAS_LGBM = False, False, False
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError: pass
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError: pass
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError: pass

class Color:
    """Terminal colors for highlighting best models."""
    GREEN = "\033[92m"   # Best
    BLUE = "\033[94m"    # 2nd
    YELLOW = "\033[93m"  # 3rd
    RESET = "\033[0m"

# Constants
SEED = 42
TEST_SIZE = 0.2
WID_RE = re.compile(r"(W\d{6,})", re.IGNORECASE)

def parse_group_wid(row: pd.Series) -> str:
    """Extract a grouping ID (Wafer/Lot) from a row to ensure data isolation."""
    folder = str(row.get('parent_folder', '')).strip()
    if folder and folder.lower() != 'nan':
        return folder
    
    s = f"{row.get('file_path','')} {row.get('file_name','')}"
    m = WID_RE.search(s)
    return m.group(1).upper() if m else str(s).strip() or "UNKNOWN_GROUP"

def sanitize_feature_names(cols: List[str]) -> List[str]:
    """Sanitize column names for library compatibility (e.g., XGBoost)."""
    out, seen = [], {}
    for c in cols:
        s = str(c).replace("[", "(").replace("]", ")").replace("<", "_").replace(">", "_")
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^0-9a-zA-Z_().:+-]", "_", s)
        if s in seen:
            seen[s] += 1
            s = f"{s}__{seen[s]}"
        else:
            seen[s] = 0
        out.append(s)
    return out

class ModelTrainer:
    """Manages training, evaluation, and persistence of multiple regression models."""

    def __init__(self, data_csv: Path, tune: bool = False, log_func: Optional[Callable] = None):
        self.data_csv = data_csv
        self.tune = tune
        self.log_func = log_func or logger.info
        self.generic_suffix = self._derive_suffix(data_csv)
        self.results: List[Dict[str, Any]] = []
        self.top_models: List[Tuple[str, Any]] = []

    def _derive_suffix(self, path: Path) -> str:
        s = path.stem
        return s[len("merged_features_"):] if s.startswith("merged_features_") else s

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load and clean features (TP_) and target(s) (FT_)."""
        df = pd.read_csv(self.data_csv)
        groups = df.apply(parse_group_wid, axis=1)
        meta = [c for c in ["file_path", "file_name", "parent_folder"] if c in df.columns]
        
        # Partition TP_ and FT_ features if they exist
        ft_cols = [c for c in df.columns if str(c).startswith("FT_")]
        tp_cols = [c for c in df.columns if str(c).startswith("TP_") and c not in ["TP_y", "TP_pass_count"]]
        meta_exclude = meta + ft_cols + ["TP_y", "TP_pass_count"]
        
        if ft_cols:
            y = df[ft_cols].apply(pd.to_numeric, errors="coerce")
            # Require valid yield at minimum or valid rows
            valid = y.notna().any(axis=1)
            y = y.loc[valid]
            y = y.dropna(axis=1, how='all') # Drop completely empty columns
            y = y.fillna(y.median(numeric_only=True)).fillna(0.0).copy() # Fill any remaining NaNs
            X = df.drop(columns=[c for c in meta_exclude if c in df.columns]).apply(pd.to_numeric, errors="coerce")
        else:
            if "y" not in df.columns:
                raise ValueError("Input CSV must contain target column 'y' or multi-output 'FT_' columns")
            y = pd.to_numeric(df["y"], errors="coerce")
            valid = y.notna()
            y = y.loc[valid].copy()
            X = df.drop(columns=meta + ["y"]).apply(pd.to_numeric, errors="coerce")
            
        X = X.loc[valid].replace([np.inf, -np.inf], np.nan).dropna(axis=1, how='all')
        X.columns = sanitize_feature_names(list(X.columns))
        
        # Ensure y is a dataframe if multi-output
        if isinstance(y, pd.Series):
            y = y.to_frame()
            
        return X.copy(), y.copy(), groups.loc[valid].copy()

    def get_pipelines(self, n_train: int, is_multi: bool = False) -> List[Tuple[str, Pipeline]]:
        """Construct the standard model pipelines."""
        # Fix Dimensionality: We have hundreds of features but few rows. Prevent P > N overfitting!
        n_comp = min(int(n_train * 0.7), 25) if n_train > 5 else 'mle'
        
        # Avoid PyInstaller subprocess spawning crash by forcing n_jobs=1 when frozen
        n_jobs_to_use = 1 if getattr(sys, 'frozen', False) else -1
        
        tree_pre = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        scale_pre = Pipeline([
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=SEED))
        ])
        
        models = [
            ("Ridge", Pipeline([("pre", scale_pre), ("m", Ridge(alpha=10.0, random_state=SEED))])),
            ("Lasso", Pipeline([("pre", scale_pre), ("m", Lasso(alpha=0.1, random_state=SEED) if not is_multi else MultiOutputRegressor(Lasso(alpha=0.1, random_state=SEED)))])),
            ("ElasticNet", Pipeline([("pre", scale_pre), ("m", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED) if not is_multi else MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED)))])),
            ("KNN", Pipeline([("pre", scale_pre), ("m", KNeighborsRegressor(n_neighbors=min(5, n_train), weights="distance"))])),
            ("ExtraTrees", Pipeline([("pre", tree_pre), ("m", ExtraTreesRegressor(n_estimators=100, max_depth=7, random_state=SEED, n_jobs=n_jobs_to_use))])),
            ("RandomForest", Pipeline([("pre", tree_pre), ("m", RandomForestRegressor(n_estimators=100, max_depth=7, random_state=SEED, n_jobs=n_jobs_to_use))])),
            ("HistGBR", Pipeline([("pre", tree_pre), ("m", HistGradientBoostingRegressor(random_state=SEED, max_iter=100) if not is_multi else MultiOutputRegressor(HistGradientBoostingRegressor(random_state=SEED, max_iter=100)))])),
        ]

        if HAS_XGB:
            models.append(("XGBoost", Pipeline([("pre", tree_pre), ("m", XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED, n_jobs=n_jobs_to_use) if not is_multi else MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=SEED, n_jobs=n_jobs_to_use)))])))
        if HAS_LGBM:
            models.append(("LightGBM", Pipeline([("pre", tree_pre), ("m", LGBMRegressor(n_estimators=100, max_depth=4, random_state=SEED, n_jobs=n_jobs_to_use, verbosity=-1) if not is_multi else MultiOutputRegressor(LGBMRegressor(n_estimators=100, max_depth=4, random_state=SEED, n_jobs=n_jobs_to_use, verbosity=-1)))])))
        
        return models

    def get_pipeline_with_params(self, name: str, params: dict, n_train: int, is_multi: bool, n_jobs_to_use: int, tree_pre: Pipeline, scale_pre: Pipeline) -> Pipeline:
        """Construct a pipeline using specified/tuned parameters."""
        if name == "Ridge":
            return Pipeline([("pre", scale_pre), ("m", Ridge(alpha=params.get("alpha", 10.0), random_state=SEED))])
        elif name == "Lasso":
            m = Lasso(alpha=params.get("alpha", 0.1), random_state=SEED)
            return Pipeline([("pre", scale_pre), ("m", m if not is_multi else MultiOutputRegressor(m))])
        elif name == "ElasticNet":
            m = ElasticNet(alpha=params.get("alpha", 0.1), l1_ratio=params.get("l1_ratio", 0.5), random_state=SEED)
            return Pipeline([("pre", scale_pre), ("m", m if not is_multi else MultiOutputRegressor(m))])
        elif name == "KNN":
            return Pipeline([("pre", scale_pre), ("m", KNeighborsRegressor(n_neighbors=params.get("n_neighbors", min(5, n_train)), weights=params.get("weights", "distance")))])
        elif name == "ExtraTrees":
            m = ExtraTreesRegressor(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", 7), min_samples_split=params.get("min_samples_split", 2), random_state=SEED, n_jobs=n_jobs_to_use)
            return Pipeline([("pre", tree_pre), ("m", m)])
        elif name == "RandomForest":
            m = RandomForestRegressor(n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", 7), min_samples_split=params.get("min_samples_split", 2), random_state=SEED, n_jobs=n_jobs_to_use)
            return Pipeline([("pre", tree_pre), ("m", m)])
        elif name == "HistGBR":
            m = HistGradientBoostingRegressor(learning_rate=params.get("learning_rate", 0.1), max_iter=params.get("max_iter", 100), max_depth=params.get("max_depth", 7), random_state=SEED)
            return Pipeline([("pre", tree_pre), ("m", m if not is_multi else MultiOutputRegressor(m))])
        elif name == "XGBoost":
            m = XGBRegressor(learning_rate=params.get("learning_rate", 0.05), n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", 4), random_state=SEED, n_jobs=n_jobs_to_use)
            return Pipeline([("pre", tree_pre), ("m", m if not is_multi else MultiOutputRegressor(m))])
        elif name == "LightGBM":
            m = LGBMRegressor(learning_rate=params.get("learning_rate", 0.1), n_estimators=params.get("n_estimators", 100), max_depth=params.get("max_depth", 4), random_state=SEED, n_jobs=n_jobs_to_use, verbosity=-1)
            return Pipeline([("pre", tree_pre), ("m", m if not is_multi else MultiOutputRegressor(m))])
        else:
            raise ValueError(f"Unknown model name: {name}")

    def tune_pipeline(self, name: str, X_tr: pd.DataFrame, y_tr: pd.DataFrame, groups_tr: pd.Series, is_multi: bool, n_train: int) -> Pipeline:
        """Find the best hyperparameters for a given model using Optuna and return the optimized pipeline."""
        if not HAS_OPTUNA:
            self.log_func(f"[WARN] Optuna is not installed. Falling back to default parameters for {name}.")
            defaults = self.get_pipelines(n_train, is_multi)
            for d_name, d_pipe in defaults:
                if d_name == name:
                    return d_pipe
            raise ValueError(f"Unknown model name: {name}")

        self.log_func(f"[TUNING] Tuning hyperparameters for {name} using Optuna...")
        
        n_jobs_to_use = 1 if getattr(sys, 'frozen', False) else -1
        n_comp = min(int(n_train * 0.7), 25) if n_train > 5 else 'mle'
        
        tree_pre = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        scale_pre = Pipeline([
            ("imputer", SimpleImputer(strategy="median")), 
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=SEED))
        ])
        
        # Setup CV splits on the training set
        from sklearn.model_selection import GroupKFold, KFold
        unique_groups = groups_tr.nunique()
        if unique_groups >= 3:
            cv = GroupKFold(n_splits=3)
            splits = list(cv.split(X_tr, y_tr, groups=groups_tr))
        else:
            cv = KFold(n_splits=min(3, len(X_tr)), shuffle=True, random_state=SEED)
            splits = list(cv.split(X_tr, y_tr))
            
        def objective(trial):
            params = {}
            if name == "Ridge":
                params["alpha"] = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
            elif name == "Lasso":
                params["alpha"] = trial.suggest_float("alpha", 1e-4, 1e1, log=True)
            elif name == "ElasticNet":
                params["alpha"] = trial.suggest_float("alpha", 1e-4, 1e1, log=True)
                params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.1, 0.9)
            elif name == "KNN":
                params["n_neighbors"] = trial.suggest_int("n_neighbors", 2, min(15, n_train))
                params["weights"] = trial.suggest_categorical("weights", ["uniform", "distance"])
            elif name == "ExtraTrees":
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
            elif name == "RandomForest":
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 10)
                params["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
            elif name == "HistGBR":
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                params["max_iter"] = trial.suggest_int("max_iter", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
            elif name == "XGBoost":
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
            elif name == "LightGBM":
                params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                params["n_estimators"] = trial.suggest_int("n_estimators", 50, 300)
                params["max_depth"] = trial.suggest_int("max_depth", 3, 8)
            else:
                return -999.0
            
            try:
                pipe = self.get_pipeline_with_params(name, params, n_train, is_multi, n_jobs_to_use, tree_pre, scale_pre)
            except Exception:
                return -999.0
                
            scores = []
            for tr_i, val_i in splits:
                X_t, X_v = X_tr.iloc[tr_i], X_tr.iloc[val_i]
                y_t, y_v = y_tr.iloc[tr_i], y_tr.iloc[val_i]
                
                try:
                    pipe.fit(X_t, y_t.values if is_multi else y_t.values.ravel())
                    p = pipe.predict(X_v)
                    
                    y_v_np = y_v.values if is_multi else y_v.values.reshape(-1, 1)
                    p_np = p if is_multi else p.reshape(-1, 1)
                    
                    y_var = np.var(y_v_np, axis=0)
                    valid_tgt = y_var > 1e-6
                    if not np.any(valid_tgt):
                        r_sq = 0.0
                    else:
                        r_sq = r2_score(y_v_np[:, valid_tgt], p_np[:, valid_tgt], multioutput='variance_weighted')
                    scores.append(r_sq)
                except Exception:
                    scores.append(-999.0)
                    
            return np.mean(scores)
            
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, timeout=15)
        
        best_params = study.best_params
        self.log_func(f"[TUNING] Best params for {name}: {best_params} (CV R2: {study.best_value:.4f})")
        
        return self.get_pipeline_with_params(name, best_params, n_train, is_multi, n_jobs_to_use, tree_pre, scale_pre)

    def train(self, out_res: Path, out_val: Path, info: str = ""):
        """Execute training loop and save reports."""
        import time
        from runtime_calibration import RuntimeCalibrator
        total_start = time.time()
        X, y, groups = self.prepare_data()
        
        if groups.nunique() > 1:
            it = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(X, y, groups=groups)
        else:
            it = ShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED).split(X, y)
        tr_idx, val_idx = next(it)
        
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        is_multi = len(y_tr.columns) > 1
        
        # Get baseline model list to retrieve names
        model_names = [name for name, _ in self.get_pipelines(len(X_tr), is_multi=is_multi)]
        pipelines = []
        if self.tune:
            self.log_func(f"[INFO] Auto-tuning enabled. Running Optuna optimization on training split...")
            for name in model_names:
                try:
                    tuned_pipe = self.tune_pipeline(name, X_tr, y_tr, groups.iloc[tr_idx], is_multi, len(X_tr))
                    pipelines.append((name, tuned_pipe))
                except Exception as e:
                    self.log_func(f"[ERR] Tuning failed for {name}, falling back to default: {e}")
                    defaults = self.get_pipelines(len(X_tr), is_multi=is_multi)
                    for d_name, d_pipe in defaults:
                        if d_name == name:
                            pipelines.append((d_name, d_pipe))
        else:
            pipelines = self.get_pipelines(len(X_tr), is_multi=is_multi)
        
        # Save actual validation data
        val_preds = {"group": groups.iloc[val_idx].values}
        for c in y.columns:
            val_preds[f"actual_{c}"] = y_val[c].values

        self.log_func(f"[INFO] Dataset: {len(X)} rows, {X.shape[1]} features, {y.shape[1]} targets. Training {len(pipelines)} models...")

        evals = []
        for name, pipe in pipelines:
            try:
                start = time.time()
                pipe.fit(X_tr, y_tr.values if is_multi else y_tr.values.ravel())
                p = pipe.predict(X_val)
                elapsed = time.time() - start
                
                # Using variance-weighted metrics for multi-output to avoid zero-variance denominator drops
                y_val_np = y_val.values if is_multi else y_val.values.reshape(-1, 1)
                p_np = p if is_multi else p.reshape(-1, 1)
                
                # Check for zero variance
                y_var = np.var(y_val_np, axis=0)
                valid_tgt = y_var > 1e-6
                
                if not np.any(valid_tgt):
                    r_sq = 0.0
                else:
                    r_sq = r2_score(y_val_np[:, valid_tgt], p_np[:, valid_tgt], multioutput='variance_weighted')
                    
                res = {
                    "model": name, 
                    "mae": np.mean(mean_absolute_error(y_val_np, p_np, multioutput='raw_values')),
                    "rmse": np.sqrt(np.mean(mean_squared_error(y_val_np, p_np, multioutput='raw_values'))),
                    "r2": r_sq,
                    "time_seconds": elapsed
                }
                evals.append((name, pipe, res, p_np))
                
                # Store predictions for each target
                for i, c in enumerate(y.columns):
                    val_preds[f"{name}_{c}"] = p_np[:, i]
                    
            except Exception as e:
                self.log_func(f"[ERR] {name} failed: {e}")

        if not evals: return None

        # Sorting and Top 3 Selection
        evals.sort(key=lambda x: x[2]['r2'], reverse=True)
        res_df = pd.DataFrame([e[2] for e in evals])
        top_3 = evals[:3]
        
        # Persistence
        for old in self.data_csv.parent.glob(f"model_*_{self.generic_suffix}.joblib"):
            try: old.unlink()
            except: pass
        for old in self.data_csv.parent.glob(f"explainer_*_{self.generic_suffix}.joblib"):
            try: old.unlink()
            except: pass
        
        job_paths = []
        for name, pipe, _, _ in top_3:
            p = self.data_csv.parent / f"model_{name}_{self.generic_suffix}.joblib"
            joblib.dump(pipe, p)
            job_paths.append(str(p))
            
            if HAS_SHAP:
                try:
                    import json as _json
                    estimator = pipe[-1]
                    explainer = None
                    
                    # Unwrap MultiOutputRegressor to access the base estimator for type checks
                    base_estimator = estimator
                    if isinstance(estimator, MultiOutputRegressor):
                        base_estimator = estimator.estimators_[0] if hasattr(estimator, 'estimators_') else estimator.estimator
                    
                    tree_models = (RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor)
                    if HAS_XGB: tree_models += (XGBRegressor,)
                    if HAS_CAT: tree_models += (CatBoostRegressor,)
                    if HAS_LGBM: tree_models += (LGBMRegressor,)
                    
                    linear_models = (Ridge, Lasso, ElasticNet)
                    kernel_models = (SVR, KNeighborsRegressor)
                    
                    transformed_X_tr = pipe[:-1].transform(X_tr)
                    
                    if isinstance(base_estimator, tree_models):
                        # TreeExplainer: use the full estimator
                        if isinstance(estimator, MultiOutputRegressor):
                            explainer = shap.TreeExplainer(base_estimator)
                        else:
                            explainer = shap.TreeExplainer(estimator)
                    elif isinstance(base_estimator, linear_models):
                        # LinearExplainer
                        if isinstance(estimator, MultiOutputRegressor):
                            explainer = shap.LinearExplainer(base_estimator, transformed_X_tr)
                        else:
                            explainer = shap.LinearExplainer(estimator, transformed_X_tr)
                    elif isinstance(base_estimator, kernel_models):
                        # KernelExplainer
                        bg_sample = shap.sample(transformed_X_tr, 50) if len(transformed_X_tr) > 50 else transformed_X_tr
                        predict_fn = base_estimator.predict if isinstance(estimator, MultiOutputRegressor) else estimator.predict
                        explainer = shap.KernelExplainer(predict_fn, bg_sample)
                    
                    if explainer is not None:
                        expl_p = self.data_csv.parent / f"explainer_{name}_{self.generic_suffix}.joblib"
                        joblib.dump(explainer, expl_p)
                        
                        # Persist original and transformed feature names for UI recovery
                        fname_p = self.data_csv.parent / f"feature_names_{name}_{self.generic_suffix}.json"
                        
                        # Determine the names of the features AFTER transformations (e.g. PCA components)
                        try:
                            # Use scikit-learn's standard way to get names if available
                            if hasattr(pipe[:-1], "get_feature_names_out"):
                                transformed_names = list(pipe[:-1].get_feature_names_out())
                            elif hasattr(transformed_X_tr, "columns"):
                                transformed_names = list(transformed_X_tr.columns)
                            else:
                                n_feats = transformed_X_tr.shape[1]
                                # If the number of features hasn't changed (e.g. just Imputer/Scaler), use original names
                                if n_feats == len(X_tr.columns):
                                    transformed_names = list(X_tr.columns)
                                else:
                                    transformed_names = [f"Principal Factor {i+1}" for i in range(n_feats)]
                        except:
                            n_feats = transformed_X_tr.shape[1]
                            if n_feats == len(X_tr.columns):
                                transformed_names = list(X_tr.columns)
                            else:
                                transformed_names = [f"Principal Factor {i+1}" for i in range(n_feats)]
                            
                        meta = {
                            "original": list(X_tr.columns),
                            "transformed": transformed_names
                        }
                        with open(str(fname_p), "w") as f:
                            _json.dump(meta, f)
                        self.log_func(f"[OK] SHAP explainer saved for {name} (type: {type(base_estimator).__name__})")
                except Exception as e:
                    self.log_func(f"[WARN] Failed to create SHAP explainer for {name}: {e}")

        # Excel Report with Styling
        with pd.ExcelWriter(out_val, engine='openpyxl') as writer:
            val_df = pd.DataFrame(val_preds)
            res_df.to_excel(writer, index=False, sheet_name='Summary')
            
            # --- Yield Comparison Sheet ---
            y_col = "FT_y" if "FT_y" in y.columns else "y"
            if f"actual_{y_col}" in val_df.columns:
                actual_col = f"actual_{y_col}"
                # Top 3 model names (for highlighting)
                top_3_names = [e[0] for e in evals[:3]]
                
                # Order columns: group, actual, then predictions
                yield_comp_cols = ["group", actual_col] + [f"{name}_{y_col}" for name, _, _, _ in evals if f"{name}_{y_col}" in val_df.columns]
                yield_df = val_df[yield_comp_cols].copy()
                
                # Define highlighting colors
                colors = ["#90EE90", "#ADD8E6", "#FFD580"] # Light Green, Light Blue, Light Orange
                
                def style_top_models(df):
                    styles = pd.DataFrame('', index=df.index, columns=df.columns)
                    for i, name in enumerate(top_3_names):
                        pred_col = f"{name}_{y_col}"
                        if pred_col in df.columns:
                            styles[pred_col] = f'background-color: {colors[i]}'
                    return styles

                yield_df.style.apply(style_top_models, axis=None).to_excel(writer, index=False, sheet_name='Yield Comparison')
            
            # --- Validation Sheet (Detailed) ---
            val_df.to_excel(writer, index=False, sheet_name='Validation')
            
            # --- FT Parameter Comparisons ---
            if is_multi:
                for c in y.columns:
                    if c == y_col: continue 
                    comp_cols = ["group", f"actual_{c}"] + [f"{name}_{c}" for name, _, _, _ in evals if f"{name}_{c}" in val_df.columns]
                    sheet_name = f"{c[:20]} Comp"
                    val_df[comp_cols].to_excel(writer, index=False, sheet_name=sheet_name)

        total_elapsed = time.time() - total_start
        try:
            new_k = RuntimeCalibrator.log_training_run(
                num_rows=len(X),
                num_features=X.shape[1],
                num_targets=y.shape[1],
                actual_seconds=total_elapsed
            )
            self.log_func(f"[INFO] Runtime Calibration updated. New K factor: {new_k:.4e}")
        except Exception as ex:
            self.log_func(f"[WARN] Runtime Calibration failed: {ex}")
            
        self.log_func(f"[SUCCESS] Top Model: {top_3[0][0]} (R2: {top_3[0][2]['r2']:.4f})")
        return job_paths, str(out_val)

def run_model_training(data_csv: Path, out_res_path: Path, out_val_path: Path, info: str = "", tune: bool = False, log_func=print, progress_callback=None):
    """Entry point for Streamlit and CLI."""
    trainer = ModelTrainer(data_csv, tune=tune, log_func=log_func)
    return trainer.train(out_res_path, out_val_path, info)

def main():
    parser = argparse.ArgumentParser(description="Professional Model Training Utility")
    parser.add_argument("--data", required=True, help="Path to features CSV")
    parser.add_argument("--info", default="", help="Metadata for summary")
    parser.add_argument("--tune", action="store_true", help="Perform hyperparameter tuning with Optuna")
    args = parser.parse_args()
    
    p = Path(args.data)
    run_model_training(
        p, 
        p.parent / f"res_{p.stem}.csv", 
        p.parent / f"details_{p.stem}.xlsx", 
        info=args.info,
        tune=args.tune
    )

if __name__ == "__main__":
    main()