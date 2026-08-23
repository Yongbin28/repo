
import sys
import os
import re
import tempfile
import shutil
import logging
import subprocess
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

import pandas as pd
import numpy as np
import joblib

# Internal imports
import ml_compute_statistic
import ml_train_model
from utils import get_tester_family_for_generic, select_decryptor, explode_and_collect_data_files, ROOT_DIR, DATASET_ROOT

logger = logging.getLogger(__name__)
BASE_DIR = ROOT_DIR
IF_FROZEN = getattr(sys, 'frozen', False)

# --- WAFERPULSE MES PHYSICS ENGINE INTEGRATION ---
FACTORY_MES_DATABASE = {
    "WAF-AUTO-001": { 
        "customer": "Automotive Controller",
        "phi": 0.85,             
        "A_constant": 2.6e-8,   
        "n_exp": 2.0,
        "base_temp_K": 398.15,   
        "baseline_J": 1500000,   
        "spc_golden_years": 15.0 
    },
    "WAF-CONS-002": { 
        "customer": "Consumer USB Drive",
        "phi": 0.75,             
        "A_constant": 2.5e-8,
        "n_exp": 2.0,
        "base_temp_K": 358.15,   
        "baseline_J": 1000000,   
        "spc_golden_years": 5.0  
    }
}

def calculate_production_mttf(measured_void_percent, wafer_id, distance_from_center_mm, probe_hits=0):
    """
    Calculates MTTF using full 300mm Wafer Physics (Edge Exclusion, Scrub Marks, Lot Drift)
    """
    recipe = FACTORY_MES_DATABASE["WAF-CONS-002"]
    wafer_id_str = str(wafer_id).upper()
    for key in FACTORY_MES_DATABASE:
        if key in wafer_id_str:
            recipe = FACTORY_MES_DATABASE[key]
            break
    else:
        if "AUTO" in wafer_id_str:
            recipe = FACTORY_MES_DATABASE["WAF-AUTO-001"]

    k = 8.617e-5  
    
    # 2. Batch Effect (Lot-to-Lot Drift)
    random.seed(hash(wafer_id))
    lot_drift_multiplier = random.gauss(1.0, 0.02) 
    drifted_A_constant = recipe["A_constant"] * lot_drift_multiplier

    # 3. Probe Needle Age & Scrub Mark Damage
    measurement_noise = 0.05 * (probe_hits / 50000.0) 
    physical_damage_offset = 5.0 
    true_void_percent = max(0.1, measured_void_percent + physical_damage_offset - measurement_noise)

    # 4. Radial Variation (Center-to-Edge Stress)
    radial_stress_multiplier = 1.0 + (0.25 * (distance_from_center_mm / 150.0)**2)
    effective_area = 1.0 - (min(true_void_percent, 99.0) / 100.0)
    J_base = (recipe["baseline_J"] * radial_stress_multiplier) / effective_area

    # Non-Linear Current Crowding
    if true_void_percent > 40.0:
        crowding_factor = math.exp((true_void_percent - 40.0) / 10.0)
        J = J_base * crowding_factor
    else:
        J = J_base

    # Thermal Runaway Feedback Loop
    joule_heating_spike_1 = (true_void_percent * 2.5) 
    T1 = recipe["base_temp_K"] + joule_heating_spike_1
    thermal_feedback = (T1 - recipe["base_temp_K"]) * 0.15 
    T_final = T1 + thermal_feedback

    # Final Black's Equation
    try:
        mttf_hours = (1/drifted_A_constant) * (J**-recipe["n_exp"]) * math.exp(recipe["phi"] / (k * T_final))
        mttf_years = round(mttf_hours / 8760.0, 2)
        # Establish a minimum floor of 1.5 years for low-yield wafers
        mttf_years = max(mttf_years, 1.5)
    except OverflowError:
        mttf_years = 999.0

    # DYNAMIC SPC BINNING
    if mttf_years >= recipe["spc_golden_years"]:
        status = f"GOLDEN ({recipe['customer']})"
    elif mttf_years >= (recipe["spc_golden_years"] / 2):
        status = f"MARGINAL (Downgrade)"
    else:
        status = "SCRAP (Critical Risk)"

    random.seed() 
    return mttf_years, status


class YieldPredictor:
    """Manages the end-to-end yield prediction pipeline from raw STDF/CSV data."""

    def __init__(self, generic: str, partname: str = "", log_func: Optional[Callable] = None):
        self.generic = generic
        self.partname = partname
        self.log_func = log_func or logger.info
        self.models: List[Path] = []
        self.model_dir: Optional[Path] = None
        self._find_models()

    def _find_models(self) -> List[Path]:
        """Discovery logic for multi-model ensembles. Prioritizes specific partname models."""
        if not DATASET_ROOT.exists():
            self.models = []
            return []
            
        candidate_models = []
        
        # 1. Family-Generic Discovery
        for family_dir in DATASET_ROOT.iterdir():
            if not family_dir.is_dir(): continue
            
            search_dirs = []
            if self.partname:
                # Prioritize generic + partname folders
                search_dirs.append(family_dir / f"{self.generic}_{self.partname}" / "Model")
                search_dirs.append(family_dir / f"{self.generic}_{self.partname}")
            # Then check generic-only folders
            search_dirs.append(family_dir / self.generic / "Model")
            search_dirs.append(family_dir / self.generic)
            
            for md in search_dirs:
                if md.exists() and md.is_dir():
                    found = []
                    if self.partname:
                        found.extend(list(md.glob(f"model_*_{self.generic}_{self.partname}.joblib")))
                    found.extend(list(md.glob(f"model_*_{self.generic}.joblib")))
                    
                    if found:
                        candidate_models.extend(found)
                        if not self.model_dir:
                            self.model_dir = md

        # 2. Deduplicate while preserving order (approximate priority)
        seen = set()
        unique_models = []
        for m in candidate_models:
            if m not in seen:
                unique_models.append(m)
                seen.add(m)
        
        # Final Priority Sort: Specific models (Generic_PartName) first
        if self.partname:
            suffix = f"_{self.generic}_{self.partname}.joblib"
            unique_models.sort(key=lambda x: suffix in x.name, reverse=True)
            
        self.models = unique_models
        return self.models

    def _decrypt(self, payload: Path, temp_dir: Path) -> Path:
        """Decrypts raw binary into CSV if necessary."""
        if payload.suffix.lower() == '.csv':
            return payload

        self.log_func(f"Decrypting binary payload: {payload.name}")
        family = get_tester_family_for_generic(self.generic)
        module = select_decryptor(family)
        if not module:
            raise RuntimeError(f"No decrypter for {family}")

        exts = getattr(module, "CANDIDATE_EXTS", (".stdf", ".std", ".gz", ".zip"))
        leaf_files, _ = explode_and_collect_data_files(payload, exts, base_temp_dir=temp_dir)
        if not leaf_files:
            raise FileNotFoundError(f"Could not extract valid data from {payload}")

        extracted, name = leaf_files[0]
        out_csv = temp_dir / f"{name}.csv"
        
        script = module.__name__ if IF_FROZEN else getattr(module, "__file__", None)
        cmd = [sys.executable, str(script), str(extracted), str(out_csv)]
        subprocess.run(cmd, check=True, capture_output=True)
        return out_csv

    def _get_historical_baseline(self) -> Dict[str, pd.Series]:
        """Finds top-yield historical lots for this generic and builds a pooled distribution baseline."""
        if not self.model_dir:
             return {}
        
        # Training data usually lives in the same root as the models (or parent)
        tp_root = None
        # Try to find T&P_Decrypted folder (usually one level up from Model folder)
        for p in [self.model_dir, self.model_dir.parent]:
            if (p / "T&P_Decrypted").exists():
                tp_root = p / "T&P_Decrypted"
                break
        
        if not tp_root:
            # Fallback scan in DATASET_ROOT
            for p in DATASET_ROOT.rglob(f"{self.generic}/T&P_Decrypted"):
                tp_root = p
                break
        
        if not tp_root:
            return {}
            
        # Scan folders for yield
        lot_folders = []
        for d in tp_root.iterdir():
            if not d.is_dir(): continue
            y = ml_compute_statistic.parse_y_from_folder(d.name)
            if y is not None:
                lot_folders.append((d, y))
        
        # Take top 5 lots by yield for a "Golden" baseline
        lot_folders.sort(key=lambda x: x[1], reverse=True)
        top_folders = [f[0] for f in lot_folders[:5]]
        
        if not top_folders:
            return {}
            
        pooled_data = {}
        for folder in top_folders:
            csvs = list(folder.glob("*.csv"))
            if not csvs: continue
            try:
                # Read first CSV as representative of the lot
                df, t_nums = ml_compute_statistic.read_csv_dynamic_header(csvs[0])
                n_map = {ml_compute_statistic.normalize_colname(c): c for c in df.columns}
                if 'bin' in n_map:
                    df = df[pd.to_numeric(df[n_map['bin']], errors='coerce') == 1]
                
                num_df, num_cols = ml_compute_statistic.safe_numeric_df(df)
                col_to_tnum = {col: t_nums[idx] for idx, col in enumerate(df.columns) if idx < len(t_nums) and t_nums[idx]}
                for c in num_cols:
                    tn = col_to_tnum.get(c, ml_compute_statistic.normalize_colname(c))
                    vals = pd.to_numeric(num_df[c], errors='coerce').dropna()
                    if tn not in pooled_data: pooled_data[tn] = []
                    pooled_data[tn].append(vals)
            except: continue
                
        final_baseline = {}
        for k, v_list in pooled_data.items():
            if v_list:
                final_baseline[k] = pd.concat(v_list, ignore_index=True)
        
        return final_baseline

    def _extract_features(self, csv_path: Path) -> Tuple[Dict[str, float], int, pd.DataFrame, Dict, Dict, pd.DataFrame]:
        """Extracts aligned ML features from the CSV."""
        df, t_nums = ml_compute_statistic.read_csv_dynamic_header(csv_path)
        col_to_tnum = {c: t_nums[i] for i, c in enumerate(df.columns) if i < len(t_nums) and t_nums[i]}
        
        full_df = df.copy()

        # Basic filtering (Bin=1, No Alarms)
        n_map = {ml_compute_statistic.normalize_colname(c): c for c in df.columns}
        if 'bin' in n_map:
            df = df[pd.to_numeric(df[n_map['bin']], errors='coerce') == 1]
        
        for k in ['fails', 'alarms']:
            if k in n_map:
                df = df[df[n_map[k]].isna() | (df[n_map[k]].astype(str).str.strip() == "")]

        pc = len(df)
        if pc == 0: raise ValueError("No passing devices found.")

        # Limit lookup
        lim_path = ml_compute_statistic.find_limit_file(csv_path)
        
        # Fallback: if csv_path is in a temp dir (e.g. uploaded file), search the
        # generic's dataset directories for a matching limit file.
        if not lim_path and DATASET_ROOT.exists():
            csv_stem = csv_path.stem
            clean_stem = csv_stem.split('.')[0]
            
            for family_dir in DATASET_ROOT.iterdir():
                if not family_dir.is_dir():
                    continue
                for generic_dir in family_dir.iterdir():
                    if generic_dir.is_dir() and (generic_dir.name == self.generic or generic_dir.name.startswith(self.generic + "_")):
                        tp_dir = generic_dir / "T&P_Decrypted"
                        if tp_dir.exists():
                            for lf in tp_dir.rglob(f"*{clean_stem}*_limits.csv"):
                                lim_path = lf
                                break
                            if not lim_path:
                                for lf in tp_dir.rglob("*_limits.csv"):
                                    lim_path = lf
                                    break
                        if lim_path: break
                if lim_path: break
        
        lim_map = ml_compute_statistic.parse_limit_file(lim_path, log_func=self.log_func) if lim_path else {}
        if lim_map:
            unique_tests = len([k for k in lim_map.keys() if not k.islower()]) or len(lim_map) // 2
            self.log_func(f"[INFO] Successfully extracted {unique_tests} specification limits.")

        # 1. Base Feature creation
        # (TP_y and pass_count are excluded from model features per manufacturing logic)
        fd = {}
        num_df, num_cols = ml_compute_statistic.safe_numeric_df(df)
        
        # 2. Historical Baseline for Stability Features (SPC/PSI)
        baseline_data = self._get_historical_baseline()
        if baseline_data:
            self.log_func(f"Loaded historical baseline for {len(baseline_data)} parameters.")
        
        for c in num_cols:
            nc = ml_compute_statistic.normalize_colname(c)
            if nc in ml_compute_statistic.META_COLS_TO_EXCLUDE or any(p in nc for p in ml_compute_statistic.TEST_NAMES_TO_EXCLUDE):
                continue
            
            st = ml_compute_statistic.col_stats(num_df[c])
            tn = col_to_tnum.get(c)
            # px is normalized for the ML model
            px = ml_compute_statistic.normalize_colname(tn) if tn else nc
            
            # Keep original T-number name for num_df used by Streamlit UI
            col_name = c  # track actual column name in num_df
            if tn:
                num_df.rename(columns={c: tn}, inplace=True)
                col_name = tn

            lsl, usl = lim_map.get(tn, (None, None)) if tn else lim_map.get(px, (None, None))
            
            fd[f"TP_{px}__mean"] = st["mean"]
            fd[f"TP_{px}__std"] = st["std"]
            fd[f"TP_{px}__median"] = st["median"]
            fd[f"TP_{px}__iqr"] = st["iqr"]
            fd[f"TP_{px}__min"] = st["min"]
            fd[f"TP_{px}__max"] = st["max"]
            fd[f"TP_{px}__missing_rate"] = st["missing_rate"]
            fd[f"TP_{px}__outlier_rate"] = st["outlier_rate"]
            for q in ml_compute_statistic.Q_LIST:
                fd[f"TP_{px}__p{int(q*100):02d}"] = st[f"p{int(q*100):02d}"]
            ml_compute_statistic.add_limit_ppm_features(fd, num_df[col_name], lsl, usl, f"TP_{px}", st["std"])

            # 3. Add Stability Features (SPC/PSI) to align with validation logic
            if baseline_data:
                # Use T-number for baseline lookup if available, else normalized name
                b_key = tn if tn and tn in baseline_data else (px if px in baseline_data else None)
                if b_key:
                    spc = ml_compute_statistic.compute_spc_features(num_df[col_name], baseline_data[b_key], f"TP_{px}")
                    psi = ml_compute_statistic.compute_psi_features(num_df[col_name], baseline_data[b_key], f"TP_{px}")
                    fd.update(spc)
                    fd.update(psi)

        return fd, pc, num_df, lim_map, col_to_tnum, full_df

    def predict(self, dlog_path: Union[str, Path]) -> Dict[str, Any]:
        """Runs the full prediction pipeline on a file."""
        if not self.models:
            return {"status": "no_models", "message": f"No models for {self.generic}"}

        p_path = Path(dlog_path)
        temp_dir = Path(tempfile.mkdtemp(prefix="yield_p_"))
        
        try:
            # 1. Prepare Data
            csv_p = self._decrypt(p_path, temp_dir)
            
            creation_date = None
            try:
                with open(csv_p, 'r', encoding='utf-8', errors='ignore') as f:
                    for _ in range(20):
                        line = f.readline()
                        if "File Creation Date:" in line:
                            creation_date = line.split("File Creation Date:")[1].strip()
                            break
            except: pass
            
            if not creation_date:
                from datetime import datetime
                creation_date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            
            # 2. Extract
            fd, pc, num_df, lim_map, c_to_t, full_df = self._extract_features(csv_p)
            

            # 3. Align with training set
            ref_path = (self.model_dir / f"merged_features_{self.generic}.csv") if self.model_dir else None
            if not ref_path or not ref_path.exists():
                ref_path = BASE_DIR / f"merged_features_{self.generic}.csv"
            
            meta_drop = ["y", "parent_folder", "lot_id", "file_name", "file_path", "TP_y", "TP_pass_count"]
            df_ref = pd.read_csv(ref_path)
            all_cols = df_ref.columns
            ft_cols_all = [c for c in all_cols if str(c).startswith("FT_")]
            exp_cols = [c for c in all_cols if c not in meta_drop and not str(c).startswith("FT_")]
            
            X = pd.DataFrame([fd])
            for c in exp_cols:
                if c not in X.columns: X[c] = np.nan
            X = X[exp_cols].copy()
            X.columns = ml_train_model.sanitize_feature_names(list(X.columns))
            
            # Predict targets schema map fallback (if y was multi-out)
            if ft_cols_all:
                y_ref = df_ref[ft_cols_all].dropna(axis=1, how='all')
                targets_schema = list(y_ref.columns)
            else:
                targets_schema = ["y"]
            
            # 4. Inference
            preds, valid = {}, []
            shap_data_dict = {}
            for m_path in self.models:
                try:
                    pipe = joblib.load(m_path)
                    # Extract model name
                    stem = m_path.stem[len("model_"):]
                    m_name = stem[:stem.find(f"_{self.generic}")] if f"_{self.generic}" in stem else stem.split("_")[0]
                    
                    # Align X to pipeline expectations
                    X_aligned = X.copy()
                    if hasattr(pipe, "feature_names_in_"):
                        X_aligned = X_aligned.reindex(columns=pipe.feature_names_in_, fill_value=np.nan)
                    
                    val_raw = pipe.predict(X_aligned)[0]
                    if isinstance(val_raw, (np.ndarray, list)) and len(targets_schema) > 1:
                        # Pack multi-out target array into dictionary mapped explicitly
                        preds[m_name] = {col: float(val_raw[i]) for i, col in enumerate(targets_schema)}
                        valid.append(float(val_raw[targets_schema.index("FT_y")] if "FT_y" in targets_schema else val_raw[0]))
                    else:
                        preds[m_name] = float(val_raw)
                        valid.append(float(val_raw))
                    
                    # Normalize predicted yield to percentage scale for mapping
                    pred_yield = float(val_raw[targets_schema.index("FT_y")] if isinstance(val_raw, (np.ndarray, list)) and "FT_y" in targets_schema else (val_raw[0] if isinstance(val_raw, (np.ndarray, list)) else val_raw))
                    pred_yield_pct = pred_yield if pred_yield > 1.0 else pred_yield * 100.0
                    pred_yield_pct = np.clip(pred_yield_pct, 0.0, 100.0)
                    measured_void_percent = 100.0 - pred_yield_pct

                    # WaferPulse Intelligence (Mock Digital Twin Integration)
                    import random
                    die_seed = int(hash(m_name) % 1000000)
                    random.seed(die_seed)
                    
                    mock_radius = random.uniform(5, 149) # Simulated die position
                    mock_probe_hits = random.randint(1, 4) # Simulated probe mark stress
                    
                    mttf_years, spc_status = calculate_production_mttf(
                        measured_void_percent, 
                        p_path.name, # Use filename as proxy for wafer_id
                        mock_radius, 
                        mock_probe_hits
                    )
                    
                    # Augment results
                    if isinstance(preds[m_name], dict):
                        preds[m_name].update({
                            "WP_Radius_mm": mock_radius,
                            "WP_Probe_Hits": mock_probe_hits,
                            "WP_MTTF_Years": mttf_years,
                            "WP_Factory_Status": spc_status
                        })
                    else:
                        # Convert to dict if it was a scalar
                        preds[m_name] = {
                            "y": preds[m_name],
                            "WP_Radius_mm": mock_radius,
                            "WP_Probe_Hits": mock_probe_hits,
                            "WP_MTTF_Years": mttf_years,
                            "WP_Factory_Status": spc_status
                        }
                    
                    # Explainability (SHAP)
                    explainer_path = m_path.parent / m_path.name.replace("model_", "explainer_")
                    if explainer_path.exists():
                        try:
                            import shap
                            explainer = joblib.load(explainer_path)
                            X_transformed = pipe[:-1].transform(X_aligned)
                            
                            shap_explanations = explainer(X_transformed)
                            
                            # Handle different formats of multi-output SHAP values
                            if isinstance(shap_explanations.values, list):
                                # List of arrays (one per output)
                                val_arr = shap_explanations.values[0][0] if len(shap_explanations.values) > 0 else np.zeros(X_aligned.shape[1])
                                b_val = shap_explanations.base_values[0][0] if isinstance(shap_explanations.base_values, list) else (shap_explanations.base_values[0] if hasattr(shap_explanations.base_values, "__len__") else shap_explanations.base_values)
                                d_arr = shap_explanations.data[0] if hasattr(shap_explanations.data, "__len__") and len(shap_explanations.data.shape) > 1 else shap_explanations.data
                            elif len(shap_explanations.values.shape) == 3:
                                # 3D array: (n_samples, n_features, n_outputs)
                                s_val = shap_explanations.values
                                if s_val.shape[2] == len(targets_schema) or s_val.shape[2] > s_val.shape[0]:
                                    val_arr = s_val[0, :, 0]
                                else:
                                    val_arr = s_val[0, 0, :]
                                b_val = shap_explanations.base_values[0, 0] if len(shap_explanations.base_values.shape) > 1 else shap_explanations.base_values[0]
                                d_arr = shap_explanations.data[0]
                            else:
                                val_arr = shap_explanations.values[0] if len(shap_explanations.values.shape) > 1 else shap_explanations.values
                                b_val = shap_explanations.base_values[0] if hasattr(shap_explanations.base_values, "__len__") else shap_explanations.base_values
                                d_arr = shap_explanations.data[0] if hasattr(shap_explanations.data, "__len__") and len(shap_explanations.data.shape) > 1 else shap_explanations.data

                            if hasattr(val_arr, "tolist"): val_arr = val_arr.tolist()
                            if hasattr(b_val, "tolist"): b_val = b_val.tolist()
                            if isinstance(b_val, (list, np.ndarray)):
                                b_val = float(b_val[0]) if len(b_val) > 0 else 0.0
                            else:
                                b_val = float(b_val)
                            if hasattr(d_arr, "tolist"): d_arr = d_arr.tolist()
                            
                            shap_data_dict[m_name] = {
                                "values": val_arr,
                                "base_values": b_val,
                                "data": d_arr,
                                "feature_names": list(X_aligned.columns)
                            }
                        except Exception as e:
                            self.log_func(f"[WARN] SHAP prediction error for {m_name}: {e}")
                            
                except Exception as e:
                    self.log_func(f"[WARN] Inference {m_path.name} failed: {e}")

            if not valid: return {"status": "error", "message": "All model predictions failed."}

            return {
                "status": "success",
                "predictions": preds,
                "shap_data": shap_data_dict,
                "average_prediction": float(np.mean(valid)),
                "curr_df": num_df,
                "full_df": full_df,
                "limit_map": lim_map,
                "col_to_tnum": c_to_t,
                "model_dir": str(self.model_dir) if self.model_dir else "",
                "creation_date": creation_date
            }


        except Exception as e:
            return {"status": "error", "message": str(e)}
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

def predict_from_dlog(generic: str, partname: str, dlog_path: str, log_func=print) -> dict:
    """Public wrapper for YieldPredictor."""
    predictor = YieldPredictor(generic, partname, log_func)
    return predictor.predict(dlog_path)

def get_models_for_generic(generic: str, partname: str = "") -> List[Path]:
    """Retrieves a list of model paths for the given generic and partname."""
    predictor = YieldPredictor(generic, partname)
    return predictor.models

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--generic", required=True)
    parser.add_argument("--partname", default="")
    parser.add_argument("--dlog", required=True)
    args = parser.parse_args()
    print(predict_from_dlog(args.generic, args.partname, args.dlog))

if __name__ == "__main__":
    main()
