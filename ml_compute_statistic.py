"""
Batch CSV -> 1-row features per CSV (all numeric columns) + y parsed from folder name,
then train/validate ML regression models.

Requirements:
    pip install pandas numpy scikit-learn xgboost catboost lightgbm

Features:
    - Root folder scans for subfolders containing Yield percentages in their names.
    - Robust statistical feature extraction (mean, std, median, IQR, outliers, etc.).
    - Automatic limit file discovery and PPM feature calculation.
    - Weighted aggregation of wafer-level features to lot-level.
"""

from __future__ import annotations

import os
import re
import argparse
import subprocess
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Config & Consts
# ----------------------------
HEADER_ROW_1_INDEXED = 6
SKIPROWS = HEADER_ROW_1_INDEXED - 1
Q_LIST = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
OUTLIER_K = 3.0
NEAR_MARGIN_RATIO = 0.05
NEAR_SIGMA_K = 0.5

DEFAULT_ROOT = str(Path.home() / "Documents" / "T&P_Wafer_Quality_Gate" / "dataset" / "T&P_Decrypted")
DEFAULT_OUT = "merged_features.csv"

META_COLS_TO_EXCLUDE = {
    "device_", "bin", "site", "x", "y", "fails", "alarms", "device_bin_site", "test_number"
}
TEST_NAMES_TO_EXCLUDE = {
    "x_die_location", "y_die_location", "timeline", "uph", 
    "test_time_prior_", "index_time", "down_time_total_", "x_die", "y_die", "X_DIE", "Y_DIE", "X DIE", "Y DIE"
}

# ----------------------------
# Utilities
# ----------------------------
def normalize_colname(c: str) -> str:
    """Normalize column names for consistent lookup."""
    return re.sub(r'[^a-z0-9_]+', '_', str(c).strip().lower())

def parse_y_from_folder(folder_name: str) -> Optional[float]:
    """Parse yield percentage from folder names (e.g., W1234_[01]_95.5%)."""
    m = re.search(r'(\d+(?:\.\d+)?)\s*%', folder_name)
    return float(m.group(1)) if m else None

def extract_lot_id(folder_name: str) -> str:
    """Extract Lot ID from folder name (e.g., HB025401_[13]_97.9% -> HB025401)."""
    return folder_name.split('_', 1)[0]

def natural_sort_key(s: str) -> List[int | str]:
    """Sort key for strings with embedded numbers."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def render_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Returns a visual progress bar string."""
    if total <= 0: return ""
    percent = (current / total)
    filled = int(width * percent)
    bar = "#" * filled + "-" * (width - filled)
    return f"|{bar}| {current}/{total} ({percent * 100:.1f}%)"

def safe_numeric_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Identify and convert columns to numeric, returning numeric-only DataFrame."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == object:
            try:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='ignore')
            except (ValueError, TypeError):
                pass
    numeric_cols = [c for c in df_copy.columns if pd.api.types.is_numeric_dtype(df_copy[c])]
    return df_copy[numeric_cols], numeric_cols

def col_stats(s: pd.Series) -> Dict[str, float]:
    """Compute manufacturing statistics for a numeric series."""
    total = len(s)
    sd = s.dropna()
    non_missing = len(sd)
    missing_rate = 1.0 - (non_missing / total) if total > 0 else np.nan

    if non_missing == 0:
        stats = {"mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan,
                 "min": np.nan, "max": np.nan, "missing_rate": float(missing_rate), "outlier_rate": np.nan}
        for q in Q_LIST: stats[f"p{int(q*100):02d}"] = np.nan
        return stats

    q25, q50, q75 = float(sd.quantile(0.25)), float(sd.quantile(0.50)), float(sd.quantile(0.75))
    iqr = q75 - q25
    outlier_rate = 0.0
    if iqr > 0:
        low, high = q50 - OUTLIER_K * iqr, q50 + OUTLIER_K * iqr
        outlier_rate = float(((sd < low) | (sd > high)).mean())

    stats = {
        "mean": float(sd.mean()),
        "std": float(sd.std(ddof=1)) if non_missing > 1 else 0.0,
        "median": q50,
        "iqr": float(iqr),
        "min": float(sd.min()),
        "max": float(sd.max()),
        "missing_rate": float(missing_rate),
        "outlier_rate": outlier_rate,
    }
    for q in Q_LIST:
        stats[f"p{int(q*100):02d}"] = float(sd.quantile(q))
    return stats

def add_limit_ppm_features(feat: Dict[str, float], values: pd.Series, lsl: Optional[float], 
                           usl: Optional[float], prefix: str, sigma: float = 0.0) -> None:
    """Add proximity-to-limits PPM style features."""
    sd = values.dropna()
    feat[f"{prefix}__ppm_near_LSL_pct"] = np.nan
    feat[f"{prefix}__ppm_near_USL_pct"] = np.nan
    feat[f"{prefix}__ppm_near_LSL_sigma"] = np.nan
    feat[f"{prefix}__ppm_near_USL_sigma"] = np.nan

    if len(sd) == 0 or lsl is None or usl is None or not np.isfinite(lsl) or not np.isfinite(usl) or usl <= lsl:
        return

    rng = usl - lsl
    margin_pct = NEAR_MARGIN_RATIO * rng
    feat[f"{prefix}__ppm_near_LSL_pct"] = float(((sd >= lsl) & (sd <= lsl + margin_pct)).mean())
    feat[f"{prefix}__ppm_near_USL_pct"] = float(((sd <= usl) & (sd >= usl - margin_pct)).mean())

    if sigma > 1e-9:
        margin_sig = NEAR_SIGMA_K * sigma
        feat[f"{prefix}__ppm_near_LSL_sigma"] = float(((sd >= lsl) & (sd <= lsl + margin_sig)).mean())
        feat[f"{prefix}__ppm_near_USL_sigma"] = float(((sd <= usl) & (sd >= usl - margin_sig)).mean())


def compute_spc_features(curr_vals: pd.Series | np.ndarray, baseline_vals: pd.Series | np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute standard three-sigma Shewhart SPC features.

    The historical baseline defines CL = mean and UCL/LCL = mean +/- 3 sigma.
    The current lot is shifted when its mean falls outside these control limits.
    """
    out: Dict[str, float] = {
        f"{prefix}__spc__cl": np.nan,
        f"{prefix}__spc__ucl": np.nan,
        f"{prefix}__spc__lcl": np.nan,
        f"{prefix}__spc__current_mean": np.nan,
        f"{prefix}__spc__mean_shift": np.nan,
        f"{prefix}__spc__sigma_dist": np.nan,
        f"{prefix}__spc__is_shift": np.nan,
    }

    if isinstance(curr_vals, pd.Series):
        curr = curr_vals.dropna().to_numpy()
    else:
        curr = curr_vals[~np.isnan(curr_vals)]

    if isinstance(baseline_vals, pd.Series):
        base = baseline_vals.dropna().to_numpy()
    else:
        base = baseline_vals[~np.isnan(baseline_vals)]

    if len(curr) == 0 or len(base) < 5:
        return out

    baseline_mean = float(np.mean(base))
    baseline_std = float(np.std(base, ddof=1)) if len(base) > 1 else 0.0
    current_mean = float(np.mean(curr))

    out[f"{prefix}__spc__cl"] = baseline_mean
    out[f"{prefix}__spc__current_mean"] = current_mean

    if baseline_std > 1e-12:
        ucl = baseline_mean + 3.0 * baseline_std
        lcl = baseline_mean - 3.0 * baseline_std
        mean_shift = abs(current_mean - baseline_mean) / baseline_std

        out[f"{prefix}__spc__ucl"] = ucl
        out[f"{prefix}__spc__lcl"] = lcl
        out[f"{prefix}__spc__mean_shift"] = mean_shift
        out[f"{prefix}__spc__sigma_dist"] = mean_shift
        out[f"{prefix}__spc__is_shift"] = (
            1.0 if current_mean > ucl or current_mean < lcl else 0.0
        )

    return out

def compute_psi_features(curr_vals: pd.Series | np.ndarray, baseline_vals: pd.Series | np.ndarray, prefix: str, bins: int = 10) -> Dict[str, float]:
    """Compute PSI-based distribution stability features for a single test column.

    Returns three features:
      - ``<prefix>__psi__score``:    raw PSI value
      - ``<prefix>__psi__is_minor``: 1.0 if PSI >= 0.10
      - ``<prefix>__psi__is_major``: 1.0 if PSI >  0.25
    """
    out: Dict[str, float] = {
        f"{prefix}__psi__score":    np.nan,
        f"{prefix}__psi__is_minor": np.nan,
        f"{prefix}__psi__is_major": np.nan,
    }
    
    if isinstance(curr_vals, pd.Series):
        curr = curr_vals.dropna().to_numpy()
    else:
        curr = curr_vals[~np.isnan(curr_vals)]
        
    if isinstance(baseline_vals, pd.Series):
        base = baseline_vals.dropna().to_numpy()
    else:
        base = baseline_vals[~np.isnan(baseline_vals)]

    if len(curr) == 0 or len(base) < 5:
        return out

    min_v = min(float(base.min()), float(curr.min()))
    max_v = max(float(base.max()), float(curr.max()))
    if min_v == max_v:
        out[f"{prefix}__psi__score"]    = 0.0
        out[f"{prefix}__psi__is_minor"] = 0.0
        out[f"{prefix}__psi__is_major"] = 0.0
        return out

    edges = np.linspace(min_v, max_v, bins + 1)
    base_cnt, _ = np.histogram(base, bins=edges)
    curr_cnt, _ = np.histogram(curr, bins=edges)

    base_pct = base_cnt / len(base)
    curr_pct = curr_cnt / len(curr)
    base_pct = np.where(base_pct == 0, 1e-4, base_pct)
    curr_pct = np.where(curr_pct == 0, 1e-4, curr_pct)

    psi = float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))
    out[f"{prefix}__psi__score"]    = psi
    out[f"{prefix}__psi__is_minor"] = 1.0 if psi >= 0.10 else 0.0
    out[f"{prefix}__psi__is_major"] = 1.0 if psi >  0.25 else 0.0
    return out

def read_csv_dynamic_header(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Read CSV with dynamic header and T-number extraction."""
    h_idx, t_nums = -1, []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(30)]
        for i, line in enumerate(lines):
            if "device" in line.lower() and "bin" in line.lower():
                h_idx = i
                break
        if h_idx >= 2:
            t_nums = [x.strip() for x in lines[h_idx-2].split(',')]
    except: pass
    try:
        df = pd.read_csv(path, skiprows=max(0, h_idx), header=0, engine="python", on_bad_lines="skip", encoding="utf-8")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, skiprows=max(0, h_idx), header=0, engine="python", on_bad_lines="skip", encoding="cp1252")
        except UnicodeDecodeError:
            df = pd.read_csv(path, skiprows=max(0, h_idx), header=0, engine="python", on_bad_lines="skip", encoding="latin-1")
    return df, t_nums

def find_limit_file(csv_path: Path, context_root: Optional[Path] = None) -> Optional[Path]:
    """Locate appropriate limit file for single CSV context."""
    # Local dir check
    lim_dir = csv_path.parent / "limit"
    if lim_dir.is_dir():
        target = lim_dir / f"{csv_path.stem}_limits.csv"
        if target.exists(): return target
        for f in lim_dir.glob("*_limits.csv"): return f

    # Context root check
    if context_root:
        g_lim = context_root.parent / "limit"
        if g_lim.is_dir():
            for f in g_lim.glob("*_limits.csv"):
                if csv_path.stem in f.name: return f
        
        # Search the root directory for any limit files as a fallback
        for f in context_root.parent.glob("*_limits.csv"):
            return f
            
    # Final fallback: search parent of csv_path's parent (often the generic root)
    generic_root = csv_path.parent.parent.parent
    if generic_root.is_dir():
        for f in generic_root.glob("*_limits.csv"):
            return f
            
    return None

def parse_limit_file(path: Path, log_func=None) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Parse limits CSV into mapping of Test# -> (Min, Max).
    
    Keys are stored in original format (e.g. 'T1.0') and also indexed by
    their normalized form (e.g. 't1_0') so lookups work with either format.
    """
    limits = {}
    try:
        try:
            ldf = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                ldf = pd.read_csv(path, encoding="cp1252")
            except UnicodeDecodeError:
                ldf = pd.read_csv(path, encoding="latin-1")
        ldf.columns = [normalize_colname(c) for c in ldf.columns]
        t_col = next((c for c in ["test_number", "test_num"] if c in ldf.columns), ldf.columns[0])
        min_col = next((c for c in ["stdf_min", "min_limit"] if c in ldf.columns), None)
        max_col = next((c for c in ["stdf_max", "max_limit"] if c in ldf.columns), None)
        if min_col and max_col:
            for _, row in ldf.iterrows():
                try:
                    raw_key = str(row[t_col]).strip()
                    val = (float(row[min_col]) if not pd.isna(row[min_col]) else None,
                           float(row[max_col]) if not pd.isna(row[max_col]) else None)
                    # Store under original format (e.g. 'T1.0')
                    limits[raw_key] = val
                    # Also store with/without 'T' prefix for robustness
                    if raw_key.startswith('T'):
                        limits[raw_key[1:]] = val
                    elif raw_key[0].isdigit():
                        limits['T' + raw_key] = val
                        
                    # Also store under normalized key (e.g. 't1_0') for backward compat
                    norm_key = normalize_colname(raw_key)
                    if norm_key != raw_key:
                        limits[norm_key] = val
                except: pass
    except: pass
    return limits

# ----------------------------
# Core Processing Class
# ----------------------------
class FeatureExtractor:
    """Orchestrates the feature extraction pipeline from multiple CSV folders."""
    
    def __init__(self, root: Path, out_path: Path, yield_data_path: Optional[Path] = None, 
                 log_func: Callable = print, progress_callback: Optional[Callable] = None):
        self.root = root
        self.out_path = out_path
        self.yield_data_path = yield_data_path
        self.log_func = log_func
        self.progress_callback = progress_callback
        
        self.limit_file_cache: Dict[Path, Optional[Path]] = {}
        self.limit_data_cache: Dict[Path, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
        self.yield_mapping: Dict[str, float] = self._load_yield_mapping()
        self.context_root: Optional[Path] = self._resolve_context_root()

    def _resolve_context_root(self) -> Optional[Path]:
        """Ascend from root to find the 'T&P_Decrypted' base folder for global context."""
        curr = self.root
        while curr.parent != curr:
            if "T&P_Decrypted" in curr.name: return curr
            curr = curr.parent
        return None

    def _load_yield_mapping(self) -> Dict[str, float]:
        """Load external lot-to-yield mapping from Excel/CSV file."""
        mapping = {}
        if not self.yield_data_path or not self.yield_data_path.exists():
            return mapping
        try:
            df = pd.read_excel(self.yield_data_path) if self.yield_data_path.suffix == '.xlsx' else pd.read_csv(self.yield_data_path)
            lot_col = next((c for c in ['LotId', 'FabLotId', 'Lot_ID'] if c in df.columns), None)
            yld_col = next((c for c in ['YieldActual', 'CYActual', 'Yield'] if c in df.columns), None)
            if lot_col and yld_col:
                for _, row in df.iterrows():
                    l_val = str(row[lot_col]).strip().split('.')[0]
                    try:
                        y_val = float(row[yld_col])
                        if not pd.isna(y_val): mapping[l_val] = y_val
                    except: pass
            self.log_func(f"[INFO] Loaded {len(mapping)} yield records from {self.yield_data_path.name}")
        except Exception as e:
            self.log_func(f"[WARN] Failed to load mapping: {e}")
        return mapping

    def find_limit_file(self, csv_path: Path) -> Optional[Path]:
        """Locate appropriate limit file (class wrapper with caching)."""
        if csv_path.parent in self.limit_file_cache:
            return self.limit_file_cache[csv_path.parent]
        res = find_limit_file(csv_path, self.context_root)
        self.limit_file_cache[csv_path.parent] = res
        return res

    def parse_limit_file(self, path: Path) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
        """Parse limits CSV (class wrapper with caching).col_stats"""
        if path in self.limit_data_cache: return self.limit_data_cache[path]
        res = parse_limit_file(path)
        self.limit_data_cache[path] = res
        return res

    def read_csv_stdf(self, path: Path) -> Tuple[pd.DataFrame, List[str]]:
        """Read CSV with dynamic header (class wrapper for standalone utility)."""
        return read_csv_dynamic_header(path)

    def featurize_csv(self, path: Path, y: Optional[float]) -> Tuple[Dict[str, float], int, Dict[str, pd.Series]]:
        """Extract all stats features from a single CSV.

        Returns a tuple of (feature_dict, pass_count, raw_series_map) where
        raw_series_map maps normalized test prefix -> raw numeric Series.  The
        raw series are retained so the caller can build cross-lot SPC / PSI
        baselines in a second pass without re-reading the file.
        """
        df, t_nums = self.read_csv_stdf(path)
        if df.empty: return {}, 0, {}
        
        c_to_t = {c: t_nums[i] for i, c in enumerate(df.columns) if i < len(t_nums) and t_nums[i]}
        norm_map = {normalize_colname(c): c for c in df.columns}
        
        # Site filtration
        if 'bin' in norm_map:
            df = df[pd.to_numeric(df[norm_map['bin']], errors='coerce') == 1]
        for k in ['fails', 'alarms']:
            if k in norm_map:
                df = df[df[norm_map[k]].isna() | (df[norm_map[k]].astype(str).str.strip() == "")]
        
        pc = len(df)
        if pc == 0: return {}, 0, {}

        lim_map = {}
        lp = self.find_limit_file(path)
        if lp: lim_map = self.parse_limit_file(lp)

        feat = {"y": y if y is not None else np.nan, "pass_count": float(pc)}
        num_df, num_cols = safe_numeric_df(df)
        raw_series: Dict[str, pd.Series] = {}
        
        for c in num_cols:
            nc = normalize_colname(c)
            if nc in META_COLS_TO_EXCLUDE or any(px in nc for px in TEST_NAMES_TO_EXCLUDE):
                continue
            
            st = col_stats(num_df[c])
            tn = c_to_t.get(c)
            px = normalize_colname(tn) if tn else nc
            lsl, usl = lim_map.get(tn, (None, None)) if tn else lim_map.get(px, (None, None))
            
            feat.update({f"{px}__mean": st["mean"], f"{px}__std": st["std"], 
                         f"{px}__median": st["median"], f"{px}__iqr": st["iqr"],
                         f"{px}__min": st["min"], f"{px}__max": st["max"],
                         f"{px}__missing_rate": st["missing_rate"], f"{px}__outlier_rate": st["outlier_rate"]})
            for q in Q_LIST: feat[f"{px}__p{int(q*100):02d}"] = st[f"p{int(q*100):02d}"]
            add_limit_ppm_features(feat, num_df[c], lsl, usl, px, st["std"])
            raw_series[px] = num_df[c].dropna().reset_index(drop=True)

        return feat, pc, raw_series

    def extract(self, save: bool = True) -> Optional[pd.DataFrame]:
        """Full extraction and lot-level aggregation run.

        Pass 1 鈥?featurize every wafer CSV and collect raw per-test Series.
        Pass 2 鈥?for each lot, build a leave-one-out baseline from all other
                  lots, compute SPC / PSI stability features, and merge them
                  into that lot's feature row before final aggregation.
        """
        all_csvs = [f for f in self.root.rglob("*.csv") 
                    if "%" in f.parent.name and not f.name.endswith("_limits.csv") 
                    and f.stat().st_size > 10240 and f.resolve() != self.out_path.resolve()]
        if not all_csvs: return None

        # lot_data: folder_name -> list of (feat_dict, pass_count, raw_series_map)
        lot_data: Dict[str, List[Tuple[Dict, int, Dict[str, pd.Series]]]] = {}
        total = len(all_csvs)
        self.log_func(f"[INFO] Processing {total} wafers...")

        for i, cp in enumerate(all_csvs, 1):
            if self.progress_callback: self.progress_callback(i/total, i, total)
            else: sys.stdout.write(f"\r{render_progress_bar(i, total)} {cp.name[:30]}")

            try:
                fn = cp.parent.name
                lot = extract_lot_id(fn)
                y = self.yield_mapping.get(lot) or parse_y_from_folder(fn)
                f, pc, raw = self.featurize_csv(cp, y)
                if pc > 0: lot_data.setdefault(fn, []).append((f, pc, raw))
            except: pass
        if not self.progress_callback: print()

        # --- Pass 2: Compute Historical SPC / PSI Features ---
        n_lots = len(lot_data)
        MIN_BASELINE_LOTS = 5  # Require enough context for meaningful statistics.
        if n_lots >= MIN_BASELINE_LOTS:
            self.log_func(f"[INFO] Computing historical SPC/PSI features across {n_lots} lots...")
            lot_names = list(lot_data.keys())

            # Build per-lot aggregated raw series for quick baseline construction.
            # lot_pooled: folder_name -> {prefix: concatenated numpy array from all wafers in lot}
            lot_pooled: Dict[str, Dict[str, np.ndarray]] = {}
            for fn, wafers in lot_data.items():
                pooled_lists: Dict[str, List[pd.Series]] = {}
                for _, _, raw in wafers:
                    for px, s in raw.items():
                        pooled_lists.setdefault(px, []).append(s)
                
                pooled: Dict[str, np.ndarray] = {}
                for px, s_list in pooled_lists.items():
                    if len(s_list) == 1:
                        pooled[px] = s_list[0].to_numpy()
                    else:
                        pooled[px] = pd.concat(s_list, ignore_index=True).to_numpy()
                lot_pooled[fn] = pooled

            # All test prefixes that appear in at least half the lots.
            all_prefixes: Dict[str, int] = {}
            for pooled in lot_pooled.values():
                for px in pooled:
                    all_prefixes[px] = all_prefixes.get(px, 0) + 1
            min_presence = max(2, n_lots // 2)
            common_prefixes = {px for px, cnt in all_prefixes.items() if cnt >= min_presence}

            for fn, wafers in lot_data.items():
                # Baseline = all other lots concatenated.
                baseline_pooled: Dict[str, np.ndarray] = {}
                other_lots = [other_fn for other_fn in lot_names if other_fn != fn]
                
                for px in common_prefixes:
                    arrays_to_concat = [
                        lot_pooled[other_fn][px]
                        for other_fn in other_lots
                        if px in lot_pooled[other_fn]
                    ]
                    if arrays_to_concat:
                        baseline_pooled[px] = np.concatenate(arrays_to_concat)

                curr_pooled = lot_pooled[fn]
                stab_feat: Dict[str, float] = {}
                for px in common_prefixes:
                    curr_s = curr_pooled.get(px, np.array([], dtype=float))
                    base_s = baseline_pooled.get(px, np.array([], dtype=float))
                    stab_feat.update(compute_spc_features(curr_s, base_s, px))
                    stab_feat.update(compute_psi_features(curr_s, base_s, px))

                # Merge stability features into the first wafer's feature dict
                # (they are lot-level signals, so they apply equally to all wafers).
                for tup in wafers:
                    tup[0].update(stab_feat)
        else:
            self.log_func(
                f"[INFO] Skipping SPC/PSI feature computation: only {n_lots} lot(s) found "
                f"(minimum {MIN_BASELINE_LOTS} required for a meaningful baseline)."
            )

        return self._aggregate_lots(lot_data, save=save)

    def _aggregate_lots(self, data: Dict[str, List[Tuple[Dict, int, Dict]]], save: bool = True) -> Optional[pd.DataFrame]:
        """Aggregate wafer results to lot rows via weighted average.

        SPC / PSI stability features (lot-level, not per-wafer) are forwarded
        from the first wafer's dict without weighted averaging 鈥?they are the
        same value for every wafer in the lot.
        """
        if not data: return None

        # Identify stability feature keys (lot-level 鈥?no weighted avg needed).
        _STABILITY_PREFIXES = ("__spc__", "__psi__")

        # Stability check: find common features across lots.
        all_f = [vals[0] for lists in data.values() for vals in lists]
        ks = {}
        for d in all_f:
            for k in d: ks[k] = ks.get(k, 0) + 1
        common = {k for k, v in ks.items() if v >= min(3, len(all_f))}
        
        rows = []
        for fn, wafers in data.items():
            tot_pc = sum(p for _, p, _ in wafers)
            row = {"parent_folder": fn, "y": wafers[0][0].get("y", np.nan)}
            for k in common:
                if k in ["y", "pass_count"]: continue
                # Stability features are lot-level constants 鈥?take first value.
                if any(sp in k for sp in _STABILITY_PREFIXES):
                    row[k] = wafers[0][0].get(k, np.nan)
                    continue
                ws = sum(f[k] * p for f, p, _ in wafers if k in f and not pd.isna(f[k]))
                wt = sum(p for f, p, _ in wafers if k in f and not pd.isna(f[k]))
                row[k] = ws / wt if wt > 0 else np.nan
            rows.append(row)

        df = pd.DataFrame(rows)
        cols = ["parent_folder", "y"] + sorted([c for c in df.columns if c not in ["parent_folder", "y"]], key=natural_sort_key)
        df = df[cols]
        if save:
            df.to_csv(self.out_path, index=False)
            self.log_func(f"[SUCCESS] Saved {len(df)} records to {self.out_path.name}")
        return df

def trigger_models_py(data_csv: Path, info: str = ""):
    """Invokes ml_train_model.py after feature extraction."""
    script = Path(__file__).parent / "ml_train_model.py"
    if not script.exists():
        logger.warning(f"Trainer script not found: {script}")
        return
    cmd = [sys.executable, str(script), "--data", str(data_csv)]
    if info: cmd.extend(["--info", info])
    logger.info(f"Triggering training: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_feature_extraction(root: Path, out_path: Path, yield_data_path: Optional[Path] = None, 
                           log_func=print, progress_callback=None):
    """Bridge for Streamlit/CLI to the Extractor class."""
    if out_path.name == DEFAULT_OUT:
        out_path = out_path.with_name(f"merged_features_{root.parent.name}.csv")
    if not out_path.is_absolute():
        out_path = root.parent / out_path
    
    ext_tp = FeatureExtractor(root, out_path, yield_data_path, log_func, progress_callback)
    # Don't save immediately if FT exists
    ft_root = root.parent / "FT_Decrypted"
    has_ft = ft_root.exists() and ft_root.is_dir()
    
    tp_df = ext_tp.extract(save=not has_ft)
    if tp_df is None: return None
    
    if has_ft:
        log_func("[INFO] FT_Decrypted located. Extracting Final Test correlation targets...")
        ext_ft = FeatureExtractor(ft_root, Path("dummy_ft.csv"), yield_data_path, log_func, progress_callback)
        ft_df = ext_ft.extract(save=False)
        
        if ft_df is not None:
            # Prefix columns cleanly
            tp_df.columns = ["TP_" + c if c not in ["parent_folder", "y"] else ("TP_y" if c == "y" else c) for c in tp_df.columns]
            ft_df.columns = ["FT_" + c if c not in ["parent_folder", "y"] else ("FT_y" if c == "y" else c) for c in ft_df.columns]
            
            # Merge
            df = pd.merge(tp_df, ft_df, on="parent_folder", how="inner")
            df.to_csv(out_path, index=False)
            log_func(f"[SUCCESS] Saved multi-output correlation records to {out_path.name} ({len(df)} rows joined)")
            return df

    return tp_df

def main():
    parser = argparse.ArgumentParser(description="Professional ML Feature Hub")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Data directory root")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Path for CSV output")
    parser.add_argument("--yield-data", help="Optional external yield results file")
    parser.add_argument("--train", action="store_true", help="Auto-trigger training after extraction")
    parser.add_argument("--info", help="Metadata info for the training step")
    args = parser.parse_args()

    r_path, o_path = Path(args.root).resolve(), Path(args.out)
    if not o_path.is_absolute(): o_path = (r_path.parent / o_path).resolve()
    
    df = run_feature_extraction(r_path, o_path, Path(args.yield_data) if args.yield_data else None, logger.info)
    if df is not None and args.train:
        trigger_models_py(o_path, args.info or "")

if __name__ == "__main__":
    main()
