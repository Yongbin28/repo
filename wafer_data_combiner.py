
import os
import re
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Any, Union

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
MANCHESTER_SIG = "Created from DLOG"

def parse_header(filepath: Path) -> Tuple[List[str], Dict[str, Any], List[str], List[str], int]:
    """
    Parses the CSV header to extract metadata, T-names, and Units.
    
    Robustly identifies the data start line by searching for 'Device #,'.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Tuple containing:
            - metadata_lines: Raw strings before the data headers.
            - metadata_dict: Extracted key-value pairs (e.g., creation date).
            - t_names: List of test numbers/names.
            - units: List of measurement units.
            - data_start_line: 0-indexed line number where data begins.
    """
    metadata: Dict[str, Any] = {}
    all_lines: List[str] = []
    data_start_line = -1
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                all_lines.append(line)
                if "Device #," in line:
                    data_start_line = i
                    break
            
            if data_start_line == -1:
                return [], {}, [], [], 0

            # T-names: standard index (start-2). Units: standard index (start-1).
            t_row_idx = data_start_line - 2
            u_row_idx = data_start_line - 1
            
            t_names = [x.strip() for x in all_lines[t_row_idx].split(',')] if t_row_idx >= 0 else []
            units = [x.strip() for x in all_lines[u_row_idx].split(',')] if u_row_idx >= 0 else []
            metadata_lines = [l.strip() for l in all_lines[:max(0, t_row_idx)]]
            
            for line in metadata_lines:
                if "File Creation Date:" in line:
                    date_str = line.split("File Creation Date:")[1].strip()
                    metadata['file_creation_date_str'] = date_str
                    for fmt in ("%m/%d/%Y %H:%M:%S", "%b %d %Y %H:%M:%S"):
                        try:
                            metadata['file_creation_date'] = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    if 'file_creation_date' not in metadata:
                        metadata['file_creation_date'] = date_str
    except Exception as e:
        logger.error(f"Error parsing header for {filepath}: {e}")
        return [], {}, [], [], 0
            
    return metadata_lines, metadata, t_names, units, data_start_line

def extract_wafers_from_combo(name: str) -> List[str]:
    """Parses wafer IDs from combo folder name string like '[01-25]' or '[01,02]'."""
    match = re.search(r'\[(.*?)\]', name)
    if not match:
        return []
    
    raw = match.group(1)
    out: List[str] = []
    parts = [p.strip() for p in raw.replace(';', ',').split(',') if p.strip()]
    for p in parts:
        m_range = re.fullmatch(r"(\d{1,3})\s*-\s*(\d{1,3})", p)
        if m_range:
            a, b = int(m_range.group(1)), int(m_range.group(2))
            step = 1 if b >= a else -1
            for w in range(a, b + step, step):
                out.append(f"{w:02d}")
        elif re.fullmatch(r"\d{1,3}", p):
            out.append(f"{int(p):02d}")
    return out

def get_wafer_info_from_filename(filename: str, combo_folder_name: str) -> Tuple[str, Optional[int], str, str]:
    """
    Extracts Lot ID, Wafer Number, Prefix, and Suffix from a filename.
    
    Args:
        filename: The filename to parse.
        combo_folder_name: The name of the parent folder containing the wafer set.
        
    Returns:
        Tuple of (lot_id, wafer_num, prefix, suffix).
    """
    # 1. Resolve Lot ID
    parts = combo_folder_name.split('_[')
    if len(parts) > 1:
        lot_id = parts[0].split('_')[-1]
    else:
        lot_match = re.search(r'W\d+(?:[\.\-]\d+)?', combo_folder_name, re.IGNORECASE)
        lot_id = lot_match.group(0) if lot_match else combo_folder_name.split('_')[0]
    
    expected_wafers = extract_wafers_from_combo(combo_folder_name)
    tokens = re.split(r'[_,\s\-]+', filename)
    
    wafer_num: Optional[int] = None
    wafer_token_idx = -1
    prefix = ""
    suffix = "WS"

    # Identify Lot token
    lot_token_idx = -1
    for i, tok in enumerate(tokens):
        if lot_id.upper() == tok.upper() or tok.upper().startswith(lot_id.upper()):
            lot_token_idx = i
            break
            
    if lot_token_idx != -1:
        candidates = []  # (val, index, priority)
        
        def is_valid(v: int) -> bool:
            if v > 25: return False
            v_str = f"{v:02d}"
            return not expected_wafers or v_str in expected_wafers

        # Priority 1 & 3: separate tokens
        for i in range(lot_token_idx + 1, min(lot_token_idx + 4, len(tokens))):
            curr_tok = tokens[i]
            clean = re.sub(r'^[wW]?', '', curr_tok)
            if clean.isdigit() and is_valid(int(clean)):
                candidates.append((int(clean), i, 1))
            else:
                m = re.match(r'^[wW]?(0*\d{1,2})', curr_tok)
                if m and is_valid(int(m.group(1))):
                    candidates.append((int(m.group(1)), i, 3))

        # Priority 2 & 4: suffix in lot token
        lot_tok = tokens[lot_token_idx]
        if len(lot_tok) > len(lot_id):
            rem = lot_tok[len(lot_id):].lstrip('_-')
            clean = re.sub(r'^[wW]?', '', rem.lstrip('.'))
            if clean.isdigit() and is_valid(int(clean)):
                candidates.append((int(clean), lot_token_idx, 2))
            else:
                m = re.match(r'^[wW]?(0*\d{1,2})', rem.lstrip('.'))
                if m and is_valid(int(m.group(1))):
                    candidates.append((int(m.group(1)), lot_token_idx, 4))

        if candidates:
            candidates.sort(key=lambda x: x[2])
            wafer_num = candidates[0][0]
            wafer_token_idx = candidates[0][1]

    if wafer_num is not None and wafer_token_idx != -1:
        s_tokens = []
        for tok in tokens[wafer_token_idx + 1:]:
            clean = re.split(r'\.std|\.csv', tok, flags=re.IGNORECASE)[0]
            if not clean or re.fullmatch(r'\d{5,}', clean) or "combined" in clean.lower():
                break
            s_tokens.append(clean)
        if s_tokens:
            suffix = "_".join(s_tokens)

    if lot_id in filename:
        prefix = filename.split(lot_id)[0]

    return lot_id, wafer_num, prefix, suffix

class WaferCombiner:
    """Manages the merging and combination of multi-source wafer CSV files."""
    
    def __init__(self, log_func=None):
        self.log_func = log_func

    def _log(self, msg: str, level: str = "INFO"):
        """Centralized logging method."""
        print(f"[{level}] {msg}")
        if self.log_func:
            self.log_func(msg, level)
        else:
            if level == "INFO": logger.info(msg)
            elif level == "WARN": logger.warning(msg)
            elif level == "ERR": logger.error(msg)

    def merge_csvs(self, file_list: List[Path], output_path: Path):
        """
        Core logic to merge list of CSV files for one wafer.
        
        Selection Priority:
        1. Devices that PASSED (no fails).
        2. Newest file by creation date.
        """
        if not file_list: return

        # 1. Analyze files and sort by date
        valid_files = []
        for fp in file_list:
            if fp.stat().st_size == 0: continue
            _, meta, _, _, start = parse_header(fp)
            if start != -1:
                valid_files.append({'path': fp, 'date': meta.get('file_creation_date', datetime.min)})
        
        if not valid_files: return
        valid_files.sort(key=lambda x: x['date'], reverse=True)

        # 2. Canonical mapping of T-numbers
        tnum_to_header: Dict[str, Tuple[str, str]] = {}
        ordered_test_names: List[str] = []
        
        for info in valid_files:
            _, _, t_nums, units, start = parse_header(info['path'])
            df_cols = pd.read_csv(info['path'], skiprows=start, nrows=0).columns
            df_cols = [c.strip() for c in df_cols]
            for i, col in enumerate(df_cols):
                if i >= 7:
                    t_val = t_nums[i].strip() if i < len(t_nums) else ""
                    key = t_val if t_val else col
                    if key not in tnum_to_header:
                        tnum_to_header[key] = (col, units[i] if i < len(units) else "")
                        ordered_test_names.append(col)

        # 3. Row Merging (Oldest to Newest)
        master_data: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        for info in sorted(valid_files, key=lambda x: x['date']):
            _, _, t_nums, _, start = parse_header(info['path'])
            df = pd.read_csv(info['path'], skiprows=start, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            
            # Map current file cols to canonical names
            remap = {}
            for i, c in enumerate(df.columns):
                if i >= 7:
                    t_val = t_nums[i].strip() if i < len(t_nums) else ""
                    lookup = t_val if t_val else c
                    if lookup in tnum_to_header:
                        remap[c] = tnum_to_header[lookup][0]
            df.rename(columns=remap, inplace=True)

            xc, yc = df.columns[3], df.columns[4]
            for _, row in df.iterrows():
                key = (row[xc], row[yc])
                r_dict = row.to_dict()
                f_val = str(r_dict.get('Fails', '')).strip().lower()
                is_pass = f_val in ("", "nan", "0")
                
                if key not in master_data:
                    master_data[key] = r_dict
                    master_data[key].update({'_date': info['date'], '_pass': is_pass})
                else:
                    existing = master_data[key]
                    # Logic: Replace if (new is pass AND existing is fail) OR (same status AND new is newer)
                    replace = (is_pass and not existing['_pass']) or \
                              (is_pass == existing['_pass'] and info['date'] >= existing['_date'])
                    
                    if replace:
                        master_data[key] = r_dict
                        master_data[key].update({'_date': info['date'], '_pass': is_pass})
                    elif is_pass == existing['_pass']:
                        # Partial merge if same status but existing is newer - fill missing tests
                        for k, v in r_dict.items():
                            if k not in ['Device #', 'Bin', 'Site', xc, yc, 'Fails', 'Alarms']:
                                if pd.isna(existing.get(k)) or str(existing.get(k)) == '':
                                    existing[k] = v

        # 4. Final Formatting & Write
        latest_path = valid_files[0]['path']
        raw_meta, _, _, _, start = parse_header(latest_path)
        base_cols = ["Device #", "Bin", "Site", "X", "Y", "Fails", "Alarms"]
        final_cols = base_cols + ordered_test_names
        
        col_to_tname = {name: t for t, (name, _) in tnum_to_header.items()}
        col_to_unit = {name: u for _, (name, u) in tnum_to_header.items()}
        
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            for line in raw_meta:
                if MANCHESTER_SIG in line: continue
                f.write(line.strip() + "," * (len(final_cols) - 1) + "\n")
            f.write("," * (len(final_cols) - 1) + "\n")
            f.write(",".join([""]*7 + [col_to_tname.get(c, "") for c in final_cols[7:]]) + "\n")
            f.write(",".join([""]*7 + [col_to_unit.get(c, "") for c in final_cols[7:]]) + "\n")
            
            writer = csv.DictWriter(f, fieldnames=final_cols, extrasaction='ignore')
            writer.writeheader()
            for k in sorted(master_data.keys(), key=lambda z: (str(z[0]), str(z[1]))):
                row = {k_sub: ("" if pd.isna(v_sub) else v_sub) for k_sub, v_sub in master_data[k].items()}
                writer.writerow(row)

    def run_folder(self, folder_path: Union[str, Path]):
        """Recursively processes a folder for wafer combinations."""
        fp = Path(folder_path)
        if not fp.exists(): return
        
        all_csvs = [c for c in fp.rglob("*.csv") 
                    if "_limits" not in c.name.lower() and "_combined" not in c.name.lower()]
        
        groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for c in all_csvs:
            lot, wafer, pref, suff = get_wafer_info_from_filename(c.name, fp.name)
            if wafer is None: continue
            key = (lot, wafer)
            if key not in groups:
                groups[key] = {'files': [], 'pref': pref, 'suff': suff}
            groups[key]['files'].append(c)

        for (lot, wafer), gd in groups.items():
            if len(gd['files']) <= 1: continue
            
            out_name = f"{gd['pref']}{lot}_{wafer:02d}_{gd['suff']}_combined.csv"
            out_path = fp / out_name
            self._log(f"Wafer {wafer:02d}: Combining {len(gd['files'])} sources...")
            
            try:
                self.merge_csvs(gd['files'], out_path)
                if out_path.exists():
                    for src in gd['files']:
                        if src.resolve() != out_path.resolve():
                            try: os.remove(src)
                            except: pass
            except Exception as e:
                self._log(f"Failed Wafer {wafer:02d}: {e}", "ERR")

        # Cleanup limits folder
        lim_dir = fp / "limit"
        if lim_dir.exists():
            l_files = list(lim_dir.glob("*_limits.csv"))
            if len(l_files) > 1:
                def get_ts(n):
                    m = re.search(r'(\d{6,8})_(\d{6})', n)
                    if m:
                        try: return datetime.strptime(m.group(0), "%m%d%Y_%H%M%S")
                        except: pass
                    return datetime.min
                l_files.sort(key=lambda x: get_ts(x.name), reverse=True)
                for f_old in l_files[1:]:
                    try: os.remove(f_old)
                    except: pass

def _is_combo(name: str) -> bool:
    """Helper to identify if a folder name contains a combo pattern (e.g. EB_[01-25])."""
    return bool(re.search(r'_\d*[\[]', name))

def find_and_process_all(root_path: Union[str, Path], log_func=None):
    """
    Main entry point for scanning and processing all combo folders.
    
    Walks the specified directory for 'T&P_Decrypted' folders and runs 
    the WaferCombiner logic on each legitimate child directory.
    """
    combiner = WaferCombiner(log_func=log_func)
    target = Path(root_path)
    if not target.exists():
        return
    
    if _is_combo(target.name):
        combiner.run_folder(target)
    else:
        for root, dirs, _ in os.walk(target):
            if "T&P_Decrypted" in dirs:
                tp_dir = Path(root) / "T&P_Decrypted"
                for child in tp_dir.iterdir():
                    if child.is_dir() and _is_combo(child.name):
                        combiner.run_folder(child)

def inject_resistance_sensors(folder_path: Union[str, Path]):
    """
    WaferPulse Data Augmenter
    Injects virtual resistance sensor data into decrypted CSV logs to simulate 
    Probe Mark Damage (PMD) signatures.
    """
    fp = Path(folder_path)
    if not fp.exists(): return
    
    import random
    import numpy as np

    print(f"[INFO] Augmenting factory logs with WaferPulse virtual sensors: {fp.name}")
    all_csvs = list(fp.rglob("*.csv"))
    
    for csv_path in all_csvs:
        if "_limits" in csv_path.name.lower(): continue
        if "_combined" in csv_path.name.lower(): continue
        
        try:
            # Read metadata and data start
            meta_lines, meta_dict, t_names, units, start_line = parse_header(csv_path)
            if start_line <= 0: continue
            
            df = pd.read_csv(csv_path, skiprows=start_line)
            
            # 15% probability of PMD defect per wafer
            has_pmd = random.random() < 0.15
            
            # Inject sensor columns
            # Healthy range: mean 0.9, std 0.02
            # Defect range: mean 1.5, std 0.35
            if has_pmd:
                df['Resistance_mean'] = np.random.normal(1.5, 0.35, len(df))
                df['Resistance_median'] = df['Resistance_mean'] + np.random.normal(0, 0.05, len(df))
                df['Resistance_std'] = np.random.normal(0.35, 0.1, len(df))
            else:
                df['Resistance_mean'] = np.random.normal(0.9, 0.02, len(df))
                df['Resistance_median'] = df['Resistance_mean'] + np.random.normal(0, 0.005, len(df))
                df['Resistance_std'] = np.random.normal(0.02, 0.005, len(df))
                
            # Rewrite file with original headers
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                for line in meta_lines:
                    f.write(line.strip() + "\n")
                
                # The data structure has T-names at start-2 and Units at start-1
                # If we have 7 metadata lines, data starts at line 9 (0-indexed).
                # T-names are at line 7, Units at line 8.
                
                # Fill t_names and units to match the new column count
                new_t_names = t_names + ["Resistance_mean", "Resistance_median", "Resistance_std"]
                new_units = units + ["Ohm", "Ohm", "Ohm"]
                
                f.write(",".join(new_t_names) + "\n")
                f.write(",".join(new_units) + "\n")
                
                df.to_csv(f, index=False)
                
        except Exception as e:
            logger.error(f"Failed to inject sensors into {csv_path.name}: {e}")

def main():
    import sys
    if len(sys.argv) > 1:
        find_and_process_all(sys.argv[1])

if __name__ == "__main__":
    main()
