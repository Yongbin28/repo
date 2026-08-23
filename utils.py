import pandas as pd
import re
import os
import sys
import hashlib
from pathlib import Path
import streamlit as st

# --- Path Configuration ---
if getattr(sys, 'frozen', False):
    # If the application is packaged (PyInstaller), ROOT_DIR is the folder containing the .exe
    ROOT_DIR = Path(sys.executable).resolve().parent
else:
    # In development, ROOT_DIR is the script directory
    ROOT_DIR = Path(__file__).resolve().parent

# Add ROOT_DIR to sys.path to allow importing sibling modules
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Standardized Data Roots
DATASET_ROOT = ROOT_DIR / "dataset"
PIPELINE_ROOT = ROOT_DIR / "pipeline"
CFC_ROOT = ROOT_DIR / "cfc"

# Create essential directories (avoid auto-creating cfc directory as it's not needed by default)
for d in [DATASET_ROOT, PIPELINE_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

# --- Config ---
def _get_resource_path(filename: str) -> Path:
    if getattr(sys, "frozen", False):
        # 1) check next to EXE (user-modifiable resources)
        external_path = ROOT_DIR / filename
        if external_path.exists(): return external_path
        # 2) fallback to internal PyInstaller cache (bundled-only resources)
        if hasattr(sys, "_MEIPASS"): return Path(sys._MEIPASS).resolve() / filename
        return external_path
    return ROOT_DIR / filename

TESTER_MAPPING_FILE = _get_resource_path("TesterFamilyMap.xlsx")
LOG_FILE = PIPELINE_ROOT / "pipeline_execution.log"

def log(msg, level="INFO"):
    ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{ts}] [{level}] {msg}"
    
    # Use sys.stdout.buffer to print with a specific encoding to avoid 'charmap' errors on Windows
    try:
        print(formatted_msg)
    except UnicodeEncodeError:
        # Fallback for old/unsupported consoles: print as ASCII-safe or just ignore non-encodable chars
        print(formatted_msg.encode('ascii', 'replace').decode('ascii'))
    
    # 1. Update Session State
    if "logs" in st.session_state:
        st.session_state.logs.append(formatted_msg)
    
    # 2. Persist to File
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(formatted_msg + "\n")
    except Exception:
        # This prevents the application from crashing due to logging errors (e.g., a file lock).
        pass

def get_tester_family_for_generic(generic):
    """
    Reads the mapping file to determine the tester family associated with a generic.
    Returns: family_name or "UNKNOWN_FAMILY"
    """
    if not TESTER_MAPPING_FILE.exists():
        log(f"Mapping file not found: {TESTER_MAPPING_FILE}", "WARN")
        return "UNKNOWN_FAMILY"
    
    try:
        # Simple pandas logic, similar to the approach used in ADYAP, is employed.
        df = pd.read_excel(TESTER_MAPPING_FILE)
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        
        # Generic and tester columns are identified.
        g_col = next((c for c in df.columns if "generic" in c), None)
        f_col = next((c for c in df.columns if "tester" in c), None)
        
        if not g_col or not f_col:
            log("Could not identify 'Generic' or 'Tester' columns in map.", "ERR")
            return "UNKNOWN_FAMILY"
            
        match = df[df[g_col].astype(str).str.strip().str.lower() == generic.lower()]
        if not match.empty:
            fam = str(match.iloc[0][f_col]).strip().upper()
            fam = re.sub(r'_C[1-5]$', '', fam)
            if "J750" in fam or "TS88" in fam or "ETS" in fam:
                return "J750"
            if "EAGLE" in fam:
                return "EAGLE"
            if "CATALYST" in fam:
                return "CATALYST"
            if "HP94K" in fam:
                return "HP94K"
            return fam
    except Exception as e:
        log(f"Error reading mapping file: {e}", "ERR")
        
    return "UNKNOWN_FAMILY"

@st.cache_resource
def get_generics_by_tester():
    """
    Reads the mapping file and returns a dictionary where keys are tester families and values are lists of generics.
    """
    if not TESTER_MAPPING_FILE.exists():
        return {}
    
    tester_groups = {}
    try:
        df = pd.read_excel(TESTER_MAPPING_FILE)
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        
        g_col = next((c for c in df.columns if "generic" in c), None)
        f_col = next((c for c in df.columns if "tester" in c), None)
        
        if g_col and f_col:
            # Drop NaN
            df = df.dropna(subset=[g_col, f_col])
            
            for _, row in df.iterrows():
                gen = str(row[g_col]).strip()
                if not gen or gen.upper() == "NAN":
                    continue
                fam = str(row[f_col]).strip().upper() # The family name is normalized to uppercase.
                
                # The _C1, _C2, _C3, and _C4 suffixes are removed first.
                fam = re.sub(r'_C[1-5]$', '', fam)
                
                # Basic normalization of family names is performed to facilitate grouping.
                if "EAGLE" in fam: fam = "EAGLE"
                elif "CATALYST" in fam: fam = "CATALYST"
                elif "J750" in fam or "TS88" in fam or "ETS" in fam: fam = "J750"
                elif "HP94K" in fam: fam = "HP94K"
                
                if fam not in tester_groups:
                    tester_groups[fam] = []
                tester_groups[fam].append(gen)
                
            for k in tester_groups:
                tester_groups[k] = sorted(list(set(tester_groups[k])))
                
    except Exception as e:
        log(f"Error reading mapping for grouping: {e}", "WARN")
        
    return tester_groups

@st.cache_resource
def get_all_generics_from_map():
    """Extracts all unique Generic names from the mapping file."""
    if not TESTER_MAPPING_FILE.exists():
        return []
    try:
        df = pd.read_excel(TESTER_MAPPING_FILE)
        # Normalize columns
        df.columns = [c.strip().lower() for c in df.columns]
        g_col = next((c for c in df.columns if "generic" in c), None)
        if g_col:
            # NaN values are dropped, entries are converted to strings and stripped of whitespace, and the unique, sorted list is returned.
            all_gens = sorted(df[g_col].dropna().astype(str).str.strip().unique().tolist())
            return [g for g in all_gens if g and g.upper() != "NAN"]
    except Exception:
        pass
    return []

def select_decryptor(family):
    """Returns the universal decryptor module. Imports are handled lazily."""
    import stdf_decryptor
    return stdf_decryptor

def _is_combo_folder_name(name: str) -> bool:
    """Return True if name matches FABLOT_[WW] or FABLOT_[WW]_YY% (e.g., W2512024_[19])."""
    if not name or name.count('_') < 1:
        return False
    parts = name.split('_')
    
    # CASE 1: FABLOT_[WW]_YY% (3+ parts)
    if len(parts) >= 3 and parts[-1].endswith('%'):
        fablot = '_'.join(parts[:-2])
        wafer = parts[-2]
        yld = parts[-1]
        if fablot and wafer.startswith('[') and wafer.endswith(']'):
            wafer_num = wafer[1:-1]
            if all(c.isdigit() or c in "-, " for c in wafer_num):
                try:
                    float(yld[:-1])
                    return True
                except: pass

    # CASE 2: FABLOT_[WW] (2+ parts)
    if len(parts) >= 2:
        fablot = '_'.join(parts[:-1])
        wafer = parts[-1]
        if fablot and wafer.startswith('[') and wafer.endswith(']'):
            wafer_num = wafer[1:-1]
            if all(c.isdigit() or c in "-, " for c in wafer_num):
                return True

    return False

def shorten_wafer_list(name: str) -> str:
    """
    # Long wafer lists in folder names are shortened to avoid Windows path length limits.
    # E.g., [01-25] is used instead of [01-02-03...-25].
    """
    if '[' not in name or ']' not in name:
        return name
    
    prefix = name.split('[')[0]
    wafer_part = name.split('[')[1].split(']')[0]
    suffix = name.split(']')[1]
    
    # Parsing of the range is attempted.
    import re
    tokens = re.split(r'[-\s,]+', wafer_part)
    nums = []
    for t in tokens:
        if t.isdigit(): nums.append(int(t))
    
    if len(nums) > 5:
        min_w, max_w = min(nums), max(nums)
        # If the sequence is perfect or excessively long, range notation is used.
        new_wafer_part = f"{min_w:02d}-{max_w:02d}"
        return f"{prefix}[{new_wafer_part}]{suffix}"
    
    return name

import zipfile
import tempfile
def explode_and_collect_data_files(input_path: Path, candidate_exts, base_temp_dir=None):
    """
    Recursively find all data files (STDF, XFS, etc.) within a folder or zip.
    Handles nested zips (even nested .stdf.zip or .xfs.zip).
    Returns: (list of (extracted_file_path, original_data_name), list_of_temp_dirs)
    """
    results = []
    temp_dirs = []
    
    # The .zip extension is added to candidate_exts for the recursive search phase.
    
    # The .zip extension is added to candidate_exts for the recursive search phase
    # If it is not already present, ensuring intermediate zips are not missed.
    search_exts = set(candidate_exts)
    search_exts.add(".zip")

    def process_recursive(current_path: Path, display_name: str):
        name = current_path.name.lower()
        is_zip = name.endswith(".zip") or zipfile.is_zipfile(current_path)
        
        # 1. If the file matches a final data extension (excluding zips unless they are a known data-zip type).
        # Certain testers use .stdf.zip as the data container.
        # A check is performed to see if the file is a "leaf" data file.
        is_data = any(name.endswith(ext) for ext in candidate_exts)
        
        if is_data and not is_zip:
            results.append((current_path, display_name))
            return

        # 2. If the file is a zip, it is extracted and recursion is performed.
        if is_zip:
            if base_temp_dir:
                t = Path(tempfile.mkdtemp(prefix="app_explode_", dir=base_temp_dir))
            else:
                t = Path(tempfile.mkdtemp(prefix="app_explode_"))
            temp_dirs.append(t)
            try:
                with zipfile.ZipFile(str(current_path), "r") as zf:
                    for n in zf.namelist():
                        # Mac system files are skipped.
                        if n.startswith('__MACOSX/') or n.endswith('.DS_Store'):
                            continue
                        
                        zf.extract(n, t)
                        extracted_p = t / n
                        if extracted_p.is_file():
                            # For nested items, the innermost name is kept for display, although it can be relative if required.
                            process_recursive(extracted_p, n)
            except (zipfile.BadZipFile, Exception):
                pass

    process_recursive(input_path, input_path.name)
    return results, temp_dirs

def infer_combo_folder_name(source_path: Path):
    """Extracts the combo folder name from the real container/folder structure, without using fuzzy matching."""
    # 1) If the source is a ZIP file, its top-level folders are read.
    try:
        if source_path.is_file() and zipfile.is_zipfile(source_path):
            with zipfile.ZipFile(str(source_path), 'r') as zf:
                top_levels = set()
                for n in zf.namelist():
                    n = n.replace('\\', '/')
                    if not n or n.startswith('__MACOSX/'):
                        continue
                    top = n.split('/', 1)[0]
                    if top:
                        top_levels.add(top)
                for top in sorted(top_levels):
                    if _is_combo_folder_name(top):
                        return top
    except Exception:
        pass

    # 2) The source path is traversed upward, and the first folder matching the format is returned.
    for parent in [source_path, source_path.parent, *source_path.parents]:
        if _is_combo_folder_name(parent.name):
            return parent.name

    return None

def extract_fiscal_year(text: str) -> str:
    """
    Extracts a 4-digit fiscal year from text (filename or folder name).
    Supports patterns like MMDDYYYY and YYYYMMDD often found in datalogs.
    """
    if not text: return "Unknown"
    
    # 1. Look for MMDDYYYY or YYYYMMDD in an 8-digit block
    # Matches _04252026_ (MMDDYYYY) or _20260425_ (YYYYMMDD)
    matches = re.findall(r'_(\d{8})_', text)
    for m in matches:
        # Check MMDDYYYY (last 4 digits are year)
        year_cand = m[4:8]
        if 2010 <= int(year_cand) <= 2040:
            return year_cand
        # Check YYYYMMDD (first 4 digits are year)
        year_cand = m[0:4]
        if 2010 <= int(year_cand) <= 2040:
            return year_cand
            
    # 2. Look for YYYY at start of 8-digit block with different separators
    matches = re.findall(r'(?:\b|_|-)((?:20|19)\d{6})(?:\b|_|-)', text)
    for m in matches:
        return m[:4]
        
    # 3. Look for YYYY at end of 8-digit block with different separators
    matches = re.findall(r'(?:\b|_|-)(\d{4}(?:20|19)\d{2})(?:\b|_|-)', text)
    for m in matches:
        return m[4:]
        
    # 4. Fallback to any standalone 4-digit year in range 2010-2040
    matches = re.findall(r'(?:\b|_|-)(20\d{2})(?:\b|_|-)', text)
    if matches:
        return matches[0]
        
    return "Unknown"
