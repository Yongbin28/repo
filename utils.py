import pandas as pd
import re
import os
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
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


def sniff_dlog_metadata(payload: Any, filename: str = "") -> dict:
    """
    Intelligently extracts semiconductor metadata (Generic, PartName, Lot ID, Tester) 
    from binary STDF files, CSV text headers, or filenames, and cross-references with TesterFamilyMap.xlsx.
    """
    import struct

    meta = {
        "generic": "",
        "partname": "",
        "lot_id": "",
        "job": "",
        "tester": "",
    }

    file_bytes = b""
    if isinstance(payload, Path) or (isinstance(payload, str) and os.path.isfile(str(payload))):
        p = Path(payload)
        filename = filename or p.name
        try:
            with open(p, "rb") as f:
                file_bytes = f.read(65536)
        except Exception:
            pass
    elif isinstance(payload, bytes):
        file_bytes = payload[:65536]
    elif hasattr(payload, "read"):
        try:
            pos = payload.tell() if hasattr(payload, "tell") else 0
            file_bytes = payload.read(65536)
            if hasattr(payload, "seek"):
                payload.seek(pos)
        except Exception:
            pass
    elif hasattr(payload, "getvalue"):
        try:
            file_bytes = payload.getvalue()[:65536]
        except Exception:
            pass

    # 1. Binary STDF Master Information Record (MIR: type 1, sub 10)
    if len(file_bytes) >= 4:
        first_4 = file_bytes[:4]
        try:
            endian = ">" if struct.unpack(">H", first_4[:2])[0] == 2 else "<"
            offset = 0
            limit_search = min(len(file_bytes), 65536)
            while offset < limit_search - 4:
                rec_len, rec_typ, rec_sub = struct.unpack_from(f"{endian}HBB", file_bytes, offset)
                offset += 4
                if rec_typ == 1 and rec_sub == 10:  # MIR
                    curr = offset + 15
                    lim = offset + rec_len

                    def _read_str(p: int) -> tuple[str, int]:
                        if p >= lim:
                            return "", p
                        sl = file_bytes[p]
                        p += 1
                        if sl == 0 or p + sl > lim:
                            return "", p + sl
                        try:
                            return file_bytes[p : p + sl].decode("utf-8", "ignore").strip(), p + sl
                        except Exception:
                            return "", p + sl

                    lot, curr = _read_str(curr)
                    ptyp, curr = _read_str(curr)
                    node, curr = _read_str(curr)
                    tstr, curr = _read_str(curr)
                    job, _ = _read_str(curr)
                    meta["lot_id"] = lot
                    meta["partname"] = ptyp
                    meta["tester"] = tstr or node
                    meta["job"] = job
                    break
                offset += rec_len
        except Exception:
            pass

    # 2. Text CSV / DLog header lines
    if not meta["partname"] and not meta["lot_id"] and file_bytes:
        try:
            head_text = file_bytes[:4096].decode("utf-8", "ignore")
            for line in head_text.splitlines()[:25]:
                clean_l = line.strip()
                m_part = re.search(r"(?:part\s*name|part|ptyp)[\s:=,]+([A-Za-z0-9_\-]+)", clean_l, re.I)
                if m_part and not meta["partname"]:
                    meta["partname"] = m_part.group(1).strip()
                m_lot = re.search(r"(?:lot\s*id|lot)[\s:=,]+([A-Za-z0-9_\-]+)", clean_l, re.I)
                if m_lot and not meta["lot_id"]:
                    meta["lot_id"] = m_lot.group(1).strip()
                m_gen = re.search(r"(?:generic)[\s:=,]+([A-Za-z0-9_\-]+)", clean_l, re.I)
                if m_gen and not meta["generic"]:
                    meta["generic"] = m_gen.group(1).strip()
        except Exception:
            pass

    # 3. Infer from filename
    if filename:
        fn_clean = Path(filename).stem
        tokens = re.split(r"[_.\-]+", fn_clean)
        for tok in tokens:
            t_upper = tok.upper()
            if not meta["generic"] and t_upper in ["DDR4SDRAM", "DDR5SDRAM", "LPDDR4", "LPDDR5", "NAND"]:
                meta["generic"] = t_upper
            if not meta["partname"] and re.match(r"^MT[0-9A-Z]{5,}", t_upper):
                meta["partname"] = t_upper
            if not meta["lot_id"] and re.match(r"^(?:LOT|CDK|WAFER)[0-9A-Z]*", t_upper):
                meta["lot_id"] = tok

    # 4. Cross-reference with TesterFamilyMap.xlsx
    try:
        if TESTER_MAPPING_FILE.exists():
            df = pd.read_excel(TESTER_MAPPING_FILE)
            df.columns = [c.strip().lower() for c in df.columns]
            p_col = next((c for c in df.columns if "part" in c), None)
            g_col = next((c for c in df.columns if "generic" in c), None)
            t_col = next((c for c in df.columns if "tester" in c), None)
            if p_col and g_col:
                if meta["partname"]:
                    p_str = meta["partname"].strip().upper()
                    match = df[df[p_col].astype(str).str.strip().str.upper() == p_str]
                    if not match.empty:
                        if not meta["generic"]:
                            meta["generic"] = str(match.iloc[0][g_col]).strip()
                        if not meta["tester"] and t_col:
                            meta["tester"] = str(match.iloc[0][t_col]).strip()
                elif meta["generic"]:
                    g_str = meta["generic"].strip().upper()
                    match = df[df[g_col].astype(str).str.strip().str.upper() == g_str]
                    if not match.empty and not meta["partname"]:
                        meta["partname"] = str(match.iloc[0][p_col]).strip()
    except Exception:
        pass

    return meta


def discover_dataset_products(dataset_root: Path = DATASET_ROOT) -> list:
    """
    Scans the dataset directory to automatically discover all available products (Generic, PartName),
    along with their tester family, trained models, and available lot data files.
    Allows the UI to present ready-to-use selections without manual user typing.
    """
    products = []
    if not dataset_root.exists():
        return products

    candidate_prod_dirs = []
    for top_dir in sorted(dataset_root.iterdir()):
        if not top_dir.is_dir() or top_dir.name.startswith("."):
            continue

        # Check for tester family subdirectories (e.g. J750/DDR4SDRAM_MT40A1G8)
        sub_dirs = [d for d in top_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
        is_family = False
        for sub in sub_dirs:
            if (sub / "Model").exists() or (sub / "T&P_Decrypted").exists() or any(sub.glob("*.csv")):
                candidate_prod_dirs.append((top_dir.name, sub))
                is_family = True

        # If top_dir is not a family folder, it could be a direct product folder
        if not is_family and ((top_dir / "Model").exists() or (top_dir / "T&P_Decrypted").exists() or any(top_dir.glob("*.csv"))):
            candidate_prod_dirs.append(("Default", top_dir))

    for family, prod_dir in candidate_prod_dirs:
        folder_name = prod_dir.name
        # Example: DDR4SDRAM_MT40A1G8 -> Generic=DDR4SDRAM, PartName=MT40A1G8
        if "_" in folder_name:
            parts = folder_name.split("_", 1)
            generic = parts[0]
            partname = parts[1]
        else:
            generic = folder_name
            partname = ""

        # Collect available lot / wafer files
        sample_files = []

        # 1. Direct CSV/STDF files in the product directory
        for f in sorted(prod_dir.glob("*.csv")):
            if not f.name.endswith("_limits.csv") and not f.name.startswith("limit"):
                m_yd = re.search(r"(\d+\.?\d*)%", f.name)
                yd_str = f" [Yield: {m_yd.group(1)}%]" if m_yd else ""
                m_lot = re.search(r"(SYN_\d+)", f.name)
                lot_str = m_lot.group(1) if m_lot else f.stem[:15]
                sample_files.append({
                    "path": f,
                    "name": f.name,
                    "lot_id": lot_str,
                    "label": f"Sample: {lot_str}{yd_str} ({f.name})"
                })

        for f in sorted(prod_dir.glob("*.stdf")):
            sample_files.append({
                "path": f,
                "name": f.name,
                "lot_id": f.stem,
                "label": f"STDF Sample: {f.name}"
            })

        # 2. Historical lots in T&P_Decrypted
        tp_dir = prod_dir / "T&P_Decrypted"
        if tp_dir.is_dir():
            for lot_dir in sorted(tp_dir.iterdir()):
                if lot_dir.is_dir():
                    csvs = list(lot_dir.glob("*.csv"))
                    if csvs:
                        m_yd = re.search(r"(\d+\.?\d*)%", lot_dir.name)
                        yd_str = f" [Yield: {m_yd.group(1)}%]" if m_yd else ""
                        m_lot = re.search(r"(SYN_\d+)", lot_dir.name)
                        lot_str = m_lot.group(1) if m_lot else lot_dir.name[:15]
                        sample_files.append({
                            "path": csvs[0],
                            "name": csvs[0].name,
                            "lot_id": lot_str,
                            "label": f"Historical: {lot_str}{yd_str} ({lot_dir.name})"
                        })

        has_model = (prod_dir / "Model").is_dir() and any((prod_dir / "Model").glob("*.joblib"))
        
        if sample_files or has_model:
            label = f"{generic} — {partname} ({family})" if partname else f"{generic} ({family})"
            products.append({
                "label": label,
                "generic": generic,
                "partname": partname,
                "family": family,
                "path": prod_dir,
                "has_model": has_model,
                "files": sample_files,
            })

    return products
