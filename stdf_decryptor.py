
import os
import sys
import struct
import csv
import gzip
import zipfile
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Set, Optional, Any, BinaryIO, Union
import concurrent.futures
import numpy as np

# =========================
# CONFIG
# =========================
CHUNK_SIZE = 1024 * 1024  # 1MB buffer for reading
FAILS_MODE = "all"
ALARMS_MODE = "testnum"
INCLUDE_SUMMARY_COLS = False
CANDIDATE_EXTS = (".stdf", ".std", ".std_1", ".stdf.gz", ".std.gz", ".std_1.gz")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# HELPERS
# =========================
def testnum_to_display_id(test_num: int, scale: int = 1) -> str:
    """Convert raw test number to T-code format (e.g., T1.0)."""
    major = test_num // scale
    minor = test_num % scale
    return f"T{major}.{minor}" if scale > 1 else f"T{major}.0"

def fmt_limit(v: Optional[float]) -> str:
    """Format limit values for CSV output, handling infinity/None."""
    if v is None or not np.isfinite(v):
        return ""
    return f"{v:g}"

def fmt_epoch(ts: Optional[int]) -> str:
    """Convert unix epoch timestamp to readable string."""
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts, timezone.utc).strftime("%m/%d/%Y %H:%M:%S")
    except Exception:
        return ""

def read_str_safe(b: bytes, p: int, limit: int) -> Tuple[str, int]:
    """Safely read a Pascal-style string from bytes."""
    if p >= limit:
        return "", p
    sl = b[p]
    p += 1
    if sl == 0:
        return "", p
    if p + sl > limit:
        return "", limit
    try:
        return b[p:p+sl].decode('utf-8', 'ignore').strip(), p + sl
    except Exception:
        return "", p + sl

# =========================
# BINARY PARSER
# =========================
class STDFParser:
    """
    High-performance STDF V4 parser focused on extracting test results to CSV.
    Supports stream-based and memory-mapped parsing.
    """
    def __init__(self, stream: Optional[BinaryIO] = None):
        self.stream = stream
        self.active_parts: Dict[Tuple[int, int], Dict[str, Any]] = {}  # (head, site) -> state
        self.rows: List[Dict[str, Any]] = []
        
        # Metadata maps
        self.col_key_order: List[str] = []
        self.key_from_testnum: Dict[int, str] = {}
        self.best_name_by_key: Dict[str, str] = {}
        self.units_by_key: Dict[str, str] = {}
        self.tcode_by_key: Dict[str, str] = {}
        self.testnum_by_key: Dict[str, int] = {}
        self.lo_limit_by_key: Dict[str, Optional[float]] = {}
        self.hi_limit_by_key: Dict[str, Optional[float]] = {}
        
        self.seen_testnums: Set[int] = set()
        self.test_program: str = ""
        self.file_creation_date: str = ""
        self.site_counters: Dict[int, int] = {}
        self.ptr_cnt: int = 0
        self.endian: str = "<"
        self.scale: int = 1
        
        self._init_structs()

    def _init_structs(self) -> None:
        """Prefetch struct objects for optimized unpacking."""
        e = self.endian
        self.header_struct = struct.Struct(f"{e}HBB")      # len, type, sub
        self.ptr_fixed_struct = struct.Struct(f"{e}IBBBBf") # 12 bytes
        self.prr_fixed_struct = struct.Struct(f"{e}BBBHHHhh") # 13 bytes
        self.pir_fixed_struct = struct.Struct(f"{e}BB")     # 2 bytes
        self.mir_fixed_struct = struct.Struct(f"{e}II")     # 8 bytes
        self.limits_struct = struct.Struct(f"{e}ff")        # 8 bytes
    
    def _detect_endian(self, first_4: bytes) -> str:
        """Determine file endianness from the FAR record length."""
        l_be = struct.unpack(">H", first_4[:2])[0]
        return ">" if l_be == 2 else "<"
    
    def get_col_key(self, tn: int) -> str:
        """Get or create a unique column key for a test number."""
        if tn not in self.key_from_testnum:
            k = f"TEST_{tn}"
            self.key_from_testnum[tn] = k
            self.col_key_order.append(k)
            self.testnum_by_key[k] = tn
        return self.key_from_testnum[tn]

    def parse(self, data: Optional[bytes] = None) -> None:
        """Main entry point for parsing."""
        if data:
            self.endian = self._detect_endian(data[:4])
            self._init_structs()
            self._parse_buffer(data)
        elif self.stream:
            first_4 = self.stream.read(4)
            if not first_4: return
            self.endian = self._detect_endian(first_4)
            self._init_structs()
            
            # Process first record
            rec_len, rec_typ, rec_sub = self.header_struct.unpack(first_4)
            body = self.stream.read(rec_len)
            self._process_rec_body(rec_len, rec_typ, rec_sub, body, 0)
            
            self._parse_stream()
        
        # Determine Scale Factor (Heuristic)
        if self.testnum_by_key:
            min_tn = min(self.testnum_by_key.values())
            self.scale = 100 if min_tn < 100000 else 100000
            for k, tn in self.testnum_by_key.items():
                self.tcode_by_key[k] = testnum_to_display_id(tn, self.scale)

    def _parse_buffer(self, data: bytes) -> None:
        """High-performance buffer parser using direct memory access."""
        offset = 0
        total_len = len(data)
        unpack_header = self.header_struct.unpack_from
        
        while offset < total_len:
            if total_len - offset < 4: break
            rec_len, rec_typ, rec_sub = unpack_header(data, offset)
            offset += 4
            if total_len - offset < rec_len: break
            self._process_rec_body(rec_len, rec_typ, rec_sub, data, offset)
            offset += rec_len

    def _parse_stream(self) -> None:
        """Stream-based parser for large files."""
        if not self.stream: return
        read = self.stream.read
        unpack_header = self.header_struct.unpack
        
        while True:
            h_bytes = read(4)
            if len(h_bytes) < 4: break
            rec_len, rec_typ, rec_sub = unpack_header(h_bytes)
            body = read(rec_len)
            if len(body) < rec_len: break
            self._process_rec_body(rec_len, rec_typ, rec_sub, body, 0)

    def _process_rec_body(self, rec_len: int, rec_typ: int, rec_sub: int, data: Union[bytes, Any], offset: int) -> None:
        """Internal router for record processing."""
        # 1. PTR (Parametric Test Record)
        if rec_typ == 15 and rec_sub == 10:
            self._handle_ptr(data, offset, rec_len)
        # 2. PRR (Part Results Record)
        elif rec_typ == 5 and rec_sub == 20:
            self._handle_prr(data, offset, rec_len)
        # 3. PIR (Part Information Record)
        elif rec_typ == 5 and rec_sub == 10:
            self._handle_pir(data, offset)
        # 4. MIR (Master Information Record)
        elif rec_typ == 1 and rec_sub == 10:
            self._handle_mir(data, offset, rec_len)

    def _handle_ptr(self, data: Union[bytes, Any], offset: int, rec_len: int) -> None:
        """Process PTR records."""
        self.ptr_cnt += 1
        try:
            tn, head, site, tflg, pflg, res = self.ptr_fixed_struct.unpack_from(data, offset)
            key = (head, site)
            if key not in self.active_parts:
                self.active_parts[key] = {"tests": {}, "fail_tests": [], "alarm_tests": [], "alarm_ids": set()}
            
            act = self.active_parts[key]
            ck = self.get_col_key(tn)
            is_fail = (tflg & 0x80) != 0
            is_alarm = (tflg & 0x01) != 0

            # Lazy metadata extraction
            if tn not in self.seen_testnums:
                self.seen_testnums.add(tn)
                curr = offset + 12
                limit = offset + rec_len
                txt, curr = read_str_safe(data, curr, limit)
                aid, curr = read_str_safe(data, curr, limit)
                
                lo, hi, units = None, None, ""
                if curr + 12 <= limit:
                    curr += 4 # Skip OptFlag
                    lo, hi = self.limits_struct.unpack_from(data, curr)
                    curr += 8
                    units, _ = read_str_safe(data, curr, limit)
                
                if txt: self.best_name_by_key[ck] = txt
                if units: self.units_by_key[ck] = units
                self.lo_limit_by_key[ck] = lo
                self.hi_limit_by_key[ck] = hi
            
            act["tests"][ck] = res
            if is_fail: act["fail_tests"].append(tn)
            if is_alarm: act["alarm_tests"].append(tn)
        except Exception:
            pass

    def _handle_prr(self, data: Union[bytes, Any], offset: int, rec_len: int) -> None:
        """Process PRR records and commit rows."""
        try:
            head, site, _, _, hb, sb, x, y = self.prr_fixed_struct.unpack_from(data, offset)
            key = (head, site)
            if key not in self.active_parts: return

            act = self.active_parts[key]
            # Device sequence
            self.site_counters[site] = self.site_counters.get(site, 0) + 1
            dev_id = self.site_counters[site]

            # Coordinate resolution
            def get_val(patterns):
                for ck, val in act["tests"].items():
                    name = self.best_name_by_key.get(ck, "").upper()
                    if any(p in name for p in patterns): return val
                return None

            final_x, final_y = x, y
            if (x == 0 and y == 0) or (x == -32768):
                vx = get_val(["X DIE LOCATION", "X DIE", "X_DIE"])
                vy = get_val(["Y DIE LOCATION", "Y DIE", "Y_DIE"])
                if vx is not None: final_x = vx
                if vy is not None: final_y = vy

            def clean_c(v):
                if v == -32768 or v is None: return ""
                try: return int(v) if float(v).is_integer() else v
                except: return v

            row = act["tests"]
            row.update({
                "Device #": dev_id, "Bin": sb if sb != 65535 else hb, "Site": site,
                "X": clean_c(final_x), "Y": clean_c(final_y),
                "Fails": act["fail_tests"], "Alarms": act["alarm_tests"]
            })
            self.rows.append(row)
            del self.active_parts[key]
        except Exception:
            pass

    def _handle_pir(self, data: Union[bytes, Any], offset: int) -> None:
        """Process PIR records."""
        try:
            head, site = self.pir_fixed_struct.unpack_from(data, offset)
            self.active_parts[(head, site)] = {"tests": {}, "fail_tests": [], "alarm_tests": [], "alarm_ids": set()}
        except: pass

    def _handle_mir(self, data: Union[bytes, Any], offset: int, rec_len: int) -> None:
        """Process MIR records."""
        try:
            st, et = self.mir_fixed_struct.unpack_from(data, offset)
            self.file_creation_date = fmt_epoch(st if st > 0 else et)
            
            curr, limit = offset + 15, offset + rec_len
            lot, curr = read_str_safe(data, curr, limit)
            ptyp, curr = read_str_safe(data, curr, limit)
            node, curr = read_str_safe(data, curr, limit)
            tstr, curr = read_str_safe(data, curr, limit)
            job, _ = read_str_safe(data, curr, limit)
            self.test_program = (job or ptyp or lot or "Unknown").strip()
        except: pass


# =========================
# OUTPUT WRITING
# =========================
def write_output(parser: STDFParser, output_path: Path, input_name: str) -> None:
    """Generate result CSV and standard limits CSV."""
    fixed_cols = ["Device #", "Bin", "Site", "X", "Y", "Fails", "Alarms"]
    sorted_keys = sorted(parser.col_key_order, key=lambda k: parser.testnum_by_key.get(k, 999999))
    
    # Write Main Results
    try:
        with output_path.open("w", encoding="utf-8", newline="", buffering=1024*1024) as f:
            w = csv.writer(f)
            w.writerow([f"Created from DLOG: {input_name}"])
            w.writerow([f"Test Program: {parser.test_program}"])
            w.writerow([f"File Creation Date: {parser.file_creation_date}"])
            w.writerow([""] * len(fixed_cols) + [parser.tcode_by_key.get(k, "") for k in sorted_keys])
            w.writerow([""] * len(fixed_cols) + [parser.units_by_key.get(k, "") for k in sorted_keys])
            w.writerow(fixed_cols + [parser.best_name_by_key.get(k, k) for k in sorted_keys])
            
            # Use per-site counters for sequential numbering on write
            counters: Dict[int, int] = {}
            for row in parser.rows:
                s = row["Site"]
                counters[s] = counters.get(s, 0) + 1
                
                row_vals = []
                for c in fixed_cols:
                    val = row.get(c, "")
                    if c == "Device #": val = counters[s]
                    elif c in ("Fails", "Alarms") and isinstance(val, list):
                        val = ",".join(testnum_to_display_id(tn, parser.scale) for tn in val)
                    row_vals.append(val)
                row_vals.extend([row.get(k, "") for k in sorted_keys])
                w.writerow(row_vals)
                
        # Write Limits
        lim_dir = output_path.parent / "limit"
        lim_dir.mkdir(exist_ok=True)
        lim_path = lim_dir / f"{output_path.stem}_limits.csv"
        with lim_path.open("w", encoding="utf-8", newline="") as f2:
            w2 = csv.writer(f2)
            w2.writerow(["Test Number", "Test Description", "Units", "STDF Min", "STDF Max"])
            for k in sorted_keys:
                tn = parser.testnum_by_key[k]
                w2.writerow([
                    testnum_to_display_id(tn, parser.scale),
                    parser.best_name_by_key.get(k, k),
                    parser.units_by_key.get(k, ""),
                    fmt_limit(parser.lo_limit_by_key.get(k)),
                    fmt_limit(parser.hi_limit_by_key.get(k))
                ])
    except Exception as e:
        logger.error(f"Failed to write output: {e}")


# =========================
# EXECUTION
# =========================
def decrypt_file(input_path: Path, output_path: Path) -> bool:
    """Standalone file decryption logic."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = None
    
    try:
        if input_path.suffix.lower() == ".gz":
            with gzip.open(input_path, "rb") as f:
                parser = STDFParser()
                parser.parse(data=f.read())
        elif zipfile.is_zipfile(input_path):
            temp_dir = Path(tempfile.mkdtemp(prefix="stdf_zip_"))
            with zipfile.ZipFile(input_path) as zf:
                for name in zf.namelist():
                    if any(name.lower().endswith(ext) for ext in CANDIDATE_EXTS):
                        zf.extract(name, temp_dir)
                        with open(temp_dir / name, "rb") as f:
                            parser = STDFParser()
                            parser.parse(data=f.read())
                        break
        else:
            with open(input_path, "rb") as f:
                parser = STDFParser(f)
                parser.parse()
        
        write_output(parser, output_path, input_path.name)
        return True
    except Exception as e:
        logger.error(f"Decryption error for {input_path.name}: {e}")
        return False
    finally:
        if temp_dir: shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    if len(sys.argv) > 2:
        decrypt_file(Path(sys.argv[1]), Path(sys.argv[2]))
        return

    logger.info("Auto-discovery mode starting...")
    script_dir = Path(__file__).parent
    ds_root = script_dir / "dataset"
    if not ds_root.exists(): return

    tasks = []
    for root, _, files in os.walk(ds_root):
        rp = Path(root)
        if not any(p.lower() in ("trim&probe", "t&p") for p in rp.parts): continue
        
        for f in files:
            if not any(f.lower().endswith(ext) for ext in CANDIDATE_EXTS): continue
            if "_limits" in f or "_Decrypted" in root: continue
            
            inp = rp / f
            parts = list(inp.parts)
            try:
                idx = next(i for i, p in enumerate(parts) if p.lower() in ("trim&probe", "t&p"))
                parts[idx] = parts[idx] + "_Decrypted"
                out = Path(*parts[:-1]) / (f + ".csv")
                tasks.append((inp, out))
            except: pass

    if tasks:
        with concurrent.futures.ThreadPoolExecutor() as exe:
            futures = [exe.submit(decrypt_file, i, o) for i, o in tasks]
            concurrent.futures.wait(futures)
    logger.info("Auto-discovery decryption complete.")

if __name__ == "__main__":
    main()
