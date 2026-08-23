
import streamlit as st
import pandas as pd
import re
import json
import sys
import os
import numpy as np
import time
import threading
import subprocess
import asyncio
import zipfile
import tempfile
import shutil
import hashlib
import html
import plotly.graph_objects as go
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx  # type: ignore
except ImportError:
    # Fallback for older Streamlit versions or specific build environments.
    from streamlit.scriptrunner import add_script_run_ctx, get_script_run_ctx  # type: ignore

# Streamlit and Playwright on Windows often default to SelectorEventLoop, which does not support subprocesses.
if sys.platform == 'win32':
    import warnings
    # The Proactor deprecation warning is suppressed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass # The policy may have already been set.

# "Event loop is closed" errors are suppressed during shutdown.
if sys.platform == "win32":
    # Modifying asyncio.base_events.BaseEventLoop._check_closed to reduce noise is considered risky.
    # Instead, it is left as is, but the main runner is wrapped.
    pass

# --- The current directory is added to the path to import sibling modules. ---
import sys
from pathlib import Path

# If the application is packaged by PyInstaller, the directory from which the .exe is launched is used.
# Otherwise, the directory of the app.py file is used.
if getattr(sys, 'frozen', False):
    # CURRENT_DIR is the root where the .exe resides (used for dataset/logs)
    CURRENT_DIR = Path(sys.executable).parent.resolve()
    
    # INTERNAL_DIR is where the bundled data lives in --onedir mode
    INTERNAL_DIR = CURRENT_DIR / "_internal"
    
    # Playwright browsers are bundled inside the internal folder
    if (INTERNAL_DIR / "ms-playwright").exists():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(INTERNAL_DIR / "ms-playwright")
    else:
        # Fallback for --onefile mode where files are in the same temp root
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(CURRENT_DIR / "ms-playwright")
else:
    CURRENT_DIR = Path(__file__).parent.resolve()


sys.path.append(str(CURRENT_DIR))

# --- Import Pipeline Scripts --- 
# Are deferred and lazily loaded within the run_pipeline steps to significantly improve
# The startup and hot-reload latency of the Streamlit web application.


# --- Config ---
from utils import (
    log, get_tester_family_for_generic, get_generics_by_tester,
    get_all_generics_from_map, select_decryptor, TESTER_MAPPING_FILE,
    explode_and_collect_data_files, _is_combo_folder_name,
    shorten_wafer_list, infer_combo_folder_name, extract_fiscal_year
)
import utils
DATASET_ROOT = CURRENT_DIR / "dataset"
STATUS_FILE = CURRENT_DIR / "pipeline_status.json"

# --- UI Setup ---
st.set_page_config(page_title="Explainable AI-Driven Wafer Reliability Risk Gate", layout="wide")

# Inject Custom CSS for Sticky Header and Layout Compaction
st.markdown(
    """
    <style>
    /* Container adjustment for sticky header */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Improved sticky header that works within Streamlit's block container */
    .sticky-header {
        position: -webkit-sticky;
        position: sticky;
        top: 0px; 
        background-color: white;
        z-index: 1000;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        margin-top: 1rem; /* Pull back to the very top */
        margin-left: -2rem;
        margin-right: -2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        border-bottom: 2px solid #F0f2f6;
    }
    .sticky-header h1 {
        padding: 0 !important;
        margin: 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply sticky header class to the title using a container
st.markdown('<div class="sticky-header"><h1>Explainable AI-Driven Wafer Reliability Risk Gate</h1></div>', unsafe_allow_html=True)

# --- Session State ---
if "logs" not in st.session_state:
    st.session_state.logs = []
if "generic_results" not in st.session_state:
    st.session_state.generic_results = {}
if "processing" not in st.session_state:
    st.session_state.processing = False
if "pipeline_metrics" not in st.session_state:
    st.session_state.pipeline_metrics = {}
if "generic_logs" not in st.session_state:
    st.session_state.generic_logs = {}
if "pipeline_finished" not in st.session_state:
    st.session_state.pipeline_finished = False

# --- Helper Functions ---
# (Imports moved to Config section for consistency)

# --- Helper Functions ---
def format_time(seconds):
    if seconds is None or seconds < 0: return "Calculating..."
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"

def free_up_onedrive_space(folder_path):
    """
    Directs OneDrive to 'Free up space' for a specific folder.
    The Windows 'attrib +U -P' command is utilized.
    +U: Online-only attribute
    -P: Clear 'Always keep on this device' (unpin)
    """
    if not folder_path: return
    p = Path(folder_path)
    if not p.exists(): return
    
    try:
        import subprocess
        log(f"Triggering OneDrive 'Free up space' for: {p.name}...", "INFO")
        # The /s and /d flags are used to handle all files and subdirectories.
        subprocess.run(["attrib", "+U", "-P", "/s", "/d", str(p) + "\\*"], 
                       capture_output=True, check=True, shell=True)
        log(f"Cleanup command sent for {p.name}.", "OK")
    except Exception as e:
        log(f"Failed to free up space for {p.name}: {e}", "WARN")

def load_root_cause_recommendations() -> str:
    path = Path(CURRENT_DIR) / "root_cause_recommendations.txt"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return (
        "1. check if known failure\n"
        "2. retest to verify issue (golden wafer/correlation unit/swap hardware)\n"
        "3. identify shifted test (maybe one or more)\n"
        "4. create summary report (statistic)"
    )

def get_root_cause_html() -> str:
    rec_content = load_root_cause_recommendations()
    lines = rec_content.split('\n')
    li_items = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('.', 1)
        if len(parts) == 2 and parts[0].strip().isdigit():
            step_num = parts[0].strip()
            step_text = parts[1].strip()
            
            # Map standard steps to explanations and icons
            if "known failure" in step_text.lower():
                title = "Check if known failure"
                desc = "Audit historical failure logs and matching signature databases."
                icon = "🔍"
                badge_style = "background-color: #e3f2fd; color: #0d47a1; border: 1px solid #bbdefb;"
            elif "retest" in step_text.lower():
                title = "Retest to verify issue"
                desc = "Rerun with Golden Wafer, check Correlation Unit, or swap hardware."
                icon = "🔄"
                badge_style = "background-color: #fff3e0; color: #e65100; border: 1px solid #ffe0b2;"
            elif "shifted test" in step_text.lower():
                title = "Identify shifted test"
                desc = "Analyze SPC and distribution charts below to isolate the parameter drift."
                icon = "🎯"
                badge_style = "background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2;"
            elif "summary report" in step_text.lower():
                title = "Create summary report"
                desc = "Compile statistical summaries of the current lot vs golden baselines."
                icon = "📊"
                badge_style = "background-color: #e8f5e9; color: #1b5e20; border: 1px solid #c8e6c9;"
            else:
                title = step_text
                desc = ""
                icon = "📋"
                badge_style = "background-color: #f5f5f5; color: #333; border: 1px solid #e0e0e0;"
                
            li_items += f"""<div style="display: flex; margin-bottom: 12px; font-family: sans-serif; align-items: flex-start;">
<div style="flex-shrink: 0; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 15px; margin-right: 12px; {badge_style}">
{icon}
</div>
<div style="text-align: left;">
<div style="font-weight: 600; font-size: 14px; color: #2c3e50;">{step_num}. {title}</div>
{f'<div style="font-size: 12.5px; color: #5f6c7d; margin-top: 2px;">{desc}</div>' if desc else ''}
</div>
</div>"""
        else:
            li_items += f"""<div style="padding: 6px 12px; background-color: #f8f9fa; border-radius: 4px; border-left: 3px solid #6c757d; margin-bottom: 8px; font-size: 13px; color: #495057; text-align: left;">
{line}
</div>"""
            
    html_content = f"""<div style="background-color: #fffaf0; border: 1px solid #ffd8a8; border-left: 5px solid #ff922b; padding: 18px; border-radius: 8px; margin: 15px 0; box-shadow: 0 2px 5px rgba(0,0,0,0.02); text-align: left;">
<div style="display: flex; align-items: center; margin-bottom: 14px;">
<span style="font-size: 20px; margin-right: 10px;">⚙️</span>
<h4 style="color: #e65100; margin: 0; font-size: 16px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Root Cause Recommendation Engine</h4>
</div>
<p style="margin: 0 0 14px 0; font-size: 13.5px; color: #495057; font-weight: 500;">Dynamic troubleshooting actions for warned test parameter:</p>
<div style="margin-left: 4px;">
{li_items}
</div>
</div>"""
    return html_content.replace("\n", "")

def render_dashboard(metrics, container):
    """Render the dashboard metrics into the given container."""
    # Simplified Dashboard: Generic | Total Elapsed | Total ETA
    c1, c2, c3 = container.columns([1, 1, 1])
    c1.markdown(f"### Processing: `{metrics.get('generic', 'Unknown')}`")
    c1.caption(f"Batch Progress: {metrics.get('progress_str', 'N/A')}")
    
    c2.metric("Total Elapsed Time", metrics.get('elapsed_str', "0s"))
    c3.metric("Est. Total Time Remaining", metrics.get('eta_str', "N/A"))

def render_logs(logs, container):
    """Render logs into the given container with rich formatting (chronological)."""
    # Optimization: Render all logs as a single HTML block to prevent WebSocket flooding and UI lag.
    DISPLAY_LIMIT = 100
    total_logs = len(logs)
    
    if total_logs > DISPLAY_LIMIT:
        display_logs = logs[-DISPLAY_LIMIT:]
    else:
        display_logs = logs
        
    # Optional: Reverse to show latest at top (User preference)
    display_logs = list(reversed(display_logs))
        
    html_lines = []
    # Changed: Always show the record count as requested.
    record_count_text = f"Showing newest {min(DISPLAY_LIMIT, total_logs)} of {total_logs} logs (Latest at top)."
    html_lines.append(f"<div style='color: #888; font-size: 0.8em; margin-bottom: 10px;'>{record_count_text}</div>")
        
    for l in display_logs:
        if any(x in l for x in ["[ERR]", "[ERROR]","[FAIL]"]):
            color = "#Ff4b4b"
            bg = "#Ff4b4b1a"
            border = "1px solid #Ff4b4b4d"
        elif any(x in l for x in ["[WARN]", "[WARNING]"]):
            color = "#Ffa421"
            bg = "#Ffa4211a"
            border = "1px solid #Ffa4214d"
        elif "[OK]" in l:
            color = "#21c354"
            bg = "#21c3541a"
            border = "1px solid #21c3544d"
        elif "[STEP]" in l or "Pipeline Finished!" in l:
            # Use a more distinctive color for Step milestones and completion
            color = "#9966ff" # Purple for milestones
            bg = "#9966ff1a"
            border = "1px solid #9966ff4d"
            if "Pipeline Finished!" in l:
                color = "#21c354" # Green for final finish
                bg = "#21c3541a"
                border = "1px solid #21c3544d"
        else:
            # Force full visibility using theme color variable
            color = "var(--text-color) !important"
            bg = "transparent"
            border = "1px solid transparent"
            
        # White-space: pre-wrap is used to preserve formatting for tables (e.g. from ML_Train_Model).
        html_lines.append(f"<div style='color: {color}; background-color: {bg}; border: {border}; padding: 6px 10px; border-radius: 4px; margin-bottom: 4px; font-family: monospace; font-size: 13px; word-wrap: break-word; line-height: 1.4; white-space: pre-wrap;'>{l}</div>")
        
    container.markdown("".join(html_lines), unsafe_allow_html=True)


def normalize_lot_id(lot_id: str) -> str:
    """Removes suffixes like .1, .2 from Lot ID for consistent matching."""
    if not lot_id: return ""
    return re.sub(r'\.[a-zA-Z0-9]+$', '', str(lot_id)).strip()

def find_result_excel(generic: str) -> Path | None:
    """Dynamically locates Result_{generic}.xlsx in root or dataset hierarchy."""
    # Check both root and dataset folder
    search_dirs = [CURRENT_DIR]
    
    for base in search_dirs:
        if not base.exists(): continue
        # First check the direct expected path to avoid expensive walk
        for family_dir in base.iterdir():
            if family_dir.is_dir():
                target = family_dir / generic / f"Result_{generic}.xlsx"
                if target.exists(): return target
                
                # Check with generic_suffixed folders
                for gen_dir in family_dir.iterdir():
                    if gen_dir.is_dir() and gen_dir.name.startswith(f"{generic}_"):
                        target = gen_dir / f"Result_{generic}.xlsx"
                        if target.exists(): return target

        # Fallback to broader walk if not found in standard paths
        for root, dirs, files in os.walk(base):
            p_root = Path(root)
            if ".gemini" in p_root.parts or ".git" in p_root.parts: continue
            
            if p_root.name == generic or p_root.name.startswith(f"{generic}_"):
                for r, d, f in os.walk(p_root):
                    if f"Result_{generic}.xlsx" in f:
                        return Path(r) / f"Result_{generic}.xlsx"
    return None

def parse_test_number(test_str: str) -> tuple:
    """Extracts numeric parts from test string like 'T1.10' for sorting."""
    match = re.search(r'T(\d+)(?:\.(\d+))?', test_str)
    if match:
        major = int(match.group(1))
        minor = int(match.group(2)) if match.group(2) else 0
        return (major, minor)
    return (999999, 999999)


class GenericUITracker:
    def __init__(self, generic_name, container):
        self.generic = generic_name
        self.container = container
        self.logs = []
        self.start_time = None
        self.status_ph = None
        self.log_ph = None
        self.time_ph = None
        self.progress_ph = None
        self.step_status_ph = None
        self.expander = None
        self.setup_ui("Pending", expanded=False)

    def setup_ui(self, status="Pending", expanded=False):
        # An icon is determined based on the status.
        icon = "⏳"
        if status == "Done": icon = "✅"
        elif status == "Error" or status == "Failed": icon = "❌"
        elif "Running" in status or "Processing" in status: icon = "🔄"
        
        label = f"{icon} {self.generic} - {status}"
        
        # The entire expander block is re-rendered to update the header.
        with self.container.container():
            self.expander = st.expander(label, expanded=expanded)
            with self.expander:
                c1, c2 = st.columns([3, 1])
                self.status_ph = c1.empty()
                self.time_ph = c2.empty()
                self.status_ph.write("Waiting to start..." if status == "Pending" else f"**Status:** {status}")
                
                elapsed_str = "0s"
                if self.start_time:
                    elapsed = time.time() - self.start_time
                    elapsed_str = format_time(elapsed)
                self.time_ph.write(f"⏱ {elapsed_str}")
                
                self.progress_ph = st.empty()
                st.write("---")
                # Moved status text here to replace the 'Logs' caption per user request
                self.step_status_ph = st.empty()
                self.log_ph = st.empty()
                if self.logs:
                    # Logs are re-rendered in the new placeholder.
                    log_text = "\n".join(self.logs)
                    self.log_ph.text_area("Log Output", value=log_text, height=200, key=f"log_{self.generic}_{status}")

    def start(self):
        self.start_time = time.time()
        # The label is toggled to "Running" and expanded.
        self.setup_ui(status="Running", expanded=True)
        self.update_status("Starting...")

    def update_status(self, status_text, active_task="", expanded=None):
        if expanded is not None:
            self.setup_ui(status=status_text, expanded=expanded)
            return

        # The sub-status is constructed without re-rendering the expander header.
        self.status_ph.markdown(f"**Status:** {status_text}")
        if active_task:
            self.status_ph.caption(f"Current Task: {active_task}")
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.time_ph.write(f"⏱ {format_time(elapsed)}")

    def update_progress(self, frac):
        """Update the progress bar within the tracker."""
        if self.progress_ph:
            self.progress_ph.progress(float(frac))

    def update_step_status(self, text):
        """Update the step-specific info text (e.g. ETA)."""
        if self.step_status_ph:
            self.step_status_ph.markdown(text)

    def log(self, msg, level="INFO"):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] [{level}] {msg}"
        self.logs.append(entry)
        try:
            with self.log_ph.container():
                 # The logs are reversed to display the latest entries at the top.
                 log_text = "\n".join(reversed(self.logs))
                 st.text_area("Log Output", value=log_text, height=200, key=f"log_{self.generic}")
        except Exception:
            pass

    def finish(self, success=True):
        elapsed = time.time() - (self.start_time or time.time())
        status = "Done" if success else "Error"
        # The block is re-rendered a final time with the terminal status and collapsed.
        self.setup_ui(status=status, expanded=False)
        self.time_ph.write(f"🏁 Final Time: {format_time(elapsed)}")


def run_pipeline(generics_list, dashboard_ph, logs_ph, status_container, 
                 force_decryptor=None, skip_existing=False, 
                 combine_wafer_data=True, run_feature_extraction_step=True, 
                 run_model_training_step=True, cleanup_after_generic=True):
    # Set dynamic LOG_FILE for this pipeline run
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    utils.LOG_FILE = CURRENT_DIR / f"pipeline_execution_{timestamp}.log"
    
    st.session_state.processing = True
    st.session_state.logs = []
    if utils.LOG_FILE.exists(): 
        try: os.remove(utils.LOG_FILE)
        except: pass
        
    # The status of each generic is tracked.
    st.session_state.generic_results = {}
    if STATUS_FILE.exists():
        try: os.remove(STATUS_FILE)
        except: pass

    # Ensure task messages have a placeholder within the status container
    with status_container:
        task_ui_container = st.container() 
        step_msg_ph = task_ui_container.empty()
    
    # --- External logs are redirected using a stable implementation. ---
    ctx = get_script_run_ctx()
    
    # Throttling state is maintained for UI updates.
    state = {
        "last_update": 0,
        "lock": threading.Lock()
    }
    UPDATE_INTERVAL = 0.5 # Seconds

    def update_log_ui(force=False):
        now = time.time()
        if force or (now - state["last_update"] > UPDATE_INTERVAL):
            if state["lock"].acquire(blocking=False):
                try:
                    # A UI update is only attempted if a valid script context is available.
                    # (this prevents errors if the background thread exceeds the session duration).
                    if get_script_run_ctx():
                        render_logs(st.session_state.logs, logs_ph)
                        state["last_update"] = now
                except Exception:
                    # WebSocketClosedError or other UI synchronization issues are handled silently.
                    pass
                finally:
                    state["lock"].release()

    # The default log function is overridden to also update the UI.
    original_internal_log = globals()['log'] # The original log function is retrieved.
    def internal_log_with_ui_update(msg, level="INFO"):
        original_internal_log(msg, level) # The original log is called to append to session_state.logs.
        
        # Intercept Warnings and Errors for the accuracy summary
        if "pipeline_state" in locals() and pipeline_state.get("current_generic"):
            curr_gen = pipeline_state["current_generic"]
            if any(x in level.upper() for x in ["WARN", "ERR", "ERROR", "FAIL"]):
                if curr_gen not in st.session_state.generic_logs:
                    st.session_state.generic_logs[curr_gen] = []
                st.session_state.generic_logs[curr_gen].append(msg)
                
        update_log_ui()
        
    globals()['log'] = internal_log_with_ui_update # The global log function is replaced.

    # Functions for lazy-loaded modules are patched.
    def redirected_log_info(msg):
        if ctx and not get_script_run_ctx(): add_script_run_ctx(threading.current_thread(), ctx)
        internal_log_with_ui_update(f" {msg}", "INFO")

    def redirected_log_error(msg):
        if ctx and not get_script_run_ctx(): add_script_run_ctx(threading.current_thread(), ctx)
        internal_log_with_ui_update(f"[ERR] {msg}", "ERR")

    total_generics = len(generics_list)
    log("Starting pipeline run...", "INFO")
    update_log_ui(force=True)
    
    start_pipeline_time = time.time()
    
    # The state to track the current generic being processed is initialized.
    pipeline_state = {
        "current_generic": None,
        "trackers": {}
    }

    # Trackers are initialized.
    with status_container:
        for g in generics_list:
            pipeline_state["trackers"][g] = GenericUITracker(g, st.empty())

    # The session temporary directory is created.
    session_temp_dir = Path(tempfile.mkdtemp(prefix="FYP_Session_"))
    log(f"Created temporary processing directory: {session_temp_dir}")

    
    try:
        for idx, generic in enumerate(generics_list):
            pipeline_state["current_generic"] = generic
            tracker = pipeline_state["trackers"][generic]
            tracker.start()
            
            try:
                # --- A sub-folder is created for each generic to isolate its logic. ---
                # This ensures that each generic has its own workspace and avoids issues with redundant synchronization.
                generic_temp_dir = session_temp_dir / generic
                generic_temp_dir.mkdir(parents=True, exist_ok=True)
                log(f"Isolated workspace: {generic_temp_dir}", "INFO")

                all_temp_dirs = [] # All temporary directories for this generic are tracked.

                # The initial metrics update is performed.
                def update_dashboard(frac_of_current_item=0.0):
                    pipe_elapsed = time.time() - start_pipeline_time
                    pipe_prog = (idx + frac_of_current_item) / total_generics
                    
                    if pipe_prog > 0:
                        total_est = pipe_elapsed / pipe_prog
                        total_rem_est = total_est - pipe_elapsed
                        total_eta_s = format_time(total_rem_est)
                    else:
                        total_eta_s = "Calculating..."
                    
                    metrics = {
                        "generic": generic,
                        "progress_str": f"{idx+1}/{total_generics}",
                        "elapsed_str": format_time(pipe_elapsed),
                        "eta_str": total_eta_s if pipe_prog < 1.0 else "0s"
                    }
                    st.session_state.pipeline_metrics = metrics
                    render_dashboard(metrics, dashboard_ph)
                    tracker.update_status(f"Processing ({min(100, int(frac_of_current_item*100))}%)")
                
                update_dashboard(0.0)
                
                def make_prog_cb(task_name):
                    start_t = time.time()
                    def cb(frac, current=0, total=0):
                        if ctx and not get_script_run_ctx(): add_script_run_ctx(threading.current_thread(), ctx)
                        elapsed = time.time() - start_t
                        eta_str = format_time((elapsed / frac) - elapsed) if frac > 0 else "..."
                        tracker.update_progress(frac)
                        tracker.update_step_status(f"**{task_name}** | Elapsed: {format_time(elapsed)} | ETA: {eta_str}")
                        if total > 0:
                            tracker.update_status(f"Processing: {current}/{total} Lots", active_task=task_name)
                        update_dashboard(0.5 + (0.4 * frac))
                    return cb
                
                # --- Locate T&P_Decrypted Dataset ---
                tester_family = get_tester_family_for_generic(generic)
                if tester_family == "UNKNOWN_FAMILY":
                    tester_family = "J750"
                tracker.update_status("Locating Dataset", active_task=f"Searching dataset/{tester_family}")
                step_msg_ph.info(f"Searching local dataset for {generic}...")
                
                target_tp_dir = None
                real_ds_root = CURRENT_DIR / "dataset" / tester_family
                if real_ds_root.exists():
                    for root_dir, dirs, files in os.walk(real_ds_root):
                        dirname = Path(root_dir).name
                        if "T&P_Decrypted" in dirs and (dirname == generic or dirname.startswith(f"{generic}_")):
                            target_tp_dir = Path(root_dir) / "T&P_Decrypted"
                            log(f"Found existing decrypted data at: {target_tp_dir.relative_to(CURRENT_DIR)}", "OK")
                            break
                
                if not target_tp_dir:
                    log(f"No decrypted data found in {real_ds_root} for {generic}.", "WARN")
                
                # --- Yield Mapping (Obsolete: Yield is embedded in folder names) ---
                yield_data_path = None
                
                update_dashboard(0.5)

                # --- Wafer Data Combiner ---
                if combine_wafer_data:
                    # Guard: Only run if target_tp_dir exists
                    if target_tp_dir and target_tp_dir.exists():
                        tracker.update_status("Combining Wafer Data", active_task="Merging CSVs")
                        step_msg_ph.info(f"Running Wafer Data Combiner for {generic}...")
                        log("Running Wafer Data Combiner...", "INFO")
                        
                        # Wafer_Data_Combiner is lazily loaded.
                        import wafer_data_combiner
                        
                        try:
                            # Processing is performed using the correctly identified parent directory for T&P_Decrypted.
                            wafer_data_combiner.find_and_process_all(str(target_tp_dir.parent), log_func=log)
                        except Exception as e:
                            log(f"Combiner error: {e}", "ERR")
                        
                        # FORCE UI UPDATE AFTER COMBINER
                        log("Finished Wafer Data Combiner Phase.", "INFO")
                        update_dashboard(0.75)
                        update_log_ui(force=True)
                    else:
                        log(f"Skipping Wafer Data Combiner for {generic}: No input directory found.", "WARN")
                        
                # --- Feature Extraction (ML_Compute_Statistic.py) ---
                extracted_csv_path = None
                if run_feature_extraction_step:
                    tracker.update_status("Extracting Features", active_task="Feature Extraction")
                    step_msg_ph.info(f"Running Feature Extraction for {generic}...")
                    log("Running Feature Extraction...", "INFO")
                    
                    # The dynamically discovered T&P_Decrypted folder is used.
                    input_tp_dir = target_tp_dir
                    
                    # A check is performed to verify if target_tp_dir exists and contains data.
                    if not target_tp_dir or not target_tp_dir.exists():
                        log(f"Skipping Feature Extraction for {generic}: No decrypted data found.", "WARN")
                        extracted_csv_path = None
                    else:
                        # Inspect directory
                        try:
                            has_data = any(target_tp_dir.iterdir())
                        except:
                            has_data = False
                            
                        if not has_data:
                            log(f"Skipping Feature Extraction for {generic}: {target_tp_dir.name} is empty.", "WARN")
                            extracted_csv_path = None
                        else:
                            # The output Model folder is created only if valid data exists.
                            model_dir = target_tp_dir.parent / "Model"
                            model_dir.mkdir(parents=True, exist_ok=True)
                            
                            output_features_csv = model_dir / f"merged_features_{generic}.csv"
                            
                            # A custom info log wrapper is passed to log_func, and a progress bar is provided to progress_callback.
                            feat_cb = make_prog_cb("Feature Extraction")
                            
                            # Feature Extraction is lazily loaded.
                            import ml_compute_statistic
                            
                            try:
                                folder_data = ml_compute_statistic.run_feature_extraction(
                                    root=input_tp_dir, 
                                    out_path=output_features_csv,
                                    yield_data_path=yield_data_path,
                                    log_func=lambda msg: log(msg, "INFO"),
                                    progress_callback=feat_cb
                                )
                                if folder_data is not None and output_features_csv.exists():
                                    log(f"Feature Extraction complete. Saved to {output_features_csv.name}", "OK")
                                    extracted_csv_path = output_features_csv
                                else:
                                    log(f"Feature Extraction failed or produced no data.", "WARN")
                            except Exception as e:
                                log(f"Feature Extraction error: {e}", "ERR")
                    
                    update_dashboard(0.85)
                
                # --- Model Training (ML_Train_Model.py) ---
                if run_model_training_step and extracted_csv_path and extracted_csv_path.exists():
                    tracker.update_status("Checking dataset", active_task="Validating extracted features")
                    step_msg_ph.info(f"Validating dataset size for {generic}...")
                    
                    df_check = pd.read_csv(extracted_csv_path)
                    
                    # Ensure more than 1 row (so test_size split leaves train split not empty). 
                    if len(df_check) <= 1:
                        target_col = [c for c in df_check.columns if 'yield' in c.lower() or c.lower() in ('sb_count', 'site')]
                        log(f"Merged dataset has only {len(df_check)} row(s). Skipping Model Training to prevent split errors.", "WARN")
                        # Proceed without training
                    else:
                        is_low = len(df_check) <= 10
                        if is_low:
                            log(f"Merged dataset has ONLY {len(df_check)} rows. [WARNING] LOW DATASET - Training anyway but reliability will be restricted.", "WARN")
                            step_msg_ph.warning(f"⚠️ Low Dataset Alert: Only {len(df_check)} samples found for {generic}.")
                        
                        # Dynamic training duration estimation using the hardware calibration system
                        try:
                            from runtime_calibration import RuntimeCalibrator
                            ft_cols = [c for c in df_check.columns if str(c).startswith("FT_")]
                            meta_cols = ["y", "parent_folder", "lot_id", "file_name", "file_path"]
                            n_tgt = max(1, len(ft_cols))
                            n_feat = max(1, len([c for c in df_check.columns if c not in meta_cols and c not in ft_cols]))
                            est_sec = RuntimeCalibrator.estimate_training_time(len(df_check), n_feat, n_tgt)
                            est_min = int(est_sec // 60)
                            est_rem_sec = int(est_sec % 60)
                            est_time_str = f"{est_min}m {est_rem_sec}s" if est_min > 0 else f"{est_rem_sec}s"
                            duration_info = f" (Estimated Duration: {est_time_str})"
                        except Exception:
                            duration_info = ""
                            
                        tracker.update_status("Training Models", active_task=f"Model Training{duration_info}")
                        step_msg_ph.info(f"Running Model Training for {generic}...{duration_info}")
                        log(f"Dataset has {len(df_check)} row(s). Running Model Training...{duration_info}", "INFO")
                        
                        # Models are stored in the same Model directory.
                        model_dir = extracted_csv_path.parent
                        
                        out_res_csv = model_dir / f"model_results_{generic}.csv"
                        out_val_xlsx = model_dir / f"model_details_{generic}.xlsx"
                        info_str = f"Trained on {generic}\nExtracted Features: {extracted_csv_path.name}"
                        
                        train_cb = make_prog_cb("Model Training")
                        
                        # The ML Training Script is lazily loaded.
                        import ml_train_model
                        
                        try:
                            result = ml_train_model.run_model_training(
                                data_csv=extracted_csv_path,
                                out_res_path=out_res_csv,
                                out_val_path=out_val_xlsx,
                                info=info_str,
                                log_func=lambda msg: log(msg, "INFO"),
                                progress_callback=train_cb
                            )
                            if result:
                                joblibs, xlsx_path = result
                                log(f"Model Training complete. Top models saved. Details in {Path(xlsx_path).name}", "OK")
                            else:
                                log(f"Model Training generated no output or failed.", "WARN")
                        except Exception as e:
                            log(f"Model Training error: {e}", "ERR")
                
                update_dashboard(0.95)


                # --- Results Collection ---
                tracker.update_status("Collecting results", active_task="Finalizing")
                step_msg_ph.info(f"Collecting results for {generic}...")
                results_found = []
                # Generic_temp_dir is scanned for output files to be registered.
                for root, dirs, files in os.walk(generic_temp_dir):
                    for f in files:
                        p = Path(root) / f
                        f_lower = f.lower()
                        # .xlsx, .csv, and new .joblib model files are collected.
                        if f_lower.endswith(".xlsx") or f_lower.endswith(".csv") or f_lower.endswith(".joblib") or f_lower.endswith(".json"):
                            # Skip limit files in the results tab to avoid clutter
                            if "_limits.csv" in f_lower: continue
                            results_found.append(str(p))
                
                # Session state is updated with absolute paths. (pointing to temp for now, will sync later)
                # These paths will break after cleanup unless they are updated to the final OneDrive path.
                final_onedrive_root = CURRENT_DIR
                st.session_state.final_csvs = [
                    str(Path(rf.replace(str(generic_temp_dir), str(final_onedrive_root))))
                    for rf in results_found
                ]
                        
                # --- Sync Back to OneDrive ---
                tracker.update_status("Syncing...", active_task="Copying to OneDrive")
                step_msg_ph.info("Syncing results to OneDrive...")
                log(f"Syncing {generic} results to OneDrive...", "INFO")
                
                final_dest_root = CURRENT_DIR
                if generic_temp_dir.exists():
                    # A long path prefix is used on Windows to bypass the 260-character limit.
                    # Shutil.copytree does not always handle \\?\ consistently with various source and destination combinations,
                    # But absolute paths are attempted.
                    src = str(generic_temp_dir.resolve())
                    dst = str(final_dest_root.resolve())
                    
                    if sys.platform == 'win32':
                        if not src.startswith('\\\\?\\'): src = '\\\\?\\' + src
                        if not dst.startswith('\\\\?\\'): dst = '\\\\?\\' + dst
                        
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    log(f"Synced {generic} results to {final_dest_root}", "OK")
                
                update_dashboard(1.0)
                step_msg_ph.empty() # Clear at the end of generic
                tracker.finish(success=True)
                
                st.session_state.generic_results[generic] = "Success"
                
                # OneDrive 'Free up space' (attrib +U) is triggered only upon full success.
                if cleanup_after_generic:
                    tracker.update_status("Offloading...", active_task="OneDrive Free Up Space")
                    # The folder just synchronized in OneDrive (dataset/Family/Generic) is targeted.
                    # The parent of the identified target_tp_dir is used if it exists in CURRENT_DIR.
                    cleanup_target = None
                    if target_tp_dir and str(CURRENT_DIR) in str(target_tp_dir):
                        cleanup_target = target_tp_dir.parent
                    else:
                        # Fallback heuristic
                        inferred_fam = getattr(st.session_state, 'inferred_fam', "Auto")
                        cleanup_target = CURRENT_DIR / "dataset" / inferred_fam / generic
                    
                    if cleanup_target and cleanup_target.exists() and sys.platform == 'win32':
                        try:
                            log(f"Triggering OneDrive 'Free up space' for {cleanup_target.name}...", "INFO")
                            cmd = ["attrib", "+U", "/s", "/d", str(cleanup_target.resolve())]
                            subprocess.run(cmd, capture_output=True, text=True, check=False)
                            log(f"OneDrive space freed for {generic}.", "OK")
                        except Exception as e:
                            log(f"Failed to free OneDrive space: {e}", "WARN")

                # The status is saved after each generic is processed.
                try:
                    with open(STATUS_FILE, "w") as f:
                        json.dump(st.session_state.generic_results, f)
                except: pass
                
            except Exception as ge:
                log(f"Generic {generic} failed: {ge}", "ERR")
                tracker.finish(success=False)
                st.session_state.generic_results[generic] = f"❌ Failed: {ge}"
                try:
                    with open(STATUS_FILE, "w") as f:
                        json.dump(st.session_state.generic_results, f)
                except: pass
            finally:
                # Local Temp Cleanup for this Generic (Must happen in finally for the generic)
                try:
                    # The cleanup of extraction temporary files, which was deferred from the decryption loop, is performed.
                    if 'all_temp_dirs' in locals():
                        for td in all_temp_dirs:
                            shutil.rmtree(td, ignore_errors=True)
                    
                    shutil.rmtree(generic_temp_dir, ignore_errors=True)
                    log(f"Local isolated workspace for {generic} cleaned.", "INFO")
                except: pass
                
    finally:
        # Final Step: Extract Model Accuracies after all generics are processed
        try:
            log("Running summary extraction as a final summary step...", "INFO")
            import app_summary
            app_summary.extract_model_accuracies(generics_list=generics_list, generic_logs=st.session_state.generic_logs)
            log("Model accuracies extracted and appended to summary.", "OK")
        except Exception as e:
            log(f"app_summary failed: {e}", "ERR")

        # Cleanup Session Temp
        try:
            shutil.rmtree(session_temp_dir)
            log("Cleaned up temporary directory.", "INFO")
        except: pass
        
        st.session_state.processing = False
        st.session_state.pipeline_finished = True
        
        # Ensure ETA shows 0s after full completion
        if "pipeline_metrics" in st.session_state:
            st.session_state.pipeline_metrics["eta_str"] = "0s"
            render_dashboard(st.session_state.pipeline_metrics, dashboard_ph)
            
        log("Pipeline Finished!", "STEP")
        update_log_ui(force=True) # Final push to ensure browser shows the last logs
        st.success("Pipeline Completed!")


# --- Prediction helpers are ported from GUI.py. ---
def stable_seed(key: str) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16)

def historical_avg_yield(lot_id: str) -> float:
    rng = np.random.default_rng(stable_seed(f"HIST\n{lot_id}"))
    base = rng.uniform(0.86, 0.98)
    return float(np.clip(rng.normal(base, 0.008), 0, 1))

def classify_grade(y: float, thr: dict) -> str:
    if y >= thr["Z"]: return "Z"
    if y >= thr["A"]: return "A"
    if y >= thr["H"]: return "H"
    if y >= thr["I"]: return "I"
    if y >= thr["C"]: return "C"
    return "F"

def confidence_probability(pred_yield: float, hist_avg: float, thr: dict) -> float:
    p = 0.95
    diff = abs(pred_yield - hist_avg)
    p -= min(0.20, (diff / 0.10) * 0.15)
    thresholds = np.array([thr[k] for k in ["C", "I", "H", "A", "Z"]], dtype=float)
    dmin = float(np.min(np.abs(pred_yield - thresholds)))
    if dmin < 0.005: p -= 0.08
    elif dmin < 0.015: p -= 0.04
    return float(np.clip(p, 0.80, 0.99))

def cpk(mean: float, sigma: float, lsl: float, usl: float) -> float:
    sigma = max(float(sigma), 1e-9)
    if pd.isna(lsl) and pd.isna(usl): return np.nan
    
    cpu = (usl - mean) / (3 * sigma) if not pd.isna(usl) else np.inf
    cpl = (mean - lsl) / (3 * sigma) if not pd.isna(lsl) else np.inf
    val = float(min(cpu, cpl))
    return val if val != np.inf else np.nan

def get_readable_feature_name(fname: str) -> str:
    """Converts technical feature names (TP_t1_0__mean) to human-readable labels."""
    if not fname:
        return fname
    
    # Strip common prefixes
    readable = fname
    if readable.startswith("TP_"):
        readable = readable.replace("TP_", "")
    elif readable.startswith("FT_"):
        readable = readable.replace("FT_", "")
    elif readable.startswith("WP_"):
        return readable.replace("WP_", "").replace("_", " ").title()

    # Handle Factor labels (PCA)
    if readable.startswith("Principal Factor"):
        return readable
    if readable.startswith("Factor"):
        return readable.replace("Factor", "Principal Factor")
    
    # Handle stat suffixes
    stat_map = {
        "__mean": " (Mean)",
        "__std": " (Std)",
        "__median": " (Median)",
        "__iqr": " (IQR)",
        "__outlier_rate": " (Outlier%)",
        "__missing_rate": " (Missing%)",
        "__min": " (Min)",
        "__max": " (Max)",
        "__p01": " (1st Percentile)",
        "__p05": " (5th Percentile)",
        "__p25": " (25th Percentile)",
        "__p50": " (50th Percentile)",
        "__p75": " (75th Percentile)",
        "__p95": " (95th Percentile)",
        "__p99": " (99th Percentile)",
        "__ppm_near_LSL_pct": " (PPM near LSL %)",
        "__ppm_near_LSL_sigma": " (Sigma to LSL)",
        "__ppm_near_USL_pct": " (PPM near USL %)",
        "__ppm_near_USL_sigma": " (Sigma to USL)",
        "__psi__is_major": " (PSI Major Shift)",
        "__psi__is_minor": " (PSI Minor Shift)",
        "__psi__score": " (PSI Score)",
        "__spc__is_shift": " (SPC Shift)",
        "__spc__median_shift": " (SPC Median Shift)",
        "__spc__sigma_dist": " (SPC Sigma Distance)"
    }
    
    for suffix, label in stat_map.items():
        if readable.endswith(suffix):
            base = readable[:-len(suffix)]
            # Convert t1_0 -> T1.0
            if ": " in base:
                prefix, test = base.split(": ", 1)
                test_clean = test.upper().replace("_", ".")
                return f"{prefix} {test_clean}{label}"
            return f"{base.upper().replace('_', '.')}{label}"

    # Default fallback
    if ": " in readable:
        prefix, test = readable.split(": ", 1)
        return f"{prefix} {test.upper().replace('_', '.')}"
    
    return readable.upper().replace("_", ".")

def get_shap_feature_names(pipe, model_dir: Path, model_name: str, generic: str, fallback_cols=None):
    """Return transformed feature names mapped back to readable test parameters when possible."""
    fallback_cols = list(fallback_cols or [])
    raw_names = []
    model_input_names = []
    try:
        model_input_names = list(getattr(pipe, "feature_names_in_", []) or [])
    except Exception:
        model_input_names = []

    try:
        feature_name_path = model_dir / f"feature_names_{model_name}_{generic}.json"
        if feature_name_path.exists():
            with open(feature_name_path, "r") as f:
                meta = json.load(f)
            transformed = meta.get("transformed")
            original = meta.get("original")
            if transformed and all(re.fullmatch(r"(?:[Ff]eature|[Xx])\s*\d+", str(n)) for n in transformed) and original and len(original) == len(transformed):
                raw_names = list(original)
            else:
                raw_names = list(transformed or original or [])
    except Exception:
        raw_names = []

    if not raw_names:
        try:
            if hasattr(pipe[:-1], "get_feature_names_out"):
                raw_names = list(pipe[:-1].get_feature_names_out())
        except Exception:
            raw_names = []

    if model_input_names and (
        not raw_names
        or len(model_input_names) == len(raw_names)
        and all(re.fullmatch(r"x\d+", str(name)) for name in raw_names)
    ):
        raw_names = model_input_names

    if not raw_names:
        raw_names = fallback_cols

    return [get_readable_feature_name(str(name)) for name in raw_names]

def plot_shap_beeswarm_plotly(explanation, max_display=15, title="SHAP Impact Analysis", fallback_feature_names=None):
    """Creates an interactive Plotly version of the SHAP beeswarm plot."""
    import pandas as pd
    import plotly.express as px
    import numpy as np

    # Get values and names
    if isinstance(explanation, dict):
        shap_values = explanation.get("values", None)
        feature_names = explanation.get("feature_names", [])
        data = explanation.get("data", None)
    elif hasattr(explanation, "values"):
        shap_values = explanation.values
        try:
            feature_names = explanation.feature_names if getattr(explanation, "feature_names", None) is not None else []
        except AttributeError:
            feature_names = []
        data = explanation.data if hasattr(explanation, "data") else None
    else:
        shap_values = None
        feature_names = []
        data = None

    if not feature_names and fallback_feature_names:
        feature_names = fallback_feature_names

    if shap_values is None:
        return None

    shap_values = np.array(shap_values)

    def get_colorbar_config(vals):
        if len(vals) == 0:
            return [-1.0, 0.0, 1.0], ["Lower Yield", "Baseline", "Higher Yield"]
        t_min = float(np.nanmin(vals))
        t_max = float(np.nanmax(vals))
        if pd.isna(t_min): t_min = -1.0
        if pd.isna(t_max): t_max = 1.0
        if t_min == t_max:
            if t_min == 0:
                t_min, t_max = -1.0, 1.0
            else:
                t_min, t_max = min(t_min, 0.0) - 1.0, max(t_max, 0.0) + 1.0
        
        tick_vals = sorted(list(set([t_min, 0.0, t_max])))
        tick_text = []
        for v in tick_vals:
            if v < -1e-5:
                tick_text.append("Lower Yield")
            elif abs(v) <= 1e-5:
                tick_text.append("Baseline")
            else:
                tick_text.append("Higher Yield")
        return tick_vals, tick_text

    # 1D waterfall style (single wafer/lot) adapted to horizontal bar
    if len(shap_values.shape) == 1 or (len(shap_values.shape) == 2 and shap_values.shape[0] == 1):
        if len(shap_values.shape) == 2:
            shap_values = shap_values.flatten()
            if data is not None:
                try:
                    data = np.array(data).flatten()
                except Exception:
                    data = None

        n_vals = len(shap_values)
        f_names = list(feature_names)
        if len(f_names) < n_vals:
            f_names += [f"FEATURE {j}" for j in range(len(f_names), n_vals)]
        elif len(f_names) > n_vals:
            f_names = f_names[:n_vals]
            
        readable_names = [get_readable_feature_name(str(name)) for name in f_names]
        
        df = pd.DataFrame({
            "Test Parameter": readable_names,
            "SHAP Value": shap_values,
            "Parameter Value": data if (data is not None and len(data) == n_vals) else [np.nan] * n_vals
        })
        df["Absolute Impact"] = df["SHAP Value"].abs()
        df = df.sort_values(by="Absolute Impact", ascending=False).head(max_display)
        df = df.sort_values(by="Absolute Impact", ascending=True) # For correct top-to-bottom

        fig = px.bar(
            df, x="SHAP Value", y="Test Parameter", 
            color="SHAP Value", 
            color_continuous_scale=[(0, "#B71C1C"), (0.5, "#FFFF8D"), (1, "#1B5E20")],
            color_continuous_midpoint=0,
            title=title, 
            hover_data={"Test Parameter": True, "Parameter Value": True, "SHAP Value": ":.4f"},
            orientation="h"
        )
        
        tick_vals, tick_text = get_colorbar_config(df["SHAP Value"])
        
        fig.update_layout(
            height=250 + (30 * len(df)),
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=50),
            coloraxis_showscale=True,
            coloraxis_colorbar=dict(
                title="Yield Impact",
                tickvals=tick_vals,
                ticktext=tick_text,
                thickness=15
            )
        )
        fig.add_vline(x=0.0, line_dash="dash", line_color="gray", opacity=0.5)
        return fig

    # Global Beeswarm (2D)
    n_samples, n_features = shap_values.shape

    # Calculate global importance to select top features
    global_importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(global_importance)[::-1][:max_display]

    top_feature_names = []
    for i in top_indices:
        if i < len(feature_names):
            top_feature_names.append(get_readable_feature_name(feature_names[i]))
        else:
            top_feature_names.append(f"FEATURE {i}")

    # Melt the data for plotting
    plot_data = []
    for idx_in_top, i in enumerate(top_indices):
        fname = top_feature_names[idx_in_top]
        vals = shap_values[:, i]
        feat_vals = data[:, i] if data is not None else [np.nan] * n_samples
        
        for j in range(n_samples):
            plot_data.append({
                "Test Parameter": fname,
                "SHAP Value": vals[j],
                "Parameter Value": feat_vals[j]
            })
            
    df_plot = pd.DataFrame(plot_data)
    
    # Use Plotly Scatter plot to mimic beeswarm
    fig = px.scatter(
        df_plot, 
        x="SHAP Value", 
        y="Test Parameter", 
        color="SHAP Value",
        color_continuous_scale=[(0, "#B71C1C"), (0.5, "#FFFF8D"), (1, "#1B5E20")],
        color_continuous_midpoint=0,
        title=title,
        labels={"SHAP Value": "Yield Impact"},
        hover_data={"Test Parameter": True, "Parameter Value": True, "SHAP Value": ":.4f"}
    )
    
    # Customize aesthetic - larger marker size, no jitter on Y axis
    fig.update_traces(
        marker=dict(size=10, opacity=0.85, line=dict(width=0.5, color='white')),
    )
    
    # Alternate background shaded rectangles (zebra striping) and divider lines
    shapes = []
    for i in range(len(top_feature_names)):
        if i % 2 == 1:
            shapes.append(dict(
                type="rect",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=i - 0.5,
                y1=i + 0.5,
                fillcolor="rgba(0,0,0,0.03)",
                layer="below",
                line=dict(width=0),
            ))
            
    for i in range(len(top_feature_names) + 1):
        shapes.append(dict(
            type="line",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=i - 0.5,
            y1=i - 0.5,
            line=dict(color="rgba(0,0,0,0.08)", width=1.5),
            layer="below"
        ))
        
    tick_vals, tick_text = get_colorbar_config(df_plot["SHAP Value"])
        
    fig.update_layout(
        height=200 + (40 * len(top_indices)),
        showlegend=False,
        yaxis=dict(
            categoryorder="array",
            categoryarray=list(reversed(top_feature_names)),
            title="",
            showgrid=False,
            zeroline=False,
            range=[-0.6, len(top_feature_names) - 0.4]
        ),
        xaxis=dict(
            title="Yield Impact",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False
        ),
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=50),
        coloraxis_showscale=True,
        coloraxis_colorbar=dict(
            title="Yield Impact",
            tickvals=tick_vals,
            ticktext=tick_text,
            thickness=15
        ),
        shapes=shapes
    )
    
    # Add vertical reference line at 0 (dashed line like the reference image)
    fig.add_vline(x=0.0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

@st.cache_data(show_spinner=False)
def get_historical_inventory(generic: str):
    """Scans historical files and mappings to return a list of available Lot IDs and metadata."""
    inventory = []
    base_dataset = CURRENT_DIR / "dataset"
    
    # 1. Load Year Mapping from Excel
    excel_path = find_result_excel(generic)
    lot_to_year = {}
    if excel_path:
        try:
            df_map = pd.read_excel(excel_path)
            # Find columns: FabLotId, FiscalYear
            l_col = next((c for c in df_map.columns if "fablotid" in c.lower()), None)
            y_col = next((c for c in df_map.columns if "fiscalyear" in c.lower()), None)
            if l_col and y_col:
                for _, row in df_map.iterrows():
                    l_val = str(row[l_col]).strip()
                    y_val = str(row[y_col]).strip()
                    norm_l = normalize_lot_id(l_val)
                    lot_to_year[norm_l] = y_val
        except Exception as e:
            log(f"Error reading year map: {e}", "WARN")
    else:
        pass

    # 2. Scan for T&P_Decrypted CSVs in root or dataset/
    hist_files = []
    search_dirs = [CURRENT_DIR, CURRENT_DIR / "dataset"]
    for base in search_dirs:
        if not base.exists(): continue
        for family_dir in base.iterdir():
            if family_dir.is_dir():
                for generic_dir in family_dir.iterdir():
                    if generic_dir.is_dir() and (generic_dir.name == generic or generic_dir.name.startswith(generic + "_")):
                        tp_dir = generic_dir / "T&P_Decrypted"
                        if tp_dir.exists():
                            for root, _, files in os.walk(tp_dir):
                                for f in files:
                                    if f.lower().endswith(".csv") and not f.lower().endswith("_limits.csv"):
                                        p = Path(root) / f
                                        if p.stat().st_size > 10240:
                                            hist_files.append(p)
                                            
    # 3. Parse Metadata
    for hf in hist_files:
        # Extract yield and Lot ID from folder name
        # Folder is .../T&P_Decrypted/LOTNAME_[01-02]_92%/...
        # Or parent is the combo folder
        combo_folder = hf.parent.name
        parts = combo_folder.split('_')
        lot_id_raw = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 and parts[0] == "SYN" else parts[0]
        norm_lot = normalize_lot_id(lot_id_raw)
        
        # Extract yield
        yield_val = 0.0
        if '%' in combo_folder:
            match = re.search(r'_(\d+(?:\.\d+)?)\s*%', combo_folder)
            if match:
                yield_val = float(match.group(1))
        
        # Filter: only good yield > 80
        if yield_val < 80:
            continue
            
        # Priority: 1. Filename, 2. Combo Folder, 3. Excel Map
        year_val_file = extract_fiscal_year(hf.name)
        if year_val_file == "Unknown":
             year_val_file = extract_fiscal_year(combo_folder)
             
        year_val = year_val_file if year_val_file != "Unknown" else lot_to_year.get(norm_lot, "Unknown")
        inventory.append({
            "path": str(hf),
            "filename": hf.name,
            "lot_id": lot_id_raw,
            "norm_lot": norm_lot,
            "yield": yield_val,
            "year": year_val,
            "mtime": hf.stat().st_mtime
        })
        
    # Sort by recent
    inventory.sort(key=lambda x: x["mtime"], reverse=True)
    return inventory

def _calculate_psi(expected, actual, bins=10):
    try:
        expected = expected.dropna()
        actual = actual.dropna()
        if len(expected) == 0 or len(actual) == 0:
            return np.nan
        
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        
        if min_val == max_val:
            return 0.0

        bins_edges = np.linspace(min_val, max_val, bins + 1)
        expected_counts, _ = np.histogram(expected, bins=bins_edges)
        actual_counts, _ = np.histogram(actual, bins=bins_edges)
        
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # Replace 0 with small epsilon to avoid divide by zero
        expected_pct = np.where(expected_pct == 0, 1e-4, expected_pct)
        actual_pct = np.where(actual_pct == 0, 1e-4, actual_pct)
        
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False)
def get_ft_historical_inventory(generic: str):
    """Scans FT_Decrypted historical files and returns a list of available FT Lot IDs and metadata."""
    inventory = []
    base_dataset = CURRENT_DIR / "dataset"
    
    search_dirs = [CURRENT_DIR, base_dataset]
    for base in search_dirs:
        if not base.exists(): continue
        for family_dir in base.iterdir():
            if family_dir.is_dir():
                for generic_dir in family_dir.iterdir():
                    if generic_dir.is_dir() and (generic_dir.name == generic or generic_dir.name.startswith(generic + "_")):
                        ft_dir = generic_dir / "FT_Decrypted"
                        if ft_dir.exists():
                            for root, _, files in os.walk(ft_dir):
                                for f in files:
                                    if f.lower().endswith(".csv") and not f.lower().endswith("_limits.csv"):
                                        p = Path(root) / f
                                        if p.stat().st_size > 10240:
                                            combo_folder = p.parent.name
                                            parts = combo_folder.split('_')
                                            lot_id_raw = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 and parts[0] == "SYN" else parts[0]
                                            yield_val = 0.0
                                            if '%' in combo_folder:
                                                match = re.search(r'_(\d+(?:\.\d+)?)\s*%', combo_folder)
                                                if match:
                                                    yield_val = float(match.group(1))
                                            inventory.append({
                                                "path": str(p),
                                                "filename": f,
                                                "lot_id": lot_id_raw,
                                                "yield": yield_val,
                                                "mtime": p.stat().st_mtime
                                            })
    inventory.sort(key=lambda x: x["yield"], reverse=True)
    return inventory


@st.cache_data(show_spinner=False)
def get_actual_stats_for_lot(generic: str, lot_id: str, stage: str = "ft", probe_filename: str = None):
    """Finds and computes stats for a specific lot in FT_Decrypted or T&P_Decrypted."""
    if stage == "ft":
        inv = get_ft_historical_inventory(generic)
    else:
        inv = get_historical_inventory(generic)
        
    if not inv:
        return {}
        
    # Find matching lot
    match = None
    if probe_filename:
        # Match based on prefix before _PRB_ or _FIN_
        # Example: MT40A_00_MT40A1G8_SYN_0004_3_01_J750_PRB_... -> MT40A_00_MT40A1G8_SYN_0004_3_01_J750
        probe_f = probe_filename.upper()
        probe_prefix = probe_f.split("_PRB_")[0] if "_PRB_" in probe_f else probe_f.split("_FIN_")[0]
        
        for l in inv:
            if "filename" in l:
                f_upper = l["filename"].upper()
                f_prefix = f_upper.split("_FIN_")[0] if "_FIN_" in f_upper else f_upper.split("_PRB_")[0]
                if probe_prefix == f_prefix:
                    match = l
                    break
    
    if not match:
        match = next((l for l in inv if l["lot_id"] == lot_id), None)
    
    if not match:
        return {}
        
    try:
        from ml_compute_statistic import read_csv_dynamic_header, safe_numeric_df, normalize_colname
        df, t_nums = read_csv_dynamic_header(Path(match["path"]))
        # Filter to passing devices (Bin=1)
        n_map = {normalize_colname(c): c for c in df.columns}
        bin_col = n_map.get("bin")
        if bin_col:
            try: df = df[pd.to_numeric(df[bin_col], errors='coerce') == 1]
            except: pass
        
        num_df, _ = safe_numeric_df(df)
        col_to_tnum = {c: t_nums[i] for i, c in enumerate(df.columns) if i < len(t_nums) and t_nums[i]}
        
        params = {}
        for c in num_df.columns:
            t_num = col_to_tnum.get(c, normalize_colname(c))
            vals = pd.to_numeric(num_df[c], errors='coerce').dropna()
            if not vals.empty:
                params[t_num] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(vals.median()),
                    "min": float(vals.min()),
                    "max": float(vals.max())
                }
        return params
    except:
        return {}


@st.cache_data(show_spinner=False)
def get_actual_raw_data_for_lot(generic: str, lot_id: str, stage: str = "ft", probe_filename: str = None):
    """Finds and returns raw data columns/DataFrame for a specific lot in FT_Decrypted or T&P_Decrypted."""
    if stage == "ft":
        inv = get_ft_historical_inventory(generic)
    else:
        inv = get_historical_inventory(generic)
        
    if not inv:
        return pd.DataFrame()
        
    match = None
    if probe_filename:
        probe_f = probe_filename.upper()
        probe_prefix = probe_f.split("_PRB_")[0] if "_PRB_" in probe_f else probe_f.split("_FIN_")[0]
        
        for l in inv:
            if "filename" in l:
                f_upper = l["filename"].upper()
                f_prefix = f_upper.split("_FIN_")[0] if "_FIN_" in f_upper else f_upper.split("_PRB_")[0]
                if probe_prefix == f_prefix:
                    match = l
                    break
    
    if not match:
        match = next((l for l in inv if l["lot_id"] == lot_id), None)
        
    if not match:
        return pd.DataFrame()
        
    try:
        from ml_compute_statistic import read_csv_dynamic_header, safe_numeric_df, normalize_colname
        df, t_nums = read_csv_dynamic_header(Path(match["path"]))
        n_map = {normalize_colname(c): c for c in df.columns}
        bin_col = n_map.get("bin")
        if bin_col:
            try: df = df[pd.to_numeric(df[bin_col], errors='coerce') == 1]
            except: pass
        
        num_df, _ = safe_numeric_df(df)
        col_to_tnum = {c: t_nums[i] for i, c in enumerate(df.columns) if i < len(t_nums) and t_nums[i]}
        
        rename_map = {}
        for c in num_df.columns:
            t_num = col_to_tnum.get(c)
            norm_c = normalize_colname(c)
            rename_map[c] = t_num if t_num else norm_c
            
        return num_df.rename(columns=rename_map)
    except:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def build_golden_baseline(generic: str, stage: str = "probe", top_n: int = 3):

    from ml_compute_statistic import read_csv_dynamic_header, safe_numeric_df, normalize_colname, META_COLS_TO_EXCLUDE, TEST_NAMES_TO_EXCLUDE
    
    if stage == "ft":
        inv = get_ft_historical_inventory(generic)
    else:
        inv = get_historical_inventory(generic)
    
    if not inv:
        return None
    
    # Select top-3 highest yield lots
    top_lots = sorted(inv, key=lambda x: x["yield"], reverse=True)[:top_n]
    
    golden_dfs = []
    lot_ids = []
    tnum_to_name = {}
    for lot_info in top_lots:
        try:
            df, t_nums = read_csv_dynamic_header(Path(lot_info["path"]))
            col_to_tnum = {c: t_nums[i] for i, c in enumerate(df.columns) if i < len(t_nums) and t_nums[i]}
            
            # Filter to passing devices (Bin=1)
            n_map = {normalize_colname(c): c for c in df.columns}
            bin_col = n_map.get("bin")
            if bin_col:
                try: df = df[pd.to_numeric(df[bin_col], errors='coerce') == 1]
                except: pass
            
            num_df, _ = safe_numeric_df(df)
            
            # Rename columns to T-numbers for consistency
            col_to_tnum_local = {c: t_nums[i] for i, c in enumerate(df.columns) if i < len(t_nums) and t_nums[i]}
            rename_map = {}
            for c in num_df.columns:
                t_num = col_to_tnum_local.get(c)
                norm_c = normalize_colname(c)
                rename_map[c] = t_num if t_num else norm_c
                if t_num:
                    tnum_to_name[t_num] = c
            num_df = num_df.rename(columns=rename_map)
            
            num_df["_LOT_ID"] = lot_info["lot_id"]
            num_df["_YIELD"] = lot_info["yield"]
            golden_dfs.append(num_df)
            lot_ids.append(f"{lot_info['lot_id']} ({lot_info['yield']:.1f}%)")
        except Exception:
            pass
    
    if not golden_dfs:
        return None
    
    golden_df = pd.concat(golden_dfs, ignore_index=True)
    golden_yield = np.mean([l["yield"] for l in top_lots])
    
    # Compute per-parameter statistics
    params = {}
    for c in golden_df.columns:
        if c.startswith("_") or normalize_colname(c) in META_COLS_TO_EXCLUDE:
            continue
        if any(p in normalize_colname(c) for p in TEST_NAMES_TO_EXCLUDE):
            continue
        
        vals = pd.to_numeric(golden_df[c], errors='coerce').dropna()
        if len(vals) < 5:
            continue
        
        mean_v = float(vals.mean())
        std_v = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        median_v = float(vals.median())
        
        params[c] = {
            "mean": mean_v,
            "std": std_v,
            "median": median_v,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "count": len(vals)
        }
    
    return {
        "golden_yield": golden_yield,
        "lot_ids": lot_ids,
        "params": params,
        "raw_df": golden_df,
        "tnum_to_name": tnum_to_name
    }


def compute_golden_comparison(current_stats: dict, golden_baseline: dict, actual_stats: dict = None, label_prefix: str = "Current", limit_map: dict = None):
    """Compare current wafer per-parameter stats against the golden baseline and Spec Limits.
    
    Args:
        current_stats: {test_param: {'mean': float, 'std': float, ...}}
        golden_baseline: Output of build_golden_baseline()
        actual_stats: Optional {test_param: {'mean': float, ...}} for real FT comparison
        label_prefix: Prefix for the mean/std columns (e.g. 'Current' or 'Predicted')
        limit_map: Optional dict containing {test_param: (LSL, USL)}
    
    Returns:
        pd.DataFrame with comparison rows
    """
    if not golden_baseline or not golden_baseline.get("params"):
        return pd.DataFrame()
    
    rows = []
    golden_params = golden_baseline["params"]
    
    for param, g_stats in golden_params.items():
        c_stats = current_stats.get(param)
        if not c_stats:
            continue
        
        g_mean = g_stats["mean"]
        g_std = g_stats["std"]
        c_mean = c_stats["mean"]
        c_std = c_stats["std"]
        
        # Get spec limits (LSL / USL)
        lsl, usl = np.nan, np.nan
        if limit_map:
            from ml_compute_statistic import normalize_colname
            norm_param = normalize_colname(param)
            # Direct match
            limits = limit_map.get(param)
            if limits is None:
                # Try normalization match
                for k, v in limit_map.items():
                    if normalize_colname(k) == norm_param:
                        limits = v
                        break
            if limits is not None:
                try:
                    lsl, usl = pd.to_numeric(limits[0]), pd.to_numeric(limits[1])
                except:
                    pass
        
        # Deviation Z-score: how far is the current mean from golden mean in golden-sigma units
        dev_z = abs(c_mean - g_mean) / (g_std + 1e-9)
        
        # Status logic: Z-score thresholds + hard Spec Limits
        is_out_of_spec = False
        if pd.notna(lsl) and c_mean < lsl:
            is_out_of_spec = True
        if pd.notna(usl) and c_mean > usl:
            is_out_of_spec = True
            
        if is_out_of_spec or dev_z > 2.0:
            status = "🔴 Out of Spec"
        elif dev_z > 1.0:
            status = "⚠️ Marginal"
        else:
            status = "✅ Within Spec"
        
        row = {
            "Test Parameter": param,
            "LSL": f"{lsl:.4g}" if pd.notna(lsl) else "N/A",
            "USL": f"{usl:.4g}" if pd.notna(usl) else "N/A",
            "Golden Mean": f"{g_mean:.4g}",
            "Golden Std": f"{g_std:.4g}",
            f"{label_prefix} Mean": f"{c_mean:.4g}",
            f"{label_prefix} Std": f"{c_std:.4g}"
        }

        # Add deviation and status
        row.update({
            "Deviation (Zσ)": f"{dev_z:.2f}",
            "Status": status
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Sort: out-of-spec first
        df["_sort"] = df["Status"].map(lambda x: 0 if "🔴" in x else (1 if "⚠️" in x else 2))
        df = df.sort_values(by=["_sort", "Test Parameter"]).drop(columns=["_sort"])
    return df


@st.cache_data(show_spinner=False)
def get_real_distribution_data(generic: str, limit_map: dict, curr_df: pd.DataFrame, col_to_tnum: dict = None, selected_lots: list = None):
    """
    Historic T&P_Decrypted CSVs for the generic are located, common testing columns are extracted, and a shift analysis is performed
    comparing the current DataFrame (curr_df) with the historical DataFrame (hist_df).
    """
    if curr_df is None or curr_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(columns=["Test", "LSL", "USL"]), {}, {}
        
    from ml_compute_statistic import normalize_colname, META_COLS_TO_EXCLUDE, TEST_NAMES_TO_EXCLUDE
    
    # Selected files are loaded.
    hist_dfs = []
    if not selected_lots:
        # Fallback to inventory scan if none selected
        inv = get_historical_inventory(generic)
        selected_files = [i["path"] for i in inv[:5]]
    else:
        selected_files = selected_lots

    for hf in selected_files:
        hf = Path(hf)
        if not hf.exists():
            st.cache_data.clear()
            log(f"Detected missing cache path {hf}. Cleared Streamlit cache.", "WARN")
            continue
        try:
            from ml_compute_statistic import read_csv_dynamic_header, safe_numeric_df
            df, t_nums = read_csv_dynamic_header(hf)
            norm_cols = {normalize_colname(c): c for c in df.columns}
            
            bin_col = norm_cols.get("bin")
            if bin_col:
                try: df = df[pd.to_numeric(df[bin_col]) == 1]
                except: pass
            ndf, _ = safe_numeric_df(df)
            
            col_to_tnum_local = {c: t_nums[i] for i, c in enumerate(df.columns) if i < len(t_nums) and t_nums[i]}
            rename_map = {}
            for c in ndf.columns:
                t_num = col_to_tnum_local.get(c)
                norm_c = normalize_colname(c)
                # Use original T-number format (e.g. T1.0) to match limit_map keys
                rename_map[c] = t_num if t_num else norm_c
            
            ndf = ndf.rename(columns=rename_map)
            # Tag with Lot ID and Yield for multi-trace support
            combo_folder = hf.parent.name
            parts = combo_folder.split('_')
            lot_id_short = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 and parts[0] == "SYN" else parts[0]
            yield_match = re.search(r'_(\d+(?:\.\d+)?)\s*%', combo_folder)
            yield_str = yield_match.group(1) if yield_match else "0"
            
            # Use unique label for trace separation: Lot + Yield
            unique_label = f"{lot_id_short} ({yield_str}%)"
            ndf["_LOT_ID_LABEL"] = unique_label
            ndf["_LOT_ID"] = lot_id_short
            ndf["_YIELD"] = yield_str
            hist_dfs.append(ndf)
        except Exception as e:
            import traceback
            log(f"Error loading historical lot {hf}: {e}", "WARN")
            log(traceback.format_exc(), "WARN")
        
    tnum_to_name = {}
    if col_to_tnum:
        for name, tnum in col_to_tnum.items():
            tnum_to_name[tnum] = name

    if not hist_dfs:
        return curr_df, pd.DataFrame(), pd.DataFrame(columns=["Test", "LSL", "USL"]), {}, tnum_to_name
        
    hist_df = pd.concat(hist_dfs, ignore_index=True)
    
    # The columns of curr_df are normalized similarly to match those of hist_df.
    # T-numbers are mapped back to the current DataFrame using the identified col_to_tnum.
    curr_df_norm = curr_df.copy()
    
    # A rename map for curr_df is generated based on col_to_tnum, with a fallback to the limit_map.
    curr_rename_map = {}
    known_tnums = set(col_to_tnum.values()) if col_to_tnum else set()
    norm_known_tnums = {normalize_colname(str(t)) for t in known_tnums}
    
    for c in curr_df.columns:
        if col_to_tnum and c in col_to_tnum:
            # Use original T-number format (e.g. T1.0) to match limit_map keys
            curr_rename_map[c] = str(col_to_tnum[c])
        else:
            norm_c = normalize_colname(c)
            # Try to find original T-number from known_tnums that matches
            orig_tnum = next((str(t) for t in known_tnums if normalize_colname(str(t)) == norm_c), None)
            if orig_tnum:
                curr_rename_map[c] = orig_tnum
            elif norm_c in limit_map:
                curr_rename_map[c] = norm_c
            else:
                found = False
                for k in limit_map.keys():
                    norm_k = normalize_colname(k)
                    if norm_c == norm_k or norm_c.startswith(norm_k) or norm_k.startswith(norm_c):
                        curr_rename_map[c] = k
                        found = True
                        break
                if not found:
                    curr_rename_map[c] = norm_c
                
    curr_df_norm = curr_df_norm.rename(columns=curr_rename_map)
    
    # Overlapping test columns are identified.
    curr_cols = set(curr_df_norm.columns)
    hist_cols = set(hist_df.columns)
    common_cols = curr_cols.intersection(hist_cols)
    
    # Metadata is filtered out.
    test_cols = []
    for c in common_cols:
        if c in META_COLS_TO_EXCLUDE or any(p in c for p in TEST_NAMES_TO_EXCLUDE):
            continue
        test_cols.append(c)
        
    test_cols = sorted(test_cols)
    
    # The limits DataFrame and shift flags are constructed.
    limits_rows = []
    shift_flags = {}
    
    for t in test_cols:
        lsl, usl = limit_map.get(t, (np.nan, np.nan))
        limits_rows.append({"Test": t, "LSL": lsl, "USL": usl})
        
        hist_data = hist_df[t].dropna()
        curr_data = curr_df_norm[t].dropna()
        
        hm = hist_data.median()
        cm = curr_data.median()
        
        spc_shift = False
        if not hist_data.empty and not curr_data.empty:
            q25 = hist_data.quantile(0.25)
            q75 = hist_data.quantile(0.75)
            iqr = q75 - q25
            
            # SPC Logic: absolute difference in medians > 1.5 * IQR
            if pd.notna(hm) and pd.notna(cm) and iqr > 0:
                if abs(cm - hm) > 1.5 * iqr:
                    spc_shift = True
                    
        if "_LOT_ID_LABEL" in hist_df.columns:
            psi_scores = []
            for lbl in hist_df["_LOT_ID_LABEL"].unique():
                lot_hist_data = hist_df[hist_df["_LOT_ID_LABEL"] == lbl][t].dropna()
                if not lot_hist_data.empty:
                    s = _calculate_psi(lot_hist_data, curr_data)
                    if pd.notna(s):
                        psi_scores.append(s)
            psi_val = min(psi_scores) if psi_scores else np.nan
        else:
            psi_val = _calculate_psi(hist_data, curr_data)
        
        psi_status = "Normal"
        if pd.notna(psi_val):
            if psi_val > 0.25:
                psi_status = "Major Shift"
            elif psi_val >= 0.1:
                psi_status = "Minor Shift"
                
        # FT Logic (Z-deviation from Golden Baseline)
        ft_dev = 0.0
        ft_alert = False
        if not hist_data.empty and not curr_data.empty:
            h_mean = hist_data.mean()
            h_std = hist_data.std()
            if h_std > 0:
                ft_dev = abs(curr_data.mean() - h_mean) / h_std
                if ft_dev > 2.0: # Alert if Z-deviation > 2
                    ft_alert = True
                
        shift_flags[t] = {
            "PSI": psi_val,
            "PSI_Status": psi_status,
            "SPC_Shift": spc_shift,
            "FT_Dev": ft_dev,
            "FT_Alert": ft_alert,
            "hist_mean": float(hist_data.mean()) if not hist_data.empty else np.nan,
            "hist_std": float(hist_data.std()) if not hist_data.empty else np.nan
        }
        
    if not limits_rows:
        limits_df = pd.DataFrame(columns=["Test", "LSL", "USL"])
    else:
        limits_df = pd.DataFrame(limits_rows)
    
    # Preserve labels in hist_df for visualization
    hist_out_cols = test_cols
    extra_cols = [c for c in ["_LOT_ID", "_YIELD", "_LOT_ID_LABEL"] if c in hist_df.columns]
    hist_out_cols = test_cols + extra_cols

    return curr_df_norm[test_cols], hist_df[hist_out_cols], limits_df, shift_flags, tnum_to_name

def build_real_test_summary(curr_df, hist_df, limits_df, shift_flags):
    if curr_df.empty or limits_df.empty:
        return pd.DataFrame()
        
    lim = limits_df.set_index("Test")
    rows = []
    for t in curr_df.columns:
        lsl, usl = lim.loc[t, "LSL"], lim.loc[t, "USL"]
        mc, sc = float(curr_df[t].mean()), float(curr_df[t].std(ddof=1))
        
        if t in hist_df.columns and not hist_df.empty:
            mh, sh = float(hist_df[t].mean()), float(hist_df[t].std(ddof=1))
        else:
            mh, sh = np.nan, np.nan
            
        cpk_c = cpk(mc, sc, lsl, usl)
        cpk_h = cpk(mh, sh, lsl, usl)
        
        sf = shift_flags.get(t, {})
        psi = sf.get("PSI", np.nan)
        psi_status = sf.get("PSI_Status", "N/A")
        spc_alert = sf.get("SPC_Shift", False)
        ft_alert = sf.get("FT_Alert", False)
        ft_dev = sf.get("FT_Dev", 0.0)
        
        rows.append({
            "Test": t, 
            "PSI Status": psi_status, 
            "PSI Score": psi, 
            "SPC Alert": "YES" if spc_alert else "NO",
            "FT Alert": "YES" if ft_alert else "NO",
            "FT Deviation (Z)": ft_dev,
            "Mean (Curr)": mc, "Std (Curr)": sc, "Cpk (Curr)": cpk_c if pd.notna(cpk_c) else "N/A",
            "Mean (Hist)": mh, "Std (Hist)": sh, "Cpk (Hist)": cpk_h if pd.notna(cpk_h) else "N/A"
        })
    out = pd.DataFrame(rows)
    out["_sort"] = out["PSI Status"].map(lambda x: 0 if x == "Major Shift" else (1 if x == "Minor Shift" else 2))
    out = out.sort_values(by=["_sort", "Test"]).drop(columns=["_sort"])
    
    # Floats are formatted correctly.
    for c in ["Mean (Curr)", "Std (Curr)", "Mean (Hist)", "Std (Hist)"]:
        out[c] = out[c].apply(lambda v: f"{v:.4f}" if pd.notna(v) and isinstance(v, float) else v)
    for c in ["Cpk (Curr)", "Cpk (Hist)"]:
        out[c] = out[c].apply(lambda v: f"{v:.2f}" if pd.notna(v) and isinstance(v, float) else v)
    out["PSI Score"] = out["PSI Score"].apply(lambda v: f"{v:.4f}" if pd.notna(v) and isinstance(v, float) else v)
    out["FT Deviation (Z)"] = out["FT Deviation (Z)"].apply(lambda v: f"{v:.2f}σ" if pd.notna(v) and isinstance(v, float) else v)
        
    return out

def style_shift_rows(df: pd.DataFrame):
    def _row_style(row):
        color = "#Ffe0e0" if row.get("PSI Status") == "Major Shift" else "#eef2ff"
        return [f"background-color: {color}"] * len(row)
    return df.style.apply(_row_style, axis=1)


# --- Main Layout ---
# Unified CSS Injection for Global Styling
st.markdown(
    """
    <style>
    /* Green Multiselect Tags */
    span[data-baseweb="tag"], [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: #28a745 !important;
    }
    /* Green Primary Buttons */
    button[kind="primary"], div[data-testid="stButton"] button[kind="primary"] {
        background-color: #28a745 !important;
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
    }
    button[kind="primary"]:hover {
        background-color: #218838 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
    }
    /* Red Font for Shifted Tests in Selectbox */
    /* Target the selected value display */
    [data-testid="stSelectbox"] [data-baseweb="select"] div[title*="[WARN]"],
    [data-testid="stSelectbox"] [data-baseweb="select"] span:contains("[WARN]") {
        color: #Eb5757 !important;
        font-weight: bold !important;
    }
    /* Target dropdown menu items */
    li[role="option"] div:contains("[WARN]"),
    li[role="option"] span:contains("[WARN]") {
        color: #Eb5757 !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

sidebar = st.sidebar
sidebar.header("🎛️ Navigation")
app_mode = sidebar.radio("Select Mode", ["Model Preparation Pipeline", "Single Lot Analytics & Prediction"], help="Switch between batch data preparation and single lot yield prediction.")

if app_mode == "Model Preparation Pipeline":
    sidebar.header("Configuration")

    # --- Auto-invalidate @st.cache_resource if TesterFamilyMap.xlsx changes on disk ---
    _map_mtime_key = "tester_map_mtime"
    try:
        _current_mtime = TESTER_MAPPING_FILE.stat().st_mtime if TESTER_MAPPING_FILE.exists() else 0.0
    except Exception:
        _current_mtime = 0.0
    if st.session_state.get(_map_mtime_key, -1) != _current_mtime:
        get_generics_by_tester.clear()
        get_all_generics_from_map.clear()
        st.session_state[_map_mtime_key] = _current_mtime

    # Data is loaded fresh from TesterFamilyMap.xlsx.
    tester_map = get_generics_by_tester()
    all_families = sorted(list(tester_map.keys()))

    # 1. The Tester Family Selector is initialized.
    selected_family = sidebar.selectbox(
        "Select Tester Family",
        options=["All"] + all_families,
        index=0,
        help="Filter Generics by Tester Family. This also sets the decryption mode."
    )

    # 2. Generics are filtered based on what is in TesterFamilyMap.xlsx.
    if selected_family == "All":
        available_generics = sorted({g for gens in tester_map.values() for g in gens})
        if not available_generics:
            available_generics = get_all_generics_from_map()
    else:
        available_generics = tester_map.get(selected_family, [])

    selected_generics = sidebar.multiselect(
        "Select Generic(s)",
        options=available_generics,
        help=f"Select Generics (Filtered by {selected_family})"
    )

    # 2b. Bulk Generic Input
    bulk_input = sidebar.text_area(
        "Bulk Generic Input",
        placeholder="Paste multiple generics here (one per line or comma-separated)",
        help="You can paste a list of generics here. They will be merged with the selection above."
    )

    # Merge inputs
    final_selected_generics = set(selected_generics)
    if bulk_input:
        bulk_list = [g.strip() for g in re.split(r'[\n,]+', bulk_input) if g.strip()]
        final_selected_generics.update(bulk_list)

    final_selected_generics = sorted(list(final_selected_generics))

    if final_selected_generics:
        sidebar.info(f"Summary: {len(final_selected_generics)} generic(s) ready to process.")

    force_val = "Auto"
    if selected_family != "All":
        force_val = selected_family

    # 3. Sidebar Toggles
    combine_wafer_data = True # Set as always true per user request
    cleanup_after_generic = sidebar.checkbox("Free up disk space (OneDrive Online-Only)", value=True, key="cleanup_after_generic")
    skip_existing = sidebar.checkbox("Skip Already Processed", value=True, key="skip_existing")
    
    sidebar.markdown("---")
    sidebar.subheader("Machine Learning")
    run_feature_extraction_step = sidebar.checkbox("Run Feature Extraction", value=True, key="run_feature_extraction")
    run_model_training_step = sidebar.checkbox("Run Model Training", value=True, key="run_model_training")

    def handle_run():
        st.session_state.processing = True
        st.session_state.start_pipeline = True
        st.session_state.pipeline_finished = False
        # Ensure logs are cleared immediately upon clicking Run
        st.session_state.logs = ["[INFO] Initiating pipeline..."]
        st.session_state.generic_results = {}

    def handle_reset():
        st.session_state.processing = False
        st.session_state.start_pipeline = False
        st.session_state.pipeline_finished = False
        st.session_state.logs = []
        st.session_state.generic_logs = {}
        st.session_state.generic_results = {}
        st.session_state.pipeline_metrics = {}
        st.cache_data.clear()
        st.cache_resource.clear()
        # Kill background browser drivers that might be hanging
        if sys.platform == "win32":
            try:
                # /T kills child processes as well
                subprocess.run("taskkill /F /IM chromedriver.exe /T", shell=True, capture_output=True)
                subprocess.run("taskkill /F /IM msedgedriver.exe /T", shell=True, capture_output=True)
            except: pass

    col_btn1, col_btn2 = sidebar.columns(2)
    with col_btn1:
        run_btn = st.button("Run Pipeline", disabled=st.session_state.processing, on_click=handle_run, use_container_width=True)
    with col_btn2:
        reset_btn = st.button("Stop & Reset", on_click=handle_reset, use_container_width=True, help="Stop currently running pipeline and clear all logs/cache.", type="secondary")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Result Preview", "Logs", "Help"])

    with tab2:
        st.subheader("Execution Logs")
        logs_ph = st.empty()
        if not st.session_state.processing:
            if st.session_state.logs:
                render_logs(st.session_state.logs, st)
            else:
                st.info("Logs will appear here after running the pipeline.")

    with tab1:
        st.subheader("Processing Summary")
        dashboard_ph = st.empty()
        
        has_results = bool(st.session_state.get("generic_results"))
        is_processing = st.session_state.get("processing", False)
        
        # The user requested the summary only appear after processing completes ("Pipeline Finished" as trigger).
        is_finished = st.session_state.get("pipeline_finished", False)
        if is_finished and st.session_state.get("generic_results"):
            summary_data = [{"Generic": g, "Status": s} for g, s in st.session_state.generic_results.items()]
            # Table height now scales based on content rather than being fixed.
            st.dataframe(summary_data, use_container_width=True, hide_index=True)

            # --- Section 2: Model Validation Analytics ---
            st.markdown("---")
            st.subheader("2) Model Validation Analytics")
            
            processed_generics = [g for g, s in st.session_state.generic_results.items() if "Success" in s]
            if not processed_generics:
                st.info("No successful models to analyze. Run the pipeline to see validation metrics.")
            else:
                sel_gen = st.selectbox("Select Generic to Analyze Validation Accuracy", processed_generics)
                
                # Attempt to find the model details file in the inventory or recent results
                details_file = None
                
                # Check recent results first
                for f_path in st.session_state.get("final_csvs", []):
                    if sel_gen in f_path and "model_details" in f_path and f_path.endswith(".xlsx"):
                        details_file = Path(f_path)
                        break
                
                # Fallback: Scan dataset
                if not details_file or not details_file.exists():
                    for root, _, files in os.walk(CURRENT_DIR / "dataset"):
                        for f in files:
                            if sel_gen in f and "model_details" in f and f.endswith(".xlsx"):
                                details_file = Path(root) / f
                                break
                        if details_file: break
                
                if details_file and details_file.exists():
                    try:
                        xl = pd.ExcelFile(details_file)
                        sheets = xl.sheet_names
                        # Prioritize 'Yield Comparison' or any sheet with 'Comp'
                        comp_sheets = [s for s in sheets if "Comp" in s or "Comparison" in s]
                        
                        if comp_sheets:
                            col_a, col_b = st.columns([1, 3])
                            with col_a:
                                sel_sheet = st.selectbox("Parameter", comp_sheets)
                            
                            df_val = pd.read_excel(details_file, sheet_name=sel_sheet)
                            
                            # Identify Actual and Predicted columns
                            act_cols = [c for c in df_val.columns if "actual_" in c]
                            if not act_cols:
                                st.warning("No 'actual_' columns found in the comparison sheet.")
                            else:
                                act_col = act_cols[0]
                                pred_cols = [c for c in df_val.columns if c not in ["group", act_col]]
                                
                                # 1. Stats Table
                                with col_b:
                                    st.write(f"**Validation Samples:** `{len(df_val)}` lots/wafers")
                                
                                st.dataframe(df_val, use_container_width=True, hide_index=True)
                                
                                # 2. Scatter Plot (Actual vs Predicted)
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                
                                # Handle NaNs for plot bounds
                                plot_df = df_val.dropna(subset=[act_col] + pred_cols)
                                if not plot_df.empty:
                                    min_v = min(plot_df[act_col].min(), plot_df[pred_cols].min().min())
                                    max_v = max(plot_df[act_col].max(), plot_df[pred_cols].max().max())
                                    # Add 5% padding
                                    pad = (max_v - min_v) * 0.05 if max_v > min_v else 0.1
                                    min_v -= pad
                                    max_v += pad
                                    
                                    # Identity Line (y=x)
                                    fig.add_trace(go.Scatter(
                                        x=[min_v, max_v], y=[min_v, max_v],
                                        mode='lines', name='Perfect Prediction (y=x)',
                                        line=dict(color='rgba(150,150,150,0.5)', dash='dash')
                                    ))
                                    
                                    # Model Predictions
                                    import plotly.express as px
                                    colors = px.colors.qualitative.Alphabet # 26 distinct colors
                                    for i, p_col in enumerate(pred_cols):
                                        fig.add_trace(go.Scatter(
                                            x=plot_df[act_col], 
                                            y=plot_df[p_col], 
                                            mode='markers', 
                                            name=f"Model: {p_col.split('_')[0]}",
                                            marker=dict(size=10, opacity=0.7, color=colors[i % len(colors)]),
                                            hovertemplate="<b>%{text}</b><br>Actual: %{x:.4f}<br>Predicted: %{y:.4f}<extra></extra>",
                                            text=plot_df['group'] if 'group' in plot_df.columns else None
                                        ))
                                        
                                    fig.update_layout(
                                        title=f"Validation Accuracy: Actual vs. Predicted ({sel_sheet})",
                                        xaxis_title="Actual Measurement",
                                        yaxis_title="AI Prediction",
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                        height=600,
                                        hovermode="closest"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("The selected generic has no comparison sheets available. This may be an older model.")
                    except Exception as e:
                        st.error(f"Error loading validation analytics: {e}")
                else:
                    st.warning(f"Validation details file not found for `{sel_gen}`.")

        elif not st.session_state.get("processing", False):
            st.info("No generic status to show yet. Run the pipeline to see completion status here.")
            
        # UI elements required by the run_pipeline and rendering logic (Ordered per user request)
        status_container = st.container()
        
    with tab3:
        st.markdown("""
        ### How to use
        1. Enter one or more **Generic** names in the sidebar (e.g., `DDR4SDRAM`, `DDR5SDRAM`).
        2. Click **Run Pipeline**.
        3. Monitor the **Logs** tab for progress.
        
        ### Pipeline Steps
        1. **Database**: Scrapes data.
        2. **Decryption**: Converts to CSV.
        3. **Combiner**: Merges wafer data.
        4. **ML**: Extract features & Train models.
        """)

    # Run Logic
    if st.session_state.get("start_pipeline"):
        st.session_state.start_pipeline = False
        st.session_state.logs = []
        st.session_state.generic_results = {}
        st.cache_data.clear()

        # A check is performed for an auto-triggered generic from the Predictor.
        auto_gen = st.session_state.get("auto_generic")
        g_list = [auto_gen] if auto_gen else final_selected_generics
        
        if not g_list:
            st.error("Please select or paste at least one Generic.")
            st.session_state.processing = False
            if "auto_generic" in st.session_state: del st.session_state["auto_generic"]
        else:
            # Pass dashboard_ph and status_container as the target containers for UI injection.
            run_pipeline(
                g_list, dashboard_ph, logs_ph, status_container, 
                force_decryptor=force_val, skip_existing=skip_existing,
                combine_wafer_data=combine_wafer_data,
                run_feature_extraction_step=run_feature_extraction_step,
                run_model_training_step=run_model_training_step,
                cleanup_after_generic=cleanup_after_generic
            )
            # Auto_generic is cleaned up if it was used.
            if "auto_generic" in st.session_state: del st.session_state["auto_generic"]
    elif st.session_state.get("pipeline_metrics") and st.session_state.get("logs") and not st.session_state.processing:
        render_dashboard(st.session_state.pipeline_metrics, dashboard_ph)

elif app_mode == "Single Lot Analytics & Prediction":
    import ml_yield_prediction
    import ml_train_model
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Predictor Inputs")
    p_generic = st.sidebar.text_input("Generic (e.g. DDR4SDRAM)", "")
    p_partname = st.sidebar.text_input("Part Number(e.g. MT40A1G8)", "")
    p_dlog = st.sidebar.file_uploader("Upload DLog File", type=[".stdf", ".std", ".std_1", ".gz", ".zip", ".csv"], help="Upload a DLog file to predict yield.")
    
    st.sidebar.markdown("---")
    auto_pipeline = st.sidebar.checkbox("Execute pipeline if model not found", value=False, help="Automatically runs the full Data Prep pipeline to generate models if none exist for this Generic.")
    
    

    st.header("🧬 Wafer Analytics and Predictive Gateway")
    
    if p_generic:
        models_found = ml_yield_prediction.get_models_for_generic(p_generic, p_partname)
        if not models_found:
            st.warning(f"⚠️ No trained models found for '{p_generic}' {'(' + p_partname + ')' if p_partname else ''}.")
            st.info("Models are required for prediction.")
            
            # A button is provided to trigger the pipeline manually or via auto flow.
            if auto_pipeline:
                 st.info(f"Auto-Pipeline is enabled. Will run pipeline for {p_generic}.")
            else:
                 if st.button(f"🚀 Run Pipeline for {p_generic}", type="secondary"):
                    # The pipeline state is initialized.
                    st.session_state.processing = True
                    st.session_state.start_pipeline = True
                    st.session_state.auto_generic = p_generic
                    st.rerun()
                    
        else:
            st.success(f"[INFO] Found {len(models_found)} models for '{p_generic}'. Ready to predict.")
            
        # Unified Start Button
        if st.button("Predict Yield from DLOG", type="primary"):
            if not models_found and auto_pipeline:
                 # The auto pipeline is triggered, although a Streamlit rerun may interrupt further prediction.
                 # The simplest approach is to prompt the user to wait for the pipeline and then click predict again.
                 st.session_state.processing = True
                 st.session_state.start_pipeline = True
                 st.session_state.auto_generic = p_generic
                 st.rerun()
                 
            elif not p_dlog:
                st.error("Please provide a path to the DLog file.")
            else:
                st.markdown("### Process Log")
                log_container = st.empty()
                ui_logs = []
                    
                def ui_logger(msg):
                    ui_logs.append(msg)
                    log_container.code("\n".join(ui_logs))
                
                with st.spinner("Processing DLog (Decrypting (DLog) -> Featurizing -> Predicting)..."):
                    # Create a temporary directory to preserve the original filename
                    temp_dir_predict = Path(tempfile.mkdtemp(prefix="predict_"))
                    temp_path = temp_dir_predict / p_dlog.name
                    try:
                        with open(temp_path, "wb") as f:
                            f.write(p_dlog.getbuffer())
                        
                        result = ml_yield_prediction.predict_from_dlog(p_generic, p_partname, temp_path, log_func=ui_logger)
                        if result["status"] != "success":
                            st.error(result.get("message", "Prediction halted without success status."))
                    finally:
                        try:
                            import shutil
                            shutil.rmtree(temp_dir_predict, ignore_errors=True)
                        except: pass
                    
                    if result["status"] == "success":
                        avg_val = result.get("average_prediction", 0)
                        if avg_val > 1.0: avg_val /= 100.0
                        
                        # A simulated Lot ID is derived based on the filename.
                        parts = p_dlog.name.split('_')
                        derived_lot_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 and parts[0] == "SYN" else parts[0].split('.')[0]
                        if len(derived_lot_id) < 4: derived_lot_id = "P070238.1"
                        
                        st.session_state["p_wafer"] = {
                            "Wafer_ID": f"{derived_lot_id}-01",
                            "Predicted_Yield": float(np.clip(avg_val, 0, 1)),
                            "details": result.get("predictions", {}),
                            "shap_data": result.get("shap_data", {}),
                            "curr_df": result.get("curr_df", None),
                            "full_df": result.get("full_df", None),
                            "limit_map": result.get("limit_map", {}),
                            "best_r2": result.get("best_r2", None),
                            "col_to_tnum": result.get("col_to_tnum", {}),
                            "model_dir": result.get("model_dir", ""),
                            "creation_date": result.get("creation_date")
                        }
                        st.session_state["p_hist_avg"] = historical_avg_yield(derived_lot_id)
                        st.session_state["p_done"] = True
                        st.session_state["p_lot_id"] = derived_lot_id
                        st.session_state["p_year"] = extract_fiscal_year(p_dlog.name)
                        st.session_state["p_filename"] = p_dlog.name

    if st.session_state.get("p_done"):
        import reliability_grading
        import spatial_analysis
        import pandas as pd
        import plotly.express as px
        
        res = st.session_state["p_wafer"]
        p_lot_id = st.session_state.get("p_lot_id", "P070238.1")
        hist_avg = st.session_state["p_hist_avg"]
        pred = res["Predicted_Yield"]
        
        # 1. Virtual Golden Wafer Baseline Model (Dual-Track: Probe + FT)
        golden_probe = build_golden_baseline(p_generic, stage="probe", top_n=3)
        golden_ft = build_golden_baseline(p_generic, stage="ft", top_n=3)
        
        # Compute baseline yield from golden FT (primary), fallback to golden probe, then historical avg
        if golden_ft and golden_ft.get("golden_yield", 0) > 0:
            baseline_yield = golden_ft["golden_yield"] / 100.0
        elif golden_probe and golden_probe.get("golden_yield", 0) > 0:
            baseline_yield = golden_probe["golden_yield"] / 100.0
        else:
            baseline_yield = hist_avg

        # Compute baseline_std from golden probe parameter variances
        if golden_probe and golden_probe.get("params"):
            g_stds = [v["std"] for v in golden_probe["params"].values() if v["std"] > 0]
            baseline_std = float(np.mean(g_stds)) if g_stds else 0.05
        else:
            baseline_std = 0.05
        
        # Compute current wafer per-parameter stats for comparison
        curr_df_raw = res.get("curr_df", None)
        current_param_stats = {}
        if curr_df_raw is not None and not curr_df_raw.empty:
            from ml_compute_statistic import normalize_colname, META_COLS_TO_EXCLUDE, TEST_NAMES_TO_EXCLUDE
            col_to_tnum_raw = res.get("col_to_tnum", {})
            for c in curr_df_raw.columns:
                nc = normalize_colname(c)
                if nc in META_COLS_TO_EXCLUDE or any(p in nc for p in TEST_NAMES_TO_EXCLUDE):
                    continue
                vals = pd.to_numeric(curr_df_raw[c], errors='coerce').dropna()
                if len(vals) < 5:
                    continue
                # Map to T-number if available. Check both original and already-renamed T-numbers.
                if c in col_to_tnum_raw.values():
                    t_key = c
                else:
                    t_key = col_to_tnum_raw.get(c, nc)
                
                current_param_stats[t_key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "median": float(vals.median()),
                    "min": float(vals.min()),
                    "max": float(vals.max())
                }
            
        yield_err = abs(baseline_yield - pred)
        
        # Probe deviation Z-score from golden parameters (not just yield)
        if golden_probe and golden_probe.get("params") and current_param_stats:
            dev_zs = []
            for param, g_stats in golden_probe["params"].items():
                c_stats = current_param_stats.get(param)
                if c_stats and g_stats["std"] > 1e-9:
                    dev_zs.append(abs(c_stats["mean"] - g_stats["mean"]) / g_stats["std"])
            probe_dev_z = float(np.mean(dev_zs)) if dev_zs else abs(pred - baseline_yield) / (baseline_std + 1e-9)
        else:
            probe_dev_z = abs(pred - baseline_yield) / (baseline_std + 1e-9)
        
        # 2. Bin Map Spatial Analysis (GDBN)
        full_df = res.get("full_df", None)
        if full_df is not None and not full_df.empty:
            spatial_res = spatial_analysis.compute_spatial_risk(full_df)
        else:
            spatial_res = {
                "spatial_defect_density": 0.0,
                "gdbn_count": 0,
                "gdbn_ratio_all_dies": 0.0,
                "gdbn_rate_good_dies": 0.0,
                "total_dies": 0,
                "total_good_dies": 0,
                "edge_cluster_count": 0,
                "wafer_map_df": pd.DataFrame(),
            }

        # 3. Reliability Scoring Framework & Product Grade Classification
        # Extract WaferPulse (WP) Performance metrics for unified display and scoring
        wp_mttf = 0.0
        details = res.get("details", {})
        if details:
            top_model = list(details.keys())[0]
            wp_res = details[top_model]
            if isinstance(wp_res, dict) and "WP_MTTF_Years" in wp_res:
                wp_mttf = wp_res['WP_MTTF_Years']

        # Build predicted FT stats dynamically from model predictions to replace the placeholder mock Z-score
        predicted_ft_stats = {}
        if details and golden_ft and golden_ft.get("params"):
            top_model = list(details.keys())[0]
            top_preds = details[top_model]
            if isinstance(top_preds, dict):
                golden_params = golden_ft.get("params", {})
                golden_keys = list(golden_params.keys())
                
                for k, v in top_preds.items():
                    if k in ["FT_y", "y"] or isinstance(v, str):
                        continue
                    try:
                        f_val = float(v)
                    except:
                        continue
                    
                    # Parse key like FT_t1_0__mean -> base="t1_0", stat="mean"
                    k_clean = k.replace("FT_", "")
                    if "__" in k_clean:
                        base, stat_type = k_clean.split("__", 1)
                    else:
                        base, stat_type = k_clean, "mean"
                    
                    # Convert normalized name back: t1_0 -> T1.0
                    base_upper = base.upper().replace("_", ".")
                    
                    # Find matching golden key
                    matched_key = None
                    base_norm = normalize_colname(base)
                    for gk in golden_keys:
                        # Match exact T-number (T1.0) or normalized name
                        if gk.upper().replace(" ", "") == base_upper or normalize_colname(gk) == base_norm:
                            matched_key = gk
                            break
                    
                    target_key = matched_key if matched_key else k
                    if target_key not in predicted_ft_stats:
                        predicted_ft_stats[target_key] = {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
                    
                    # Map standard stat types
                    if stat_type == "mean":
                        predicted_ft_stats[target_key]["mean"] = f_val
                    elif stat_type == "std":
                        predicted_ft_stats[target_key]["std"] = f_val
                    elif stat_type in ["median", "p50"]:
                        predicted_ft_stats[target_key]["median"] = f_val
                    elif stat_type == "min":
                        predicted_ft_stats[target_key]["min"] = f_val
                    elif stat_type == "max":
                        predicted_ft_stats[target_key]["max"] = f_val

        # Dynamic FT deviation Z-score calculation from golden FT parameters
        if golden_ft and golden_ft.get("params") and predicted_ft_stats:
            ft_dev_zs = []
            for param, g_stats in golden_ft["params"].items():
                c_stats = predicted_ft_stats.get(param)
                if c_stats and g_stats["std"] > 1e-9:
                    ft_dev_zs.append(abs(c_stats["mean"] - g_stats["mean"]) / g_stats["std"])
            ft_dev_z = float(np.mean(ft_dev_zs)) if ft_dev_zs else abs(pred - baseline_yield) / (baseline_std + 1e-9)
        else:
            ft_dev_z = abs(pred - baseline_yield) / (baseline_std + 1e-9)

        # Post-packaging reliability gate: combine package MTTF with golden-baseline correlation.
        probe_corr = reliability_grading.compute_golden_correlation(current_param_stats, golden_probe)
        ft_corr = reliability_grading.compute_golden_correlation(predicted_ft_stats, golden_ft)
        corr_values = [c for c in [probe_corr, ft_corr] if np.isfinite(c)]
        golden_correlation = float(np.mean(corr_values)) if corr_values else 0.0

        # Compute max Z-scores for failure penalty calculations
        probe_max_z = reliability_grading.compute_max_zscore(current_param_stats, golden_probe)
        ft_max_z = reliability_grading.compute_max_zscore(predicted_ft_stats, golden_ft)

        probe_penalty = float(np.clip(20.0 + 30.0 * (probe_max_z - 1.0), 0.0, 80.0)) if probe_max_z > 1.0 else 0.0
        ft_penalty = float(np.clip(20.0 + 30.0 * (ft_max_z - 1.0), 0.0, 80.0)) if ft_max_z > 1.0 else 0.0

        # Granular calculations for upgraded similarity and Z-score closeness metrics
        probe_corr_val = probe_corr if np.isfinite(probe_corr) else 0.0
        probe_corr_score = float(np.clip(max(probe_corr_val, 0.0) * 100.0 - probe_penalty, 0.0, 100.0))

        ft_corr_val = ft_corr if np.isfinite(ft_corr) else 0.0
        ft_corr_score = float(np.clip(max(ft_corr_val, 0.0) * 100.0 - ft_penalty, 0.0, 100.0))

        probe_z_closeness = reliability_grading.compute_zscore_closeness(current_param_stats, golden_probe)
        ft_z_closeness = reliability_grading.compute_zscore_closeness(predicted_ft_stats, golden_ft)

        golden_similarity_probe = 0.5 * probe_corr_score + 0.5 * probe_z_closeness
        golden_similarity_ft = 0.5 * ft_corr_score + 0.5 * ft_z_closeness

        score_dict = reliability_grading.score_post_packaging_reliability(
            wafer_id=p_lot_id,
            mttf_years=wp_mttf,
            golden_correlation=golden_correlation,
            predicted_yield=pred,
            golden_similarity_probe=golden_similarity_probe,
            golden_similarity_ft=golden_similarity_ft,
            probe_corr=probe_corr_val,
            probe_z_closeness=probe_z_closeness,
            ft_corr=ft_corr_val,
            ft_z_closeness=ft_z_closeness,
            probe_max_z=probe_max_z,
            ft_max_z=ft_max_z,
        )

        grade = score_dict['Grade']
        ri_score = score_dict['Risk_Score']
        desc = score_dict['Application']
        
        if grade == "D":
            # Scale down the displayed Expected Lifespan (MTTF) to reflect high defect rates / process drift
            wp_mttf = min(wp_mttf * 0.05, 2.8)
            wp_mttf = max(wp_mttf, 0.5) 
        
        
        THR = {"Z": 0.98, "A": 0.95, "H": 0.90, "I": 0.85, "C": 0.80}
        best_r2_val = res.get("best_r2")
        if best_r2_val is not None and not np.isnan(best_r2_val):
            conf_p = float(best_r2_val)
        else:
            conf_p = confidence_probability(pred, hist_avg, THR)
            
        st.markdown("---")
        st.subheader("1) Wafer Probe Test Parameter Distribution Shift Analysis")
        
        curr_df_raw = res.get("curr_df", None)
        limit_map_raw = res.get("limit_map", {})
        col_to_tnum_raw = res.get("col_to_tnum", {})
        
        if curr_df_raw is None or curr_df_raw.empty:
             st.info("No raw data available for distribution comparison.")
        else:
             with st.spinner("Scanning historical inventory..."):
                 inv = get_historical_inventory(p_generic)
             
                 if not inv:
                     st.info(f"No historical data found for '{p_generic}' with good yield (>80%).")
                 else:
                     tab_detail, tab_filters, tab_radar = st.tabs([
                          "📊 Detail Analysis", "🔍 Historical Comparison Filters", "📡 Radar Chart"
                     ])
                      
                     with tab_filters:
                          st.subheader("🔍 Historical Comparison Filters")
                          colf1, colf2 = st.columns([1,2])
                          
                          years = sorted(list(set(i["year"] for i in inv)), reverse=True)
                          with colf1:
                              sel_years = st.multiselect("Filter by Fiscal Year", options=years, default=years)
                          
                          filtered_inv = [i for i in inv if i["year"] in sel_years]
                          # Sort by yield descending so we compare against the highest yield historical lots
                          filtered_inv_sorted = sorted(filtered_inv, key=lambda x: x["yield"], reverse=True)
                          
                          lot_options = {f"{i['lot_id']} ({i['year']}, {i['yield']:.1f}%)": i['path'] for i in filtered_inv_sorted}
                          with colf2:
                              default_lots = list(lot_options.keys())[:3]
                              sel_lot_labels = st.multiselect("Select Lot IDs (Max 3) for comparison", options=list(lot_options.keys()), default=default_lots, max_selections=3)
                          
                          selected_paths = [lot_options[l] for l in sel_lot_labels]
                     
                     if not selected_paths:
                          with tab_detail:
                              st.warning("Please select at least one historical Lot ID in the 'Historical Comparison Filters' tab.")
                          with tab_filters:
                              st.warning("Please select at least one historical Lot ID.")
                          with tab_radar:
                              st.warning("Please select at least one historical Lot ID in the 'Historical Comparison Filters' tab.")
                     else:
                          with st.spinner("Analyzing historical vs current test distributions..."):
                              curr_df, hist_df, limits_df, shift_flags, tnum_to_name = get_real_distribution_data(p_generic, limit_map_raw, curr_df_raw, col_to_tnum_raw, selected_lots=selected_paths)
                          
                          if curr_df.empty or hist_df.empty:
                              with tab_detail:
                                  st.info(f"No common tests found between current lot and selected historical data.")
                              with tab_filters:
                                  st.info(f"No common tests found between current lot and selected historical data.")
                              with tab_radar:
                                  st.info(f"No common tests found between current lot and selected historical data.")
                          else:
                              # Render summary table in second tab (tab_filters)
                              with tab_filters:
                                  st.write("---")
                                  st.subheader("📊 Raw Test Parameter Shift Summary")
                                  summary_df = build_real_test_summary(curr_df, hist_df, limits_df, shift_flags)
                                  st.dataframe(style_shift_rows(summary_df), height=400, use_container_width=True)
                              
                              # Render detailed analysis in first tab (tab_detail)
                              with tab_detail:
                                  st.subheader("📊 Detail Component Analysis")
                                  
                                  TEST_LIST = list(curr_df.columns)
                                  TEST_LIST.sort(key=lambda x: (
                                       not (shift_flags.get(x, {}).get("PSI_Status") == "Major Shift" or shift_flags.get(x, {}).get("SPC_Shift", False)),
                                       x
                                  ))
                                  
                                  def fmt_test(t):
                                      name = tnum_to_name.get(t, "")
                                      label = f"{t}: {name}" if name else t
                                      sf = shift_flags.get(t, {})
                                      if sf.get("PSI_Status") == "Major Shift" or sf.get("SPC_Shift", False):
                                          return f"{label} [WARN]"
                                      return label
                                  
                                  try:
                                      import re
                                      def local_parse(x):
                                          m = re.search(r'T(\d+\.?\d*)', str(x))
                                          return float(m.group(1)) if m else float('inf')
                                      TEST_LIST.sort(key=lambda x: local_parse(x))
                                  except Exception:
                                      pass
                                      
                                  sel_tests = st.multiselect("Select Test(s) for Detail Component", TEST_LIST, default=[TEST_LIST[0]] if TEST_LIST else [], format_func=fmt_test)
                                  show_full_range = st.checkbox("Show 'All Data with Limits' Chart (Alongside '99% Distribution')", value=False)
                                  
                                  for sel_test in sel_tests:
                                       sf = shift_flags.get(sel_test, {})
                                       is_warned = sf.get("PSI_Status") == "Major Shift" or sf.get("SPC_Shift", False)
                                       
                                       if is_warned:
                                           col_title, col_rc = st.columns([3, 1])
                                           with col_title:
                                               st.markdown(f"### 📊 Detail Analysis: {fmt_test(sel_test)}")
                                           with col_rc:
                                               try:
                                                   with st.popover("💡 Root Cause Actions", use_container_width=True):
                                                       st.markdown(get_root_cause_html(), unsafe_allow_html=True)
                                               except Exception:
                                                   if st.button("💡 Root Cause Actions", key=f"rc_btn_{sel_test}"):
                                                       st.info("Follow root cause engine steps displayed below.")
                                       else:
                                           st.markdown(f"### 📊 Detail Analysis: {fmt_test(sel_test)}")
                                           
                                       psi_stat = sf.get("PSI_Status", "Normal")
                                       psi_val = sf.get("PSI", np.nan)
                                       spc_alert = sf.get("SPC_Shift", False)
                                       ft_alert = sf.get("FT_Alert", False)
                                       ft_dev = sf.get("FT_Dev", 0.0)
                                       
                                       c1, c2, c3, c4 = st.columns(4)
                                       c1.metric(f"PSI ({sel_test})", f"{psi_stat}", delta=f"{psi_val:.4f}" if pd.notna(psi_val) else "N/A", delta_color="inverse" if psi_stat == "Major Shift" else "normal")
                                       c2.metric(f"SPC ({sel_test})", "Alert" if spc_alert else "Pass", delta="Median Drift" if spc_alert else None, delta_color="inverse" if spc_alert else "normal")
                                       c3.metric(f"FT ({sel_test})", "Alert" if ft_alert else "Pass", delta=f"{ft_dev:.2f} Z-Dev" if pd.notna(ft_dev) else None, delta_color="inverse" if ft_alert else "normal")
                                       
                                       l_row = limits_df[limits_df["Test"] == sel_test]
                                       lsl_txt, usl_txt = "N/A", "N/A"
                                       if not l_row.empty:
                                           lsl_txt = f"{l_row.iloc[0]['LSL']:.4g}" if pd.notna(l_row.iloc[0]['LSL']) else "N/A"
                                           usl_txt = f"{l_row.iloc[0]['USL']:.4g}" if pd.notna(l_row.iloc[0]['USL']) else "N/A"
                                       c4.metric("Spec Limits", f"{lsl_txt} / {usl_txt}", help="LSL / USL from ddr4_limits.csv")
                                       
                                       def build_spc_fig(sel_test, sf):
                                           fig_spc = go.Figure()
                                           curr_data = curr_df[sel_test].dropna()
                                           if curr_data.empty: return fig_spc
                                           fig_spc.add_trace(go.Scatter(y=curr_data, mode='markers', name=f'Current Dies ({p_lot_id})', marker=dict(color='#Eb5757', size=4, opacity=0.6)))
                                           h_mean = sf.get("hist_mean", np.nan)
                                           h_std = sf.get("hist_std", np.nan)
                                           if pd.notna(h_mean):
                                               fig_spc.add_hline(y=h_mean, line_width=2, line_color="#21c354", annotation_text="Golden Mean")
                                               if pd.notna(h_std) and h_std > 0:
                                                   fig_spc.add_hline(y=h_mean + 3 * h_std, line_dash="dash", line_color="#Ffa421", annotation_text="UCL (+3σ)")
                                                   fig_spc.add_hline(y=h_mean - 3 * h_std, line_dash="dash", line_color="#Ffa421", annotation_text="LCL (-3σ)")
                                           l_row = limits_df[limits_df["Test"] == sel_test]
                                           if not l_row.empty:
                                               if pd.notna(l_row.iloc[0]["LSL"]): fig_spc.add_hline(y=float(l_row.iloc[0]["LSL"]), line_dash="dot", line_color="#dc3545", annotation_text="LSL", annotation_position="bottom left")
                                               if pd.notna(l_row.iloc[0]["USL"]): fig_spc.add_hline(y=float(l_row.iloc[0]["USL"]), line_dash="dot", line_color="#dc3545", annotation_text="USL", annotation_position="top left")
                                           fig_spc.update_layout(title=f"SPC Run Chart: {sel_test}", xaxis_title="Die Index (Sequence)", yaxis_title="Measured Value", template="plotly_white", height=400, margin=dict(l=20, r=20, t=50, b=50))
                                           return fig_spc
                          
                                       st.plotly_chart(build_spc_fig(sel_test, sf), use_container_width=True)
                                       all_vals = []
                                       if not hist_df.empty and sel_test in hist_df.columns: all_vals.append(hist_df[sel_test].dropna())
                                       if not curr_df.empty and sel_test in curr_df.columns: all_vals.append(curr_df[sel_test].dropna())
                                       q_low, q_high = 0, 0
                                       if all_vals:
                                           combined_vals = pd.concat(all_vals)
                                           q_low, q_high = combined_vals.quantile(0.005), combined_vals.quantile(0.995)
                                           iqr = q_high - q_low
                                           if iqr > 0: q_low, q_high = q_low - 0.05 * iqr, q_high + 0.05 * iqr
                                       def build_dist_fig(filtered=False):
                                          fig_dist = go.Figure()
                                          if not hist_df.empty:
                                              colors = ["#9aa0a6", "#17a2b8", "#6f42c1", "#fd7e14", "#20c997"]
                                              for idx, lbl in enumerate(hist_df["_LOT_ID_LABEL"].unique()):
                                                  lot_data = hist_df[hist_df["_LOT_ID_LABEL"] == lbl][sel_test].dropna()
                                                  if filtered: lot_data = lot_data[(lot_data >= q_low) & (lot_data <= q_high)]
                                                  if not lot_data.empty: fig_dist.add_trace(go.Histogram(x=lot_data, name=f"Hist Lot: {lbl}", opacity=0.4, marker_color=colors[idx % len(colors)], histnorm='probability density'))
                                          if not curr_df.empty and sel_test in curr_df.columns:
                                              cd_data = curr_df[sel_test].dropna()
                                              if filtered: cd_data = cd_data[(cd_data >= q_low) & (cd_data <= q_high)]
                                              if not cd_data.empty: fig_dist.add_trace(go.Histogram(x=cd_data, name=f"Current Lot ({p_lot_id})", opacity=0.75, marker_color="#Eb5757", histnorm='probability density'))
                                          if not filtered:
                                              l_row = limits_df[limits_df["Test"] == sel_test]
                                              if not l_row.empty:
                                                  if pd.notna(l_row.iloc[0]["LSL"]): fig_dist.add_vline(x=float(l_row.iloc[0]["LSL"]), line_dash="dash", line_color="red", annotation_text="LSL")
                                                  if pd.notna(l_row.iloc[0]["USL"]): fig_dist.add_vline(x=float(l_row.iloc[0]["USL"]), line_dash="dash", line_color="red", annotation_text="USL")
                                          fig_dist.update_layout(title=f"Distribution: {sel_test} (Full Range & Limits)" if not filtered else f"Distribution: {sel_test} (99% Data subset, No Limits)", xaxis_title="Measured Value", yaxis_title="Probability Density", template="plotly_white", height=500, barmode='overlay', margin=dict(l=20, r=20, t=50, b=120), legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5))
                                          return fig_dist
                                       st.plotly_chart(build_dist_fig(filtered=True), use_container_width=True)
                                       if show_full_range: st.plotly_chart(build_dist_fig(filtered=False), use_container_width=True)
                                       st.divider()
                              
                              with tab_radar:
                                  st.subheader("📡 Probe Parameter Deviation Radar Analysis")
                                  if golden_probe and golden_probe.get("params") and current_param_stats:
                                      common_params = [p for p in golden_probe["params"] if p in current_param_stats]
                                      if len(common_params) > 2:
                                          param_variability = []
                                          for p in common_params:
                                              g_std = golden_probe["params"][p]["std"]
                                              if g_std > 1e-9:
                                                  dev = abs(current_param_stats[p]["mean"] - golden_probe["params"][p]["mean"]) / g_std
                                                  param_variability.append((p, dev))
                                          param_variability.sort(key=lambda x: x[1], reverse=True)
                                          radar_params = [p for p, _ in param_variability[:8]]
                                          if radar_params:
                                              current_vals = [(current_param_stats[p]["mean"] - golden_probe["params"][p]["mean"]) / (golden_probe["params"][p]["std"] + 1e-9) for p in radar_params]
                                              fig_radar = go.Figure()
                                              fig_radar.add_trace(go.Scatterpolar(r=[0.0] * len(radar_params), theta=radar_params, fill='toself', name='Golden Baseline', line=dict(color='#28a745', width=2), opacity=0.5))
                                              fig_radar.add_trace(go.Scatterpolar(r=current_vals, theta=radar_params, fill='toself', name='Current Wafer', line=dict(color='#dc3545', width=2), opacity=0.5))
                                              fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[-5, 5])), title="Probe Parameter Deviation (Zσ units)", height=500, showlegend=True)
                                              st.plotly_chart(fig_radar, use_container_width=True)
                                  else:
                                      st.info("Golden baseline or current parameter stats not available.")

        st.markdown("---")
        st.subheader("2) AI Yield Prediction & Virtual Golden Wafer")
        p_year = st.session_state.get("p_year", "Unknown")
        st.markdown(f"**Lot ID:** `{p_lot_id}` | **Fiscal Year:** `{p_year}`")
        
        # Try to find actual yield from the filename or folder name mapping
        actual_yield = None
        p_fname = st.session_state.get("p_filename", "")
        
        # 1. Try to extract directly from filename (e.g. "..._71.54%...")
        if p_fname:
            match_pct = re.search(r'_(\d+(?:\.\d+)?)\s*%', p_fname)
            if match_pct:
                actual_yield = float(match_pct.group(1))
                
        # 2. Map based on SYN_00XX in filename or lot ID
        if actual_yield is None:
            syn_match = re.search(r'SYN_\d{4}', p_fname or p_lot_id, re.IGNORECASE)
            if syn_match:
                syn_lot = syn_match.group(0).upper()
                ft_inv = get_ft_historical_inventory(p_generic)
                for item in ft_inv:
                    if item.get("lot_id", "").upper() == syn_lot:
                        actual_yield = item["yield"]
                        break
                if actual_yield is None:
                    probe_inv = get_historical_inventory(p_generic)
                    for item in probe_inv:
                        if item.get("lot_id", "").upper() == syn_lot:
                            actual_yield = item["yield"]
                            break

        col1, col2 = st.columns(2)
        with col1:
            if actual_yield is not None:
                st.metric("Actual Yield", f"{actual_yield:.2f}%")
            else:
                st.metric("Actual Yield", "N/A")
        with col2:
            st.metric("Predicted Yield", f"{pred * 100:.2f}%", delta=f"{(pred - baseline_yield)*100:.2f}% vs Golden")

        st.write("### 🔬 Virtual Golden Wafer Baseline Comparison")
        tab_ft_golden, tab_probe_golden = st.tabs([
            "🏭 FT: Predicted vs Golden", "🔬 Probe: Current vs Golden"
        ])
        
        with tab_probe_golden:
            if golden_probe and golden_probe.get("params"):
                st.caption(f"Golden Probe Baseline built from top 3 highest-yield lots: {', '.join(golden_probe['lot_ids'])}")
                p_fname = st.session_state.get("p_filename", "")
                actual_probe_stats = get_actual_stats_for_lot(p_generic, p_lot_id, stage="probe", probe_filename=p_fname)
                probe_comparison_df = compute_golden_comparison(current_param_stats, golden_probe, actual_stats=actual_probe_stats, limit_map=limit_map_raw)
                if not probe_comparison_df.empty:
                    def style_golden_rows(row):
                        if "🔴" in str(row.get("Status", "")): return ["background-color: #ffe0e0"] * len(row)
                        elif "⚠️" in str(row.get("Status", "")): return ["background-color: #fff3cd"] * len(row)
                        return ["background-color: #d4edda"] * len(row)
                    st.dataframe(probe_comparison_df.style.apply(style_golden_rows, axis=1), use_container_width=True, height=400)
                    n_pass = len(probe_comparison_df[probe_comparison_df["Status"].str.contains("✅")])
                    n_warn = len(probe_comparison_df[probe_comparison_df["Status"].str.contains("⚠️")])
                    n_fail = len(probe_comparison_df[probe_comparison_df["Status"].str.contains("🔴")])
                    sc1, sc2, sc3 = st.columns(3)
                    sc1.metric("Within Spec", n_pass, delta=None); sc2.metric("Marginal", n_warn, delta=None); sc3.metric("Out of Spec", n_fail, delta=None)
                else: st.info("No matching test parameters found.")
            else: st.info("No historical probe data available.")
        
        with tab_ft_golden:
            if golden_ft and golden_ft.get("params"):
                st.caption(f"Golden FT Baseline built from top 3 highest-yield lots: {', '.join(golden_ft['lot_ids'])}")
                predicted_ft_stats = {}
                if details:
                    top_model = st.session_state.get("explainability_model_sel", list(details.keys())[0])
                    if top_model not in details:
                        top_model = list(details.keys())[0]
                    top_preds = details[top_model]
                    if isinstance(top_preds, dict):
                        golden_params, golden_keys = golden_ft.get("params", {}), list(golden_ft.get("params", {}).keys())
                        for k, v in top_preds.items():
                            if k in ["FT_y", "y"] or isinstance(v, str): continue
                            try: f_val = float(v)
                            except: continue
                            k_clean = k.replace("FT_", "")
                            base, stat_type = k_clean.split("__", 1) if "__" in k_clean else (k_clean, "mean")
                            base_upper = base.upper().replace("_", ".")
                            matched_key = None
                            base_norm = normalize_colname(base)
                            for gk in golden_keys:
                                if gk.upper().replace(" ", "") == base_upper or normalize_colname(gk) == base_norm:
                                    matched_key = gk; break
                            target_key = matched_key if matched_key else k
                            if target_key not in predicted_ft_stats: predicted_ft_stats[target_key] = {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
                            if stat_type == "mean": predicted_ft_stats[target_key]["mean"] = f_val
                            elif stat_type == "std": predicted_ft_stats[target_key]["std"] = f_val
                            elif stat_type in ["median", "p50"]: predicted_ft_stats[target_key]["median"] = f_val
                            elif stat_type == "min": predicted_ft_stats[target_key]["min"] = f_val
                            elif stat_type == "max": predicted_ft_stats[target_key]["max"] = f_val
                p_fname = st.session_state.get("p_filename", "")
                actual_ft_stats = get_actual_stats_for_lot(p_generic, p_lot_id, stage="ft", probe_filename=p_fname)
                ft_comparison_df = compute_golden_comparison(predicted_ft_stats, golden_ft, actual_stats=actual_ft_stats, label_prefix="Predicted", limit_map=limit_map_raw)
                if not ft_comparison_df.empty:
                    def style_ft_rows(row):
                        if "🔴" in str(row.get("Status", "")): return ["background-color: #ffe0e0"] * len(row)
                        elif "⚠️" in str(row.get("Status", "")): return ["background-color: #fff3cd"] * len(row)
                        return ["background-color: #d4edda"] * len(row)
                    st.dataframe(ft_comparison_df.style.apply(style_ft_rows, axis=1), use_container_width=True, height=400)
                    
                    st.divider()
                    
                    # 1. Prepare list of tests and format mapping
                    ft_test_list = list(ft_comparison_df["Test Parameter"].unique())
                    try:
                        import re
                        def local_parse(x):
                            m = re.search(r'T(\d+\.?\d*)', str(x))
                            return float(m.group(1)) if m else float('inf')
                        ft_test_list.sort(key=lambda x: local_parse(x))
                    except Exception:
                        ft_test_list.sort()
                    
                    ft_names_map = golden_ft.get("tnum_to_name", {})
                    
                    # Identify parameters with Warning status (Out of Spec or Marginal)
                    warn_ft_tests = set()
                    for _, row in ft_comparison_df.iterrows():
                        t_param = row.get("Test Parameter")
                        status_val = str(row.get("Status", ""))
                        if "🔴" in status_val or "⚠️" in status_val:
                            warn_ft_tests.add(t_param)
                            
                    def fmt_ft_test(t):
                        name = ft_names_map.get(t, "")
                        label = f"{t}: {name}" if name else t
                        if t in warn_ft_tests:
                            return f"{label} [WARN]"
                        return label
                        
                    sel_ft_tests = st.multiselect(
                        "Select Final Test(s) for Detail Component", 
                        ft_test_list, 
                        default=[ft_test_list[0]] if ft_test_list else [], 
                        format_func=fmt_ft_test,
                        key="ft_detail_test_sel"
                    )
                    
                    # Try to load actual raw FT data for the current lot
                    curr_ft_raw_df = get_actual_raw_data_for_lot(p_generic, p_lot_id, stage="ft", probe_filename=p_fname)
                    
                    for sel_test in sel_ft_tests:
                        # Find matching stats in predicted & golden
                        pred_stats = predicted_ft_stats.get(sel_test, {})
                        golden_stats = golden_ft["params"].get(sel_test, {})
                        
                        if not golden_stats:
                            continue
                            
                        # If pred_stats is missing, fallback to actual or default
                        pred_mean = pred_stats.get("mean", golden_stats["mean"])
                        pred_std = pred_stats.get("std", golden_stats["std"])
                        
                        # Load golden raw series
                        golden_raw_df = golden_ft.get("raw_df", pd.DataFrame())
                        golden_series = golden_raw_df[sel_test].dropna() if (not golden_raw_df.empty and sel_test in golden_raw_df.columns) else pd.Series()
                        
                        # Load current lot's FT series (actual or simulated)
                        is_simulated = True
                        if not curr_ft_raw_df.empty and sel_test in curr_ft_raw_df.columns:
                            curr_series = curr_ft_raw_df[sel_test].dropna()
                            if not curr_series.empty:
                                is_simulated = False
                        
                        if is_simulated:
                            # Simulate normal distribution representing the prediction
                            sample_size = len(golden_series) if not golden_series.empty else 1000
                            if sample_size < 10:
                                sample_size = 1000
                            np.random.seed(42)
                            curr_series = pd.Series(np.random.normal(loc=pred_mean, scale=pred_std, size=sample_size))
                            
                        # Calculate Metrics
                        # 1. PSI
                        psi_val = _calculate_psi(golden_series, curr_series)
                        psi_status = "Normal"
                        if pd.notna(psi_val):
                            if psi_val > 0.25:
                                psi_status = "Major Shift"
                            elif psi_val >= 0.1:
                                psi_status = "Minor Shift"
                                
                        # 2. SPC median shift
                        spc_alert = False
                        if not golden_series.empty and not curr_series.empty:
                            hm = golden_series.median()
                            cm = curr_series.median()
                            q25 = golden_series.quantile(0.25)
                            q75 = golden_series.quantile(0.75)
                            iqr = q75 - q25
                            if pd.notna(hm) and pd.notna(cm) and iqr > 0:
                                if abs(cm - hm) > 1.5 * iqr:
                                    spc_alert = True
                                    
                        # 3. FT Z-Deviation
                        ft_dev = abs(curr_series.mean() - golden_stats["mean"]) / (golden_stats["std"] + 1e-9)
                        ft_alert = ft_dev > 2.0
                        
                        # 4. Limits
                        lsl, usl = np.nan, np.nan
                        if limit_map_raw:
                            from ml_compute_statistic import normalize_colname
                            norm_param = normalize_colname(sel_test)
                            limits = limit_map_raw.get(sel_test)
                            if limits is None:
                                for k, v in limit_map_raw.items():
                                    if normalize_colname(k) == norm_param:
                                        limits = v
                                        break
                            if limits is not None:
                                try:
                                    lsl, usl = pd.to_numeric(limits[0]), pd.to_numeric(limits[1])
                                except:
                                    pass
                                    
                        lsl_txt = f"{lsl:.4g}" if pd.notna(lsl) else "N/A"
                        usl_txt = f"{usl:.4g}" if pd.notna(usl) else "N/A"
                        
                        # Render metrics block
                        st.markdown(f"### 📊 Detail Analysis: {fmt_ft_test(sel_test)}")
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric(
                            f"PSI ({sel_test})", 
                            f"{psi_status}", 
                            delta=f"{psi_val:.4f}" if pd.notna(psi_val) else "N/A", 
                            delta_color="inverse" if psi_status == "Major Shift" else "normal"
                        )
                        c2.metric(
                            f"SPC ({sel_test})", 
                            "Alert" if spc_alert else "Pass", 
                            delta="Median Drift" if spc_alert else None, 
                            delta_color="inverse" if spc_alert else "normal"
                        )
                        c3.metric(
                            f"FT ({sel_test})", 
                            "Alert" if ft_alert else "Pass", 
                            delta=f"{ft_dev:.2f} Z-Dev" if pd.notna(ft_dev) else None, 
                            delta_color="inverse" if ft_alert else "normal"
                        )
                        c4.metric(
                            "Spec Limits", 
                            f"{lsl_txt} / {usl_txt}", 
                            help="LSL / USL from limits map"
                        )
                        
                        # Plot Overlay Distribution
                        def build_ft_dist_fig(filtered=True):
                            fig_dist = go.Figure()
                            colors = ["#9aa0a6", "#17a2b8", "#6f42c1", "#fd7e14", "#20c997"]
                            
                            all_vals = []
                            if not golden_series.empty:
                                all_vals.append(golden_series)
                            if not curr_series.empty:
                                all_vals.append(curr_series)
                                
                            q_low, q_high = 0, 0
                            if all_vals:
                                combined_vals = pd.concat(all_vals)
                                q_low, q_high = combined_vals.quantile(0.005), combined_vals.quantile(0.995)
                                iqr_range = q_high - q_low
                                if iqr_range > 0:
                                    q_low, q_high = q_low - 0.05 * iqr_range, q_high + 0.05 * iqr_range
                                    
                            # Add Golden/Historical lots
                            if not golden_raw_df.empty and sel_test in golden_raw_df.columns:
                                lot_cols = [col for col in ["_LOT_ID", "_LOT_ID_LABEL"] if col in golden_raw_df.columns]
                                lot_col = lot_cols[0] if lot_cols else None
                                
                                if lot_col:
                                    for idx, lbl in enumerate(golden_raw_df[lot_col].unique()):
                                        lot_data = golden_raw_df[golden_raw_df[lot_col] == lbl][sel_test].dropna()
                                        if filtered and all_vals:
                                            lot_data = lot_data[(lot_data >= q_low) & (lot_data <= q_high)]
                                        if not lot_data.empty:
                                            fig_dist.add_trace(go.Histogram(
                                                x=lot_data,
                                                name=f"Golden Lot: {lbl}",
                                                opacity=0.4,
                                                marker_color=colors[idx % len(colors)],
                                                histnorm='probability density'
                                            ))
                                else:
                                    lot_data = golden_series
                                    if filtered and all_vals:
                                        lot_data = lot_data[(lot_data >= q_low) & (lot_data <= q_high)]
                                    if not lot_data.empty:
                                        fig_dist.add_trace(go.Histogram(
                                            x=lot_data,
                                            name="Golden Baseline",
                                            opacity=0.4,
                                            marker_color="#9aa0a6",
                                            histnorm='probability density'
                                        ))
                                        
                            # Add Current Lot (Predicted or Actual)
                            if not curr_series.empty:
                                cd_data = curr_series
                                if filtered and all_vals:
                                    cd_data = cd_data[(cd_data >= q_low) & (cd_data <= q_high)]
                                    
                                label_name = f"Current Lot ({p_lot_id})"
                                if is_simulated:
                                    label_name += " (Predicted)"
                                else:
                                    label_name += " (Actual)"
                                    
                                if not cd_data.empty:
                                    fig_dist.add_trace(go.Histogram(
                                        x=cd_data,
                                        name=label_name,
                                        opacity=0.75,
                                        marker_color="#Eb5757",
                                        histnorm='probability density'
                                    ))
                                    
                            # Add LSL and USL vertical lines
                            if pd.notna(lsl):
                                fig_dist.add_vline(x=float(lsl), line_dash="dash", line_color="red", annotation_text=f"LSL ({lsl:.4g})")
                            if pd.notna(usl):
                                fig_dist.add_vline(x=float(usl), line_dash="dash", line_color="red", annotation_text=f"USL ({usl:.4g})")
                                
                            fig_dist.update_layout(
                                title=f"Distribution Comparison: {sel_test}" + (" (99% Data subset)" if filtered else " (Full Range & Limits)"),
                                xaxis_title="Measured Value",
                                yaxis_title="Probability Density",
                                template="plotly_white",
                                height=500,
                                barmode='overlay',
                                margin=dict(l=20, r=20, t=50, b=120),
                                legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5)
                            )
                            return fig_dist
                            
                        # Show 99% filtered distribution chart
                        st.plotly_chart(build_ft_dist_fig(filtered=True), use_container_width=True)
            else: st.info("No historical final test data available.")
        
        # Radar Chart moved under Raw Test Parameter Distribution Shift Analysis

        st.write("### 🌍 Multi-Level Explainability Analysis")
        shap_data = res.get("shap_data", {})
        available_models = list(shap_data.keys()) if shap_data else []
        if not available_models:
            model_dir_str = res.get("model_dir", "")
            if model_dir_str:
                model_dir = Path(model_dir_str)
                available_models = [f.stem.replace("model_", "").replace(f"_{p_generic}", "") for f in model_dir.glob(f"model_*_{p_generic}.joblib")]
        
        default_idx = 0
        for i, m_name in enumerate(available_models):
            if m_name.lower() == "xgboost": default_idx = i; break
        selected_model = st.selectbox("Select Performance Model for Interpretation Analysis:", options=available_models, index=default_idx, key="explainability_model_sel")
        
        tab_feature, tab_wafer, tab_map = st.tabs(["Feature Heatmap", "Wafer Summary", "Wafer Map (GDBN)"])
        with tab_feature:
            st.markdown("#### How Probe Parameters Affect Predicted FT Statistics")
            if shap_data:
                try:
                    import plotly.express as px
                    import numpy as np
                    
                    top_model_name = selected_model
                    if top_model_name not in shap_data:
                        st.warning(f"SHAP data not available for `{top_model_name}` in this tab.")
                        st.stop()
                        
                    s_dict = shap_data[top_model_name]
                    
                    # Safely flatten arrays in case they were nested accidentally
                    vals = np.array(s_dict["values"]).flatten()
                    feature_names = s_dict["feature_names"][:len(vals)]
                    
                    # Map TP_ feature names to readable probe parameter descriptions
                    readable_names = [get_readable_feature_name(fn) for fn in feature_names]
                    
                    df_heat = pd.DataFrame({
                        "Test Parameter": readable_names,
                        "SHAP Contribution": vals
                    })
                    
                    # Sort by absolute impact and take top 15
                    df_heat["Absolute"] = df_heat["SHAP Contribution"].abs()
                    df_heat = df_heat.sort_values(by="Absolute", ascending=False).head(15)
                    
                    # Plotly bar likes ascending order for correct top-to-bottom rendering
                    df_heat = df_heat.sort_values(by="Absolute", ascending=True)
                    
                    fig_heat = px.bar(
                        df_heat, 
                        y="Test Parameter", 
                        x="SHAP Contribution", 
                        color="SHAP Contribution",
                        color_continuous_scale=[(0, "#B71C1C"), (0.5, "#FFFF8D"), (1, "#1B5E20")],
                        color_continuous_midpoint=0,
                        title=f"How Probe Parameters Drive FT Prediction — {top_model_name}",
                        orientation="h"
                    )
                    with st.container(border=True):
                        st.plotly_chart(fig_heat, use_container_width=True)
                    
                    st.caption("🔴 Red = pushes FT yield prediction lower · 🟢 Green = pushes FT yield prediction higher")
                except Exception as e:
                    st.error(f"Could not render Feature Heatmap: {e}")
            else:
                # Permutation Importance Fallback for Feature Heatmap
                st.info("SHAP explainer not available. Using **Permutation Importance** to show probe parameter impact on FT predictions.")
                try:
                    import joblib
                    from sklearn.inspection import permutation_importance
                    import plotly.express as px
                    
                    model_dir_str = res.get("model_dir", "")
                    model_dir = Path(model_dir_str) if model_dir_str else None
                    
                    if model_dir and model_dir.exists():
                        model_files = list(model_dir.glob(f"model_*_{p_generic}.joblib"))
                        ref_path = model_dir / f"merged_features_{p_generic}.csv"
                        
                        if model_files and ref_path.exists():
                            pipe = joblib.load(model_files[0])
                            m_name = model_files[0].stem.replace("model_", "").replace(f"_{p_generic}", "")
                            
                            bg_df = pd.read_csv(ref_path)
                            meta_drop = ["y", "parent_folder", "lot_id", "file_name", "file_path"]
                            ft_cols = [c for c in bg_df.columns if str(c).startswith("FT_")]
                            feature_cols = [c for c in bg_df.columns if c not in meta_drop and c not in ft_cols]
                            X_bg = bg_df[feature_cols].apply(pd.to_numeric, errors='coerce')
                            X_bg.columns = ml_train_model.sanitize_feature_names(list(X_bg.columns))
                            
                            if hasattr(pipe, "feature_names_in_"):
                                X_bg = X_bg.reindex(columns=pipe.feature_names_in_, fill_value=0)
                            
                            # Target: use full multi-output FT_ columns to match model output dimensionality
                            if ft_cols:
                                y_bg = bg_df[ft_cols].apply(pd.to_numeric, errors='coerce')
                                y_bg = y_bg.dropna(axis=1, how='all').fillna(y_bg.median(numeric_only=True)).fillna(0.0)
                            elif "y" in bg_df.columns:
                                y_bg = pd.to_numeric(bg_df["y"], errors='coerce').fillna(0).to_frame()
                            else:
                                y_bg = None
                            
                            if y_bg is not None:
                                valid_mask = X_bg.notna().all(axis=1)
                                X_bg = X_bg[valid_mask].fillna(0)
                                y_bg = y_bg[valid_mask]
                                
                                with st.spinner("Computing Permutation Importance..."):
                                    perm_result = permutation_importance(
                                        pipe, X_bg, y_bg, n_repeats=10, random_state=42, n_jobs=1
                                    )
                                
                                imp_df = pd.DataFrame({
                                    "Feature": X_bg.columns,
                                    "Importance": perm_result.importances_mean
                                }).sort_values(by="Importance", ascending=False).head(15)
                                
                                # Add readable feature names
                                imp_df["Probe Parameter"] = imp_df["Feature"].apply(get_readable_feature_name)
                                imp_df = imp_df.sort_values(by="Importance", ascending=True)
                                
                                fig_heat = px.bar(
                                    imp_df, y="Probe Parameter", x="Importance",
                                    color="Importance", color_continuous_scale="Viridis",
                                    title=f"Probe Parameter Impact on FT Prediction — {m_name}",
                                    orientation="h"
                                )
                                fig_heat.update_layout(height=500)
                                st.plotly_chart(fig_heat, use_container_width=True)
                                st.caption("Higher importance = shuffling this probe parameter causes larger drop in FT prediction accuracy.")
                            else:
                                st.warning("No target columns found for importance analysis.")
                        else:
                            st.warning("Model or training data not found.")
                    else:
                        st.warning("Model directory not available.")
                except Exception as e:
                    st.error(f"Could not compute Feature Importance: {e}")

        with tab_wafer:
            st.markdown(f"### Wafer **{p_lot_id}** is classified as **Grade {grade} ({desc})** risk")
            st.markdown("#### Primary factors:")
            
            # Analyze SHAP top factors
            if shap_data:
                try:
                    top_model_name = selected_model
                    if top_model_name not in shap_data:
                        st.info(f"SHAP details for `{top_model_name}` are not available for this wafer instance.")
                        st.stop()
                        
                    s_dict = shap_data[top_model_name]
                    vals = np.array(s_dict.get("values", [])).flatten()
                    feature_names = list(s_dict.get("feature_names", []))
                    pair_count = min(len(feature_names), len(vals))
                    
                    if pair_count > 0:
                        df_factors = pd.DataFrame({
                            "Feature": feature_names[:pair_count],
                            "Value": vals[:pair_count]
                        }).sort_values(by="Value", key=abs, ascending=False).head(3)
                        
                        for _, row in df_factors.iterrows():
                            # Convert technical names to human-readable labels
                            fname = get_readable_feature_name(row['Feature'])
                            
                            direction = "🔻" if row['Value'] < 0 else "🔺"
                            val_str = f"decreased yield by {abs(row['Value'])*100:.2f}%" if row['Value'] < 0 else f"increased yield by {row['Value']*100:.2f}%"
                            st.markdown(f"- {direction} **{fname}** {val_str}")
                    else:
                        st.caption("SHAP factor details were unavailable for this wafer.")
                except Exception as shap_err:
                    st.caption(f"Could not summarize SHAP factors: {shap_err}")
            
            # Add spatial text
            gdbn_count = spatial_res.get("gdbn_count", 0)
            gdbn_ratio_all = spatial_res.get("gdbn_ratio_all_dies", 0.0)
            gdbn_rate_good = spatial_res.get("gdbn_rate_good_dies", 0.0)
            edge_count = spatial_res.get("edge_cluster_count", 0)
            st.markdown(
                f"- Spatial Analysis found **{gdbn_count}** GDBN high-risk dies "
                f"(**{gdbn_ratio_all:.2%}** of all tested dies; "
                f"**{gdbn_rate_good:.2%}** of probe-good dies) and "
                f"**{edge_count}** edge cluster defects."
            )
            
            st.markdown("#### Summary Predictions:")
            st.markdown(f"- **Predicted FT yield:** {pred * 100:.2f}%")
            st.markdown(f"- **Historical baseline yield:** {baseline_yield * 100:.2f}%")
            
            if details:
                top_model = st.session_state.get("explainability_model_sel", list(details.keys())[0])
                if top_model not in details:
                    top_model = list(details.keys())[0]
                top_preds = details[top_model]
                if isinstance(top_preds, dict) and len(top_preds) > 1:
                    st.markdown(f"#### Predicted Final Test Parameters (via {top_model})")
                    
                    # Separate WaferPulse metrics from FT parameters for cleaner display
                    ft_rows = []
                    wp_rows = []
                    for k, v in top_preds.items():
                        if k in ["FT_y", "y"]:
                            continue
                        
                        # Make feature names human-readable
                        if k.startswith("WP_"):
                            if k in ["WP_Radius_mm", "WP_Probe_Hits", "WP_Status", "WP_Revenue_Impact"]:
                                continue
                            label = k.replace("WP_", "").replace("_", " ").title()
                            wp_rows.append({"Metric": label, "Value": f"{v:.4g}" if isinstance(v, (int, float)) else str(v)})
                        elif k.startswith("FT_"):
                            # FT_t1_0__mean -> Test T1.0 (Mean)
                            clean = k.replace("FT_", "")
                            if "__" in clean:
                                base, stat = clean.rsplit("__", 1)
                                test_name = base.upper().replace("_", ".")
                                stat_label = stat.replace("_", " ").title()
                                label = f"Test {test_name} ({stat_label})"
                            else:
                                label = clean.upper().replace("_", ".")
                            ft_rows.append({"FT Parameter": label, "Predicted Value": f"{v:.4g}" if isinstance(v, (int, float)) else str(v)})
                        else:
                            ft_rows.append({"FT Parameter": k, "Predicted Value": f"{v:.4g}" if isinstance(v, (int, float)) else str(v)})
                    
                    if ft_rows:
                        st.dataframe(pd.DataFrame(ft_rows), use_container_width=True, hide_index=True)
                    
            
        with tab_map:
            wafer_map_df = spatial_res.get("wafer_map_df")
            if wafer_map_df is not None and not wafer_map_df.empty:
                try:
                    # Normalize col names for plotting
                    n_map = {c.strip().lower(): c for c in wafer_map_df.columns}
                    x_c, y_c = n_map.get('x'), n_map.get('y')
                    
                    if x_c and y_c:
                        # Convert GDBN flag to readable color code, preserving Pass/Fail
                        # Professional Clean Color Palette
                        status_color = {
                            "Pass": "#43A047",  # Lighter Professional Green
                            "Fail": "#D32F2F",  # Professional Red
                            "GDBN Risk": "#FBC02D" # Professional Amber
                        }
                        
                        def map_status(row):
                            if row.get('gdbn_flag'): return "GDBN Risk"
                            if not row['is_pass']: return "Fail"
                            return "Pass"
                            
                        wafer_map_df['Status'] = wafer_map_df.apply(map_status, axis=1)
                        status_order = ["Pass", "Fail", "GDBN Risk"]
                        
                        # ── Wafer Map — Spatial Bin Analysis ──
                        total_dies = len(wafer_map_df)
                        pass_dies = int((wafer_map_df['Status'] == 'Pass').sum())
                        fail_dies = int((wafer_map_df['Status'] == 'Fail').sum())
                        gdbn_dies = int((wafer_map_df['Status'] == 'GDBN Risk').sum())
                        wafer_yield = (pass_dies / total_dies * 100) if total_dies > 0 else 0
                        
                        wafer_map_df['Status'] = pd.Categorical(wafer_map_df['Status'], categories=status_order, ordered=True)
                        
                        # Wafer geometry
                        x_vals = wafer_map_df[x_c].values.astype(float)
                        y_vals = wafer_map_df[y_c].values.astype(float)
                        w_cx = (x_vals.min() + x_vals.max()) / 2
                        w_cy = (y_vals.min() + y_vals.max()) / 2
                        w_r = max(x_vals.max() - w_cx, y_vals.max() - w_cy) + 2.5
                        
                        fig_map = go.Figure()
                        
                        # ── Professional Wafer Background (Light Gray Theme) ──
                        theta = np.linspace(0, 2 * np.pi, 400)
                        # Outer Bezel
                        fig_map.add_trace(go.Scatter(
                            x=(w_cx + (w_r + 1.5) * np.cos(theta)).tolist(),
                            y=(w_cy + (w_r + 1.5) * np.sin(theta)).tolist(),
                            fill='toself', fillcolor='#EEEEEE',
                            mode='lines', line=dict(color='#BDBDBD', width=1),
                            showlegend=False, hoverinfo='skip'
                        ))
                        # Inner Surface
                        fig_map.add_trace(go.Scatter(
                            x=(w_cx + (w_r + 0.4) * np.cos(theta)).tolist(),
                            y=(w_cy + (w_r + 0.4) * np.sin(theta)).tolist(),
                            fill='toself', fillcolor='#F5F5F5',
                            mode='lines', line=dict(color='#E0E0E0', width=1),
                            showlegend=False, hoverinfo='skip'
                        ))
                        
                        # Notch (bottom center) - Professional look
                        notch_w = w_r * 0.05
                        fig_map.add_trace(go.Scatter(
                            x=[w_cx - notch_w, w_cx, w_cx + notch_w],
                            y=[w_cy - w_r - 1.2, w_cy - w_r - 0.2, w_cy - w_r - 1.2],
                            fill='toself', fillcolor='#ECEFF1',
                            mode='lines', line=dict(color='#455A64', width=1.5),
                            showlegend=False, hoverinfo='skip'
                        ))
                        
                        # Plot each category
                        for status_cat in status_order:
                            subset = wafer_map_df[wafer_map_df['Status'] == status_cat]
                            if subset.empty:
                                continue
                            hover_texts = []
                            for _, r in subset.iterrows():
                                parts = [f"<b>{'Pass (Bin 1)' if status_cat == 'Pass' else ('Fail' if status_cat == 'Fail' else 'GDBN Risk')}</b>",
                                         f"X: {int(r[x_c])}, Y: {int(r[y_c])}"]
                                if 'Device #' in subset.columns: parts.append(f"Device: {int(r['Device #'])}")
                                if 'Bin' in subset.columns: parts.append(f"Bin: {int(r['Bin'])}")
                                if 'Site' in subset.columns: parts.append(f"Site: {int(r['Site'])}")
                                hover_texts.append("<br>".join(parts))
                            fig_map.add_trace(go.Scatter(
                                x=subset[x_c].tolist(), y=subset[y_c].tolist(),
                                mode='markers',
                                name=status_cat,
                                marker=dict(
                                    symbol='square', 
                                    size=8, 
                                    color=status_color[status_cat],
                                    line=dict(width=0.4, color='rgba(0,0,0,0.15)') # Subtle dark border for light background
                                ),
                                showlegend=True, text=hover_texts, hoverinfo='text'
                            ))
                        
                        # Subtitle stats line
                        subtitle = (f"Total: {total_dies:,} dies  ·  Yield: {wafer_yield:.1f}%  ·  "
                                    f"Pass: {pass_dies:,}  ·  Fail: {fail_dies:,}  ·  GDBN: {gdbn_dies}")
                        
                        fig_map.update_layout(
                            title=dict(
                                text=f"<span style='font-size:22px; color:#263238; font-weight:bold'>Wafer Map — Spatial Bin Analysis</span><br><span style='font-size:14px; color:#546E7A'>{subtitle}</span>",
                                x=0.5, xanchor='center', y=0.98, yanchor='top'
                            ),
                            plot_bgcolor='#f8f9fa',
                            paper_bgcolor='white',
                            xaxis=dict(visible=False, range=[w_cx - w_r - 5, w_cx + w_r + 5]),
                            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1, range=[w_cy - w_r - 5, w_cy + w_r + 5]),
                            height=750,
                            margin=dict(l=10, r=10, t=110, b=10),
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=12, color="#37474F"),
                                bgcolor="rgba(255,255,255,0.8)",
                                bordercolor="#CFD8DC",
                                borderwidth=1
                            )
                        )
                        st.plotly_chart(fig_map, use_container_width=True)
                    else:
                        st.warning("Wafer Map coordinates (X, Y) are missing from the parsed payload.")
                except Exception as e:
                    st.error(f"Could not render Wafer Map: {e}")
            else:
                st.info("No detailed Wafer Map raw data available or coordinate mapping failed.")


        st.markdown("---")
        st.subheader("3) WaferPulse™ Reliability Product Grading")
        
        # 3-Column Layout for Post-Packaging Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric(
            label="Reliability Score",
            value=f"{ri_score:.2f} / 100",
            help=(
                "Grading Standards:\n\n"
                "Grade A (Automotive): Score >= 90 — Requires MTTF >= 15.0 Years.\n\n"
                "Grade B (Industrial): Score >= 80 — Requires MTTF >= 10.0 Years.\n\n"
                "Grade C (Consumer): Score >= 60 — Requires MTTF >= 5.0 Years.\n\n"
                "Grade D (Reject): Score < 60 or MTTF < 3.0 Years."
            )
        )
        grade_display = f"Grade {grade}"
        if desc:
            grade_display += f" ({desc})"
        c2.metric(
            label="Predicted Grade",
            value=grade_display,
            help=(
                "Safety Overrides (Failure Gates):\n\n"
                "Lifespan Safeguard: MTTF < 3.0 Years instantly forces Grade D (Reject).\n\n"
                "Grade A/B/C Lifespan Gates: Demands >= 15Y, 10Y, and 5Y MTTF respectively "
                "(cascades down if violated).\n\n"
                "Low Yield Safeguard: Yield < 60% instantly forces Grade D (Reject).\n\n"
                "Correlation Safeguard: Golden Correlation < 0.20 instantly forces Grade D (Reject)."
            )
        )
        c3.metric(
            label="Reliability (MTTF)",
            value=f"{wp_mttf:.2f} Years",
            help="Predicted electromigration life span under nominal operating conditions."
        )
        
        # Detail breakdown section
        with st.expander("🔍 Reliability Score Component Breakdown", expanded=True):
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                st.markdown("##### 🔬 Wafer Probe: Current vs Golden")
                st.write(f"**Golden Similarity (WP):** `{score_dict['Components']['Probe_Similarity']:.1f} / 100` *(Weight: 30%)*")
                st.write(f"- Correlation Score: `{score_dict['Components']['Probe_Correlation_Score']:.1f}/100` *(Val: {score_dict['Components']['Probe_Correlation']:.4f})*")
                st.write(f"- Z-score Closeness: `{score_dict['Components']['Probe_Z_Closeness']:.1f}/100`")
                st.caption("Similarity calculated as 50% Pearson Correlation + 50% Z-score Closeness.")
            with col_b2:
                st.markdown("##### 🏭 Final Test: Predicted vs Golden")
                st.write(f"**Golden Similarity (FT):** `{score_dict['Components']['FT_Similarity']:.1f} / 100` *(Weight: 30%)*")
                st.write(f"- Correlation Score: `{score_dict['Components']['FT_Correlation_Score']:.1f}/100` *(Val: {score_dict['Components']['FT_Correlation']:.4f})*")
                st.write(f"- Z-score Closeness: `{score_dict['Components']['FT_Z_Closeness']:.1f}/100`")
                st.caption("Similarity calculated as 50% Pearson Correlation + 50% Z-score Closeness.")
            with col_b3:
                st.markdown("##### ⏳ Lifespan & Penalty Metrics")
                st.write(f"**MTTF Score:** `{score_dict['Components']['MTTF_Score']:.1f} / 100` *(Weight: 40%)*")
                st.write(f"**Yield Penalty:** `-{score_dict['Components']['Yield_Penalty']:.1f}`")
                st.caption("MTTF achieved relative to target. Penalty deducted for high latent defect cluster risks in low-yield lots.")

        
        # --- WAFER RELIABILITY HEALTH CERTIFICATE GENERATOR ---
        st.markdown("---")
        st.write("### 📜 Automated Wafer Reliability Health Certificate")
        st.write("Generate and download an official digital-twin reliability health record of this wafer for assembly, final test, or process quality review.")

        # Identify shifted probe and FT tests relative to golden baselines (Z > 2.0)
        shifted_probe_tests = []
        if golden_probe and golden_probe.get("params") and current_param_stats:
            for param, g_stats in golden_probe["params"].items():
                c_stats = current_param_stats.get(param)
                if c_stats and g_stats.get("std", 0) > 1e-9:
                    z = abs(c_stats["mean"] - g_stats["mean"]) / g_stats["std"]
                    if z > 2.0:
                        shifted_probe_tests.append(param)
                        
        shifted_ft_tests = []
        if golden_ft and golden_ft.get("params") and predicted_ft_stats:
            for param, g_stats in golden_ft["params"].items():
                c_stats = predicted_ft_stats.get(param)
                if c_stats and g_stats.get("std", 0) > 1e-9:
                    z = abs(c_stats["mean"] - g_stats["mean"]) / g_stats["std"]
                    if z > 2.0:
                        shifted_ft_tests.append(param)

        # Set theme colors based on grade
        if grade == "A":
            primary_color = "#10B981"  # Emerald
            badge_bg = "rgba(16, 185, 129, 0.08)"
            status_text = "GRADE A: PASSED - AUTOMOTIVE CLASS"
            status_desc = "Approved for mission-critical automotive assembly. Fit for high-temperature, continuous-operation safety-critical deployment."
        elif grade == "B":
            primary_color = "#3B82F6"  # Blue
            badge_bg = "rgba(59, 130, 246, 0.08)"
            status_text = "GRADE B: PASSED - INDUSTRIAL CLASS"
            status_desc = "Approved for industrial-grade servers, cloud hardware, and telecom infrastructure. Suitable for continuous workload environments."
        elif grade == "C":
            primary_color = "#F59E0B"  # Amber
            badge_bg = "rgba(245, 158, 11, 0.08)"
            status_text = "GRADE C: PASSED - CONSUMER CLASS"
            status_desc = "Approved for consumer electronics, standard mobile devices, and personal computing systems under nominal thermal environments."
        else:
            primary_color = "#EF4444"  # Red
            badge_bg = "rgba(239, 68, 68, 0.08)"
            status_text = "GRADE D: REJECTED - HIGH RISK"
            status_desc = "REJECTED. Failed reliability threshold. NOT recommended for high-reliability assembly."
            
            # Format and append shifted tests details
            details_list = []
            if shifted_probe_tests:
                probe_str = ", ".join(sorted(shifted_probe_tests)[:5])
                if len(shifted_probe_tests) > 5:
                    probe_str += f" (+{len(shifted_probe_tests) - 5} more)"
                details_list.append(f"Wafer Probe Golden shift on {probe_str}")
            if shifted_ft_tests:
                ft_str = ", ".join(sorted(shifted_ft_tests)[:5])
                if len(shifted_ft_tests) > 5:
                    ft_str += f" (+{len(shifted_ft_tests) - 5} more)"
                details_list.append(f"Final Test Golden shift on {ft_str}")
                
            if details_list:
                status_desc += " [Detected process deviations - " + " | ".join(details_list) + "]"

        # Cryptographic Hash to simulate authenticity
        import hashlib
        from datetime import datetime
        
        cert_date = res.get("creation_date")
        if not cert_date:
            cert_date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            
        cert_hash = hashlib.sha256(f"{p_lot_id}-{p_generic}-{ri_score}".encode()).hexdigest()[:16].upper()
        yield_color = "#EF4444" if pred < 0.60 else "#0F172A"

        cert_root_cause_html = ""
        if grade == "D":
            cert_root_cause_html = f"""
            <div style="margin-top: 20px; border-top: 1px dashed rgba(0, 0, 0, 0.08); padding-top: 15px;">
                <div style="font-size: 12px; font-weight: 700; color: {primary_color}; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; display: flex; align-items: center;">
                    <span style="font-size: 16px; margin-right: 6px;">⚙️</span> Root Cause Action Recommendations
                </div>
                <div style="display: grid; grid-template-columns: 1fr; gap: 10px;">
                    <div style="display: flex; align-items: flex-start; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px; text-align: left;">
                        <div style="flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; margin-right: 10px; background-color: #e3f2fd; color: #0d47a1; border: 1px solid #bbdefb; font-weight: 700;">🔍</div>
                        <div>
                            <div style="font-weight: 600; font-size: 12.5px; color: #0F172A;">1. Check if known failure</div>
                            <div style="font-size: 11.5px; color: #64748B; margin-top: 2px;">Audit historical failure logs and matching signature databases.</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: flex-start; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px; text-align: left;">
                        <div style="flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; margin-right: 10px; background-color: #fff3e0; color: #e65100; border: 1px solid #ffe0b2; font-weight: 700;">🔄</div>
                        <div>
                            <div style="font-weight: 600; font-size: 12.5px; color: #0F172A;">2. Retest to verify issue</div>
                            <div style="font-size: 11.5px; color: #64748B; margin-top: 2px;">Rerun with Golden Wafer, check Correlation Unit, or swap hardware.</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: flex-start; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px; text-align: left;">
                        <div style="flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; margin-right: 10px; background-color: #ffebee; color: #c62828; border: 1px solid #ffcdd2; font-weight: 700;">🎯</div>
                        <div>
                            <div style="font-weight: 600; font-size: 12.5px; color: #0F172A;">3. Identify shifted test</div>
                            <div style="font-size: 11.5px; color: #64748B; margin-top: 2px;">Analyze SPC and distribution charts below to isolate the parameter drift.</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: flex-start; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 6px; padding: 10px 12px; text-align: left;">
                        <div style="flex-shrink: 0; width: 24px; height: 24px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 11px; margin-right: 10px; background-color: #e8f5e9; color: #1b5e20; border: 1px solid #c8e6c9; font-weight: 700;">📊</div>
                        <div>
                            <div style="font-weight: 600; font-size: 12.5px; color: #0F172A;">4. Create summary report</div>
                            <div style="font-size: 11.5px; color: #64748B; margin-top: 2px;">Compile statistical summaries of the current lot vs golden baselines.</div>
                        </div>
                    </div>
                </div>
            </div>
            """.replace("\n", "")

        cert_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Wafer Reliability Health Certificate - {p_lot_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=Outfit:wght@400;600;800&display=swap');
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #FFFFFF;
            color: #1E293B;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 40px 0;
        }}
        .certificate-container {{
            background: #FFFFFF;
            border: 3px double {primary_color};
            border-radius: 20px;
            padding: 50px;
            width: 800px;
            position: relative;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.06);
            box-sizing: border-box;
        }}
        .certificate-container::before {{
            content: '';
            position: absolute;
            top: 15px;
            left: 15px;
            right: 15px;
            bottom: 15px;
            border: 1px solid rgba(0,0,0,0.03);
            border-radius: 12px;
            pointer-events: none;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .header h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 28px;
            font-weight: 800;
            letter-spacing: 2px;
            margin: 0 0 10px 0;
            background: linear-gradient(to right, #0F172A, {primary_color});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header p {{
            font-size: 14px;
            color: #64748B;
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 3px;
        }}
        .badge {{
            display: inline-block;
            border: 2px solid {primary_color};
            color: {primary_color};
            background: {badge_bg};
            font-family: 'Outfit', sans-serif;
            font-size: 18px;
            font-weight: 700;
            padding: 8px 24px;
            border-radius: 30px;
            margin-bottom: 30px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 14px;
            font-weight: 600;
            color: {primary_color};
            text-transform: uppercase;
            letter-spacing: 2px;
            border-bottom: 1px solid #E2E8F0;
            padding-bottom: 8px;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }}
        .meta-item {{
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px 16px;
        }}
        .meta-label {{
            font-size: 11px;
            color: #64748B;
            text-transform: uppercase;
            margin-bottom: 4px;
            letter-spacing: 1px;
        }}
        .meta-value {{
            font-size: 15px;
            font-weight: 600;
            color: #0F172A;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 20px 15px;
            text-align: center;
        }}
        .stat-val {{
            font-size: 24px;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 4px;
        }}
        .stat-label {{
            font-size: 11px;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .rec-box {{
            background: {badge_bg};
            border-left: 4px solid {primary_color};
            border-radius: 0 10px 10px 0;
            padding: 20px;
            margin-bottom: 35px;
        }}
        .rec-title {{
            font-size: 13px;
            font-weight: 700;
            color: {primary_color};
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .rec-desc {{
            font-size: 13px;
            line-height: 1.5;
            color: #334155;
            margin: 0;
        }}
        .footer {{
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px dashed #E2E8F0;
        }}
        .sign-box {{
            text-align: left;
        }}
        .signature {{
            font-family: 'Outfit', sans-serif;
            font-size: 16px;
            font-weight: 600;
            color: #0F172A;
            margin-bottom: 4px;
        }}
        .sign-title {{
            font-size: 10px;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .verification {{
            text-align: right;
        }}
        .hash {{
            font-family: monospace;
            font-size: 11px;
            color: #64748B;
            margin-bottom: 4px;
        }}
        .verification-label {{
            font-size: 10px;
            color: #475569;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
    </style>
</head>
<body>
    <div class="certificate-container">
        <div class="header">
            <h1>WAFER RELIABILITY Health Certificate</h1>
            <p>Process Control Verification</p>
        </div>
        
        <div style="text-align: center;">
            <div class="badge">{status_text}</div>
        </div>

        <div class="section-title">Wafer Metadata</div>
        <div class="meta-grid">
            <div class="meta-item">
                <div class="meta-label">Wafer ID</div>
                <div class="meta-value">{p_lot_id}-01</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Product Generic / Part</div>
                <div class="meta-value">{p_generic} / {p_partname or 'N/A'}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Testing Timestamp</div>
                <div class="meta-value">{cert_date}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Verification Type</div>
                <div class="meta-value">Probe-to-Final Physics Prediction</div>
            </div>
        </div>

        <div class="section-title">Reliability & Yield Metrics</div>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-val" style="color: {primary_color};">{ri_score:.1f}</div>
                <div class="stat-label">Reliability Index</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: {yield_color};">{pred * 100:.2f}%</div>
                <div class="stat-label">Predicted Yield</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: #3B82F6;">{wp_mttf:.2f} Yrs</div>
                <div class="stat-label">Expected Lifespan</div>
            </div>
        </div>

        <div class="section-title">Downstream Quality Review</div>
        <div class="rec-box">
            <div class="rec-title">Downstream Routing & Assembly Recommendation</div>
            <p class="rec-desc">{status_desc}</p>
            {cert_root_cause_html}
        </div>

        <div class="footer">
            <div class="sign-box">
                <div class="signature">WaferPulse™ AI Gate</div>
                <div class="sign-title">Automated Signature Verification</div>
            </div>
            <div class="verification">
                <div class="hash">SECURE HASH: {cert_hash}</div>
                <div class="verification-label">Performance Authenticity Certified</div>
            </div>
        </div>
    </div>
</body>
</html>"""

        col_c1, col_c2 = st.columns([1, 1])
        with col_c1:
            st.download_button(
                label="📥 Download Reliability Certificate (HTML)",
                data=cert_html,
                file_name=f"Wafer_Reliability_Certificate_{p_lot_id}.html",
                mime="text/html",
                type="primary",
                use_container_width=True
            )
        with col_c2:
            show_preview = st.toggle("📄 Show Certificate Preview", value=False)
            
        if show_preview:
            st.components.v1.html(cert_html, height=780, scrolling=True)




