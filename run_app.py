import os
import sys

# Fix PyInstaller multiprocessing child process argument parsing crash
if "--multiprocessing-fork" in sys.argv:
    try:
        idx = sys.argv.index("--multiprocessing-fork")
        sys.argv = [sys.argv[0]] + sys.argv[idx:]
    except ValueError:
        pass

import multiprocessing
import streamlit.web.cli as stcli
from pathlib import Path

# REQUIRED FOR subprocess + ThreadPoolExecutor
multiprocessing.freeze_support()

def resolve_path(path):
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path(__file__).parent
    return str((base / path).resolve())

if __name__ == "__main__":

    sys.argv = [
        "streamlit",
        "run",
        resolve_path("streamlit_app.py"),
        "--global.developmentMode=false",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]

    sys.exit(stcli.main())
