import sys
# Fix PyInstaller multiprocessing child process argument parsing crash
if "--multiprocessing-fork" in sys.argv:
    try:
        idx = sys.argv.index("--multiprocessing-fork")
        sys.argv = [sys.argv[0]] + sys.argv[idx:]
    except ValueError:
        pass

import multiprocessing
multiprocessing.freeze_support()

import threading
import time
import subprocess
import os
from pathlib import Path

import socket

def is_port_open(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def resolve_path(path):
    if getattr(sys, "frozen", False):
        # PyInstaller's internal path
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).parent
    return str((base / path).resolve())

if __name__ == "__main__":
    # 0. Handle Script Execution Mode (for Decryptors etc.)
    # When frozen, this allows the EXE to act as a python interpreter for bundled scripts.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        candidate = sys.argv[1]
        
        # 1. Try to run as a physical file on disk (for external scripts)
        if candidate.endswith((".py", ".pyc")) and os.path.exists(candidate):
            script_to_run = candidate
            # Shift argv to make the script think it's the main entry
            sys.argv = sys.argv[1:]
            try:
                with open(script_to_run, "rb") as _sf:
                    _src = _sf.read()
                _code = compile(_src, script_to_run, "exec")
                _globals = {"__name__": "__main__", "__file__": script_to_run}
                exec(_code, _globals)
                sys.exit(0)
            except Exception as e:
                print(f"Error running script {script_to_run}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)
        
        # 2. Try to run as a bundled module by name (for internal scripts in PYZ)
        # Only intercept known pipeline scripts and decryptors.
        mod_name = candidate.replace(".py", "").replace(".pyc", "")
        
        PIPELINE_MODULES = [
            "stdf_decryptor", "ml_train_model",
            "ml_compute_statistic", "wafer_data_combiner", "ml_yield_prediction",
            "partname_mapping", "app_summary"
        ]
        
        if mod_name in PIPELINE_MODULES:
            try:
                import importlib
                # Ensure _internal or EXE dir is in path
                internal_dir = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(sys.executable)))
                if internal_dir not in sys.path:
                    sys.path.insert(0, internal_dir)
                
                mod = importlib.import_module(mod_name)
                sys.argv = sys.argv[1:]
                
                if hasattr(mod, "main"):
                    mod.main()
                    sys.exit(0)
                else:
                    print(f"Module {mod_name} found but has no main() function.", file=sys.stderr)
                    sys.exit(1)
            except ImportError:
                pass # Continue to normal app startup
            except Exception as e:
                print(f"Error executing module {mod_name}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)

    # Setup Logging in a non-dist folder to avoid build locks
    user_home = Path.home()
    app_data = user_home / ".tp_wafer_quality_gate"
    app_data.mkdir(exist_ok=True)
    log_path = app_data / "desktop_app_log.txt"
    
    # 1. Handle Streamlit Subprocess Mode
    if len(sys.argv) > 1 and sys.argv[1] == "run_streamlit":
        try:
            import streamlit.web.cli as stcli
            # Also need to add the internal folder to path for hidden imports
            internal_dir = resolve_path(".")
            if internal_dir not in sys.path:
                sys.path.append(internal_dir)
                
            sys.argv = [
                "streamlit",
                "run",
                resolve_path("streamlit_app.py"),
                "--server.headless=true",
                "--global.developmentMode=false",
                "--browser.gatherUsageStats=false",
                "--server.port=8501",
            ]
            sys.exit(stcli.main())
        except Exception as e:
            with open(log_path, "a") as f:
                f.write(f"FATAL ERROR in run_streamlit: {e}\n")
                import traceback
                traceback.print_exc(file=f)
            sys.exit(1)

    # 2. Main GUI Mode
    with open(log_path, "w") as log_file:
        log_file.write(f"Starting app at {time.ctime()}\n")
        log_file.write(f"Frozen: {getattr(sys, 'frozen', False)}\n")
        log_file.write(f"MEIPASS: {getattr(sys, '_MEIPASS', 'N/A')}\n")
        
        # Start streamlit in a background process
        cmd = [sys.executable, "run_streamlit"]
        env = os.environ.copy()
        # Ensure the bundle path is visible if frozen
        if getattr(sys, "frozen", False):
            env["ST_FROZEN"] = "1"
            
        # CREATE_NO_WINDOW (0x08000000) avoids some GUI-related issues on launch
        creationflags = 0x08000000 if sys.platform == "win32" else 0
        st_proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, env=env, creationflags=creationflags)

        # Wait for the port to be ready (up to 45 seconds)
        ready = False
        for i in range(45):
            if is_port_open(8501):
                ready = True
                log_file.write(f"Port 8501 ready after {i} seconds\n")
                break
            time.sleep(1)

        if not ready:
            log_file.write("ERROR: Server took too long to start.\n")

    import webbrowser
    import tkinter as tk
    from tkinter import font as tkfont

    # Auto-open in browser when ready
    if ready:
        time.sleep(0.5)  # Small buffer to ensure browser can connect
        webbrowser.open("http://localhost:8501")

    # Minimal control window to keep the process alive and give user a handle
    root = tk.Tk()
    root.title("Wafer Quality Gate")
    root.geometry("380x160")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    try:
        root.iconbitmap(resolve_path("app_icon.ico"))
    except Exception:
        pass

    status_text = "✓ Server running  —  app opened in your browser." if ready else "✗ Server failed to start. Check desktop_app_log.txt."
    status_color = "#A6e3a1" if ready else "#f38ba8"

    tk.Label(root, text="Wafer Quality Gate", bg="#1e1e2e", fg="#cdd6f4",
             font=("Segoe UI", 13, "bold")).pack(pady=(18, 2))
    tk.Label(root, text=status_text, bg="#1e1e2e", fg=status_color,
             font=("Segoe UI", 9)).pack(pady=(0, 12))

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack()

    def open_browser():
        webbrowser.open("http://localhost:8501")

    def on_closing():
        if st_proc:
            try:
                # Kill the subprocess group if on windows to ensure sub-children are gone
                if sys.platform == "win32":
                    subprocess.run(f"taskkill /F /T /PID {st_proc.pid}", shell=True, capture_output=True)
                else:
                    st_proc.kill()
                    st_proc.wait()
            except Exception:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    tk.Button(btn_frame, text="Open in Browser", command=open_browser,
              bg="#89b4fa", fg="#1e1e2e", relief="flat", padx=14, pady=6,
              font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left", padx=6)
    tk.Button(btn_frame, text="Stop Server", command=on_closing,
              bg="#F38ba8", fg="#1e1e2e", relief="flat", padx=14, pady=6,
              font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left", padx=6)

    root.mainloop()
