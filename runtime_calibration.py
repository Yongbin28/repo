import json
import os
from pathlib import Path
from datetime import datetime

CALIBRATION_FILE = Path("C:/Users/user/.gemini/antigravity/knowledge/runtime_calibration.json")

class RuntimeCalibrator:
    """
    A dynamic training runtime calibration system.
    Tracks historical training runs and calibrates estimate coefficients based on hardware performance.
    """
    
    @staticmethod
    def get_calibration_data() -> dict:
        """Loads or initializes calibration data."""
        default_data = {
            "runs": [
                {
                    "timestamp": "2026-05-20T23:46:22+08:00",
                    "num_rows": 25,
                    "num_features": 525,
                    "num_targets": 526,
                    "actual_seconds": 231.0,
                    "notes": "User reference run (3m 51s)"
                }
            ],
            "calibration_factor": 3.346e-5  # Default based on 231s / (25 * 525 * 526)
        }
        
        if not CALIBRATION_FILE.exists():
            # Ensure the directory exists
            CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(CALIBRATION_FILE, "w") as f:
                    json.dump(default_data, f, indent=2)
            except Exception:
                pass
            return default_data
            
        try:
            with open(CALIBRATION_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return default_data

    @classmethod
    def get_calibration_factor(cls) -> float:
        """Returns the current calibration factor, defaulting to 3.346e-5."""
        data = cls.get_calibration_data()
        return data.get("calibration_factor", 3.346e-5)

    @classmethod
    def estimate_training_time(cls, num_rows: int, num_features: int, num_targets: int) -> float:
        """
        Estimates model training duration (in seconds) dynamically.
        Uses scaling law T = K * N_rows * N_features * N_targets.
        """
        k = cls.get_calibration_factor()
        # Scale calculation
        est_seconds = k * max(num_rows, 1) * max(num_features, 1) * max(num_targets, 1)
        # Apply reasonable boundaries (e.g. min 10s, max 10 mins)
        return float(max(10.0, min(est_seconds, 600.0)))

    @classmethod
    def log_training_run(cls, num_rows: int, num_features: int, num_targets: int, actual_seconds: float) -> float:
        """
        Logs a training run and re-calibrates the system dynamically.
        Returns the new calibrated factor.
        """
        data = cls.get_calibration_data()
        
        # Add new run entry
        new_run = {
            "timestamp": datetime.now().isoformat(),
            "num_rows": num_rows,
            "num_features": num_features,
            "num_targets": num_targets,
            "actual_seconds": actual_seconds
        }
        data.setdefault("runs", []).append(new_run)
        
        # Re-calibrate: Calculate K for each run and average them
        k_values = []
        for r in data["runs"]:
            ops = r["num_rows"] * r["num_features"] * r["num_targets"]
            if ops > 0 and r["actual_seconds"] > 0:
                k_values.append(r["actual_seconds"] / ops)
                
        new_k = sum(k_values) / len(k_values) if k_values else 3.346e-5
        data["calibration_factor"] = new_k
        
        try:
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
            
        return new_k
