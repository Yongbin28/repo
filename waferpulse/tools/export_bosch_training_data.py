"""Export decoded and engineered Bosch plasma-etch training datasets.

The source NetCDF stores process values as dictionary indexes.  This utility
decodes those indexes into physical signal values, then produces the same
per-wafer summary features used by the benchmark.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset

from waferpulse.experiments.bosch_plasma_etch_benchmark import extract_process_features


def experiment_key(group_name: str) -> str:
    """Return the benchmark's date-and-wafer identifier for a NetCDF group."""

    parts = group_name.split("_")
    return f"{parts[1]}-{parts[2]}-{parts[3]}_{int(parts[5]):02d}"


def export_decoded(process_path: Path, dictionary_path: Path, output_path: Path) -> int:
    """Write all decoded 5 Hz process samples as a wide CSV table."""

    with Dataset(dictionary_path) as dictionary:
        decoder = np.asarray(dictionary.variables["data"][:], dtype=float)

    rows = 0
    first_chunk = True
    with Dataset(process_path) as source:
        for group_name, group in source.groups.items():
            encoded = np.asarray(group.variables["data"][:], dtype=np.uint16)
            signal_names = [str(name) for name in group.variables["feature"][:]]
            frame = pd.DataFrame(decoder[encoded], columns=signal_names)
            frame.insert(0, "timestamp_seconds", np.asarray(group.variables["times"][:], dtype=float))
            frame.insert(0, "experiment_key", experiment_key(group_name))
            frame.to_csv(output_path, mode="w" if first_chunk else "a", header=first_chunk, index=False)
            first_chunk = False
            rows += len(frame)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = (
        Path("dataset/bosch_plasma_etch")
        if Path("dataset/bosch_plasma_etch").is_dir()
        else Path("data/bosch_plasma_etch")
    )
    parser.add_argument("--data-root", type=Path, default=default_data)
    args = parser.parse_args()

    root = args.data_root.resolve()
    decoded_path = root / "bosch_decoded_process_traces.csv"
    feature_csv_path = root / "bosch_process_wafer_features.csv"
    process_path = root / "Process_data.nc"
    dictionary_path = root / "Dictionary_process.nc"

    sample_rows = export_decoded(process_path, dictionary_path, decoded_path)
    features = extract_process_features(process_path, dictionary_path)
    features.to_csv(feature_csv_path, index=False)
    print(f"Decoded samples: {sample_rows:,} -> {decoded_path}")
    print(f"Wafer feature rows: {len(features):,} -> {feature_csv_path}")


if __name__ == "__main__":
    main()
