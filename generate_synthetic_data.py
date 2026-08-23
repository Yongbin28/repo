import pandas as pd
import numpy as np
import os
import random
import string
import time
from datetime import datetime

# File definition
LIMITS_FILE = 'ddr4_limits.csv'
NUM_DEVICES = 3338

# Base directories for T&P (Probe) and FT (Final Test)
BASE_TP_DIR = os.path.join('dataset', 'ETS364B', 'DDR4SDRAM_MT40A1G8', 'T&P_Decrypted')
BASE_FT_DIR = os.path.join('dataset', 'ETS364B', 'DDR4SDRAM_MT40A1G8', 'FT_Decrypted')

# Read limits
limits_df = pd.read_csv(LIMITS_FILE)

# Handle blank limits
limits_df['STDF Min'] = pd.to_numeric(limits_df['STDF Min'], errors='coerce')
limits_df['STDF Max'] = pd.to_numeric(limits_df['STDF Max'], errors='coerce')

limits_df['STDF Min'] = limits_df['STDF Min'].fillna(0)
limits_df['STDF Max'] = limits_df['STDF Max'].fillna(limits_df['STDF Min'] + 10)

def generate_base_data(type_num, set_num, lot_id_hash=0):
    data = {}
    fails_tracker = np.full(NUM_DEVICES, False)
    failed_test_str = np.full(NUM_DEVICES, "", dtype=object)

    # Hash-based macro variance for the whole lot so we have genuine signal
    np.random.seed(lot_id_hash + set_num)

    for _, row in limits_df.iterrows():
        t_num = row['Test Number']
        t_min = row['STDF Min']
        t_max = row['STDF Max']
        center = (t_max + t_min) / 2.0
        width = t_max - t_min
        std = width / 6.0  # Naturally gives around 90-95% yield without clipping
        
        # Marginal to failing shifts
        if type_num == 2 and set_num == 1 and t_num == "T3.0":
            loc = t_max - 0.67 * std  # target ~75%
        elif type_num == 2 and set_num == 2 and t_num == "T3.1":
            loc = t_min + 0.84 * std  # target ~80%
        elif type_num == 2 and set_num == 3 and t_num == "T1.0":
            loc = t_max - 0.52 * std  # target ~70%
            
        elif type_num == 3 and set_num == 1 and t_num == "T2.0":
            loc = t_max               # target ~50%
        elif type_num == 3 and set_num == 2 and t_num == "T4.0":
            loc = t_min + 0.25 * std  # target ~60%
        elif type_num == 3 and set_num == 3 and t_num == "T5.0":
            loc = t_max - 0.12 * std  # target ~55%
        else:
            loc = center

        # Add genuine LOT level variance so models have a macro signal to regress!
        lot_macro_shift = np.random.uniform(-0.6, 0.6) * std
        loc += lot_macro_shift

        test_data = np.random.normal(loc=loc, scale=std, size=NUM_DEVICES)
        data[t_num] = test_data
        
    np.random.seed() # Reset seed

    for t_num, test_data in data.items():
        t_row = limits_df[limits_df['Test Number'] == t_num].iloc[0]
        t_min = t_row['STDF Min']
        t_max = t_row['STDF Max']
        
        failed_mask = (test_data < t_min) | (test_data > t_max)
        fails_tracker = fails_tracker | failed_mask
        
        new_fails = failed_mask & (failed_test_str == "")
        failed_test_str[new_fails] = [f"Fails: {t_num}" for _ in range(new_fails.sum())]

    # Generate realistic circular wafer map coordinates dynamically based on NUM_DEVICES
    num_pts = 0
    r_val = 10.0
    while num_pts < NUM_DEVICES:
        r_val += 1.0
        cx, cy = int(r_val) + 2, int(r_val) + 2
        grid_max = int(2 * r_val) + 4
        points = [(x, y) for x in range(1, grid_max) for y in range(1, grid_max) if (x - cx)**2 + (y - cy)**2 <= r_val**2]
        num_pts = len(points)
        
    coords = points[:NUM_DEVICES]
    
    df = pd.DataFrame({
        'Device #': np.arange(1, NUM_DEVICES + 1),
        'Bin': np.where(fails_tracker, 9, 1),
        'Site': np.tile([1, 2], NUM_DEVICES // 2 + 1)[:NUM_DEVICES],
        'X': [p[0] for p in coords],
        'Y': [p[1] for p in coords],
        'Fails': failed_test_str,
        'Alarms': np.full(NUM_DEVICES, "")
    })
    
    for t_num in limits_df['Test Number']:
        df[t_num] = data[t_num]

    return df

def generate_final_from_probe(probe_df):
    final_df = probe_df.copy()
    num_devices = len(probe_df)
    fails_tracker = np.full(num_devices, False)
    failed_test_str = np.full(num_devices, "", dtype=object)
    
    # Simulating that only specific tests show package shift or thermal drift
    shift_tests = ["T3.0", "T4.1", "T5.2"]
    
    for t_num in limits_df['Test Number']:
        if t_num not in final_df.columns:
            continue
        t_row = limits_df[limits_df['Test Number'] == t_num].iloc[0]
        t_min = t_row['STDF Min']
        t_max = t_row['STDF Max']
        center = (t_max + t_min) / 2.0
        width = t_max - t_min
        if width <= 0:
            width = 1.0
            
        test_data = final_df[t_num].values
        
        # 1. Decoupled fails calculation: low noise physical model to keep yields stable and high
        clean_drift_factor = 0.08 if t_num in shift_tests else 0.0
        clean_drift = (test_data - center) * clean_drift_factor
        clean_noise = test_data * np.random.normal(0, 0.001, size=num_devices)
        clean_shifted = test_data + clean_drift + clean_noise
        
        failed_mask = (clean_shifted < t_min) | (clean_shifted > t_max)
        fails_tracker = fails_tracker | failed_mask
        new_fails = failed_mask & (failed_test_str == "")
        failed_test_str[new_fails] = [f"Fails: {t_num}" for _ in range(new_fails.sum())]
        
        # 2. Calibrated parametric data: non-linear patterns + lot noise for R2 targetting (XGBoost 0.8-0.9, linear/simpler 0.5-0.8)
        x = (test_data - center) / (width / 2.0)
        
        if t_num in shift_tests:
            # High non-linearity: linear term + sine term + quadratic term
            y_det = 0.55 * x + 0.22 * np.sin(2.5 * x) + 0.12 * (x ** 2)
            
            # Interaction term with the first test in limits_df
            ref_t_num = limits_df['Test Number'].iloc[0]
            if ref_t_num in final_df.columns and ref_t_num != t_num:
                ref_data = final_df[ref_t_num].values
                ref_center = (limits_df[limits_df['Test Number'] == ref_t_num]['STDF Max'].iloc[0] + limits_df[limits_df['Test Number'] == ref_t_num]['STDF Min'].iloc[0]) / 2.0
                ref_width = limits_df[limits_df['Test Number'] == ref_t_num]['STDF Max'].iloc[0] - limits_df[limits_df['Test Number'] == ref_t_num]['STDF Min'].iloc[0]
                if ref_width <= 0:
                    ref_width = 1.0
                x_ref = (ref_data - ref_center) / (ref_width / 2.0)
                y_det += 0.18 * x * x_ref
                
            # Irreducible lot-level variance (4% relative standard deviation) + 1% die-level noise
            lot_noise = np.random.normal(0, 0.04)
            die_noise = np.random.normal(0, 0.01, size=num_devices)
            shifted_data = center + (width / 2.0) * (y_det + lot_noise + die_noise)
        else:
            # Standard tests: mostly linear but with slight non-linearity and minor lot-level noise
            y_det = 0.85 * x + 0.08 * np.sin(1.2 * x)
            lot_noise = np.random.normal(0, 0.02)
            die_noise = np.random.normal(0, 0.004, size=num_devices)
            shifted_data = center + (width / 2.0) * (y_det + lot_noise + die_noise)
            
        final_df[t_num] = shifted_data
        
    final_df['Bin'] = np.where(fails_tracker, 9, 1)
    final_df['Fails'] = failed_test_str
    return final_df

def write_datalog(df, folder_path, filename, year_override=None):
    filepath = os.path.join(folder_path, filename)
    now = datetime.now()
    if year_override:
        # Construct a datetime-like string with the overriding year
        now_str = now.strftime(f"%m/%d/{year_override} %H:%M:%S")
    else:
        now_str = now.strftime("%m/%d/%Y %H:%M:%S")
    
    with open(filepath, 'w', newline='') as f:
        # Row 1
        f.write(f"Created from DLOG: C:\\tester_data\\{filename}\n")
        # Row 2
        f.write("Test Program: DDR4_TEST_PROG_V1\n")
        # Row 3
        f.write(f"File Creation Date: {now_str}\n")
        # Row 4
        f.write(",,,,,,," + ",".join(limits_df['Test Number']) + "\n")
        # Row 5
        f.write(",,,,,,," + ",".join(limits_df['Units'].fillna("")) + "\n")
        # Row 6
        f.write("Device #,Bin,Site,X,Y,Fails,Alarms," + ",".join(limits_df['Test Description'].str.replace(',', '')) + "\n")
        
    # Row 7+: Append CSV data
    df.to_csv(filepath, mode='a', header=False, index=False)
    print(f"Generated {filepath}")
    time.sleep(0.1)

def synt_afm():
    """
    WaferPulse Phase 0 Simulator (AFM Logic)
    Generates training data for the MTTF prediction model.
    """
    print("[INFO] Initiating WaferPulse Phase 0 Simulator...")
    num_lots = 40
    data_rows = []

    for lot_idx in range(num_lots):
        lot_id = f"HB{random.randint(1000, 9999)}P"
        # 30% of lots are "Low Yield" for training diversity
        is_low_yield = random.random() < 0.3
        
        for wafer_idx in range(1, 26):
            # Target yield for this wafer
            target_yield = random.uniform(65, 80) if is_low_yield else random.uniform(92, 99)
            
            for die_idx in range(1, 300):
                # Die positioning
                radius = random.uniform(5, 149)
                # Resistance sensor logic
                # 20% damaged dies ONLY for low yield dataset
                if is_low_yield and random.random() < 0.2:
                    res_mean = random.uniform(1.2, 1.8)
                    res_std = random.uniform(0.15, 0.45) # High variance -> Damaged
                    void_pct = random.uniform(35, 65)    # High void %
                else:
                    res_mean = random.uniform(0.8, 1.1)
                    res_std = random.uniform(0.01, 0.04) # Low variance -> Healthy
                    void_pct = random.uniform(0, 5)      # Low void %

                data_rows.append({
                    "lot_id": lot_id,
                    "wafer_id": f"{lot_id}-{wafer_idx:02d}",
                    "die_id": f"D{die_idx}",
                    "Resistance_mean": res_mean,
                    "Resistance_median": res_mean + random.uniform(-0.05, 0.05),
                    "Resistance_std": res_std,
                    "Radius_mm": radius,
                    "y": void_pct, # Target for the ML model (Void %)
                    "Yield_Actual": target_yield
                })
    
    df_wp = pd.DataFrame(data_rows)
    # Save to standard training path
    out_path = "merged_features_WaferPulse.csv"
    df_wp.to_csv(out_path, index=False)
    print(f"[SUCCESS] WaferPulse Simulator generated {len(df_wp)} die-level records at {out_path}")

def main():
    global LIMITS_FILE, NUM_DEVICES, BASE_TP_DIR, BASE_FT_DIR, limits_df
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--waferpulse", action="store_true", help="Run the WaferPulse Phase 0 Simulator")
    parser.add_argument("--lots", type=int, default=50, help="Number of standard datalog lots to generate")
    parser.add_argument("--limits", type=str, default="ddr4_limits.csv", help="Limits CSV file")
    parser.add_argument("--devices", type=int, default=3338, help="Number of devices per lot")
    parser.add_argument("--tester", type=str, default="ETS364B", help="Tester family name")
    parser.add_argument("--generic", type=str, default="DDR4SDRAM", help="Generic product name")
    parser.add_argument("--part", type=str, default="MT40A1G8", help="Part name")
    args = parser.parse_args()

    if args.waferpulse:
        synt_afm()
        return

    # Override global variables
    LIMITS_FILE = args.limits
    NUM_DEVICES = args.devices
    BASE_TP_DIR = os.path.join('dataset', args.tester, f"{args.generic}_{args.part}", 'T&P_Decrypted')
    BASE_FT_DIR = os.path.join('dataset', args.tester, f"{args.generic}_{args.part}", 'FT_Decrypted')

    # Load and process the limits file dynamically
    limits_df = pd.read_csv(LIMITS_FILE)
    limits_df['STDF Min'] = pd.to_numeric(limits_df['STDF Min'], errors='coerce')
    limits_df['STDF Max'] = pd.to_numeric(limits_df['STDF Max'], errors='coerce')
    limits_df['STDF Min'] = limits_df['STDF Min'].fillna(0)
    limits_df['STDF Max'] = limits_df['STDF Max'].fillna(limits_df['STDF Min'] + 10)

    # To generate MORE lots, simply add more entries to this list!
    types_config = []
    lot_count = args.lots 
    for i in range(lot_count):
        # 70% Good Yield, 20% Marginal, 10% Failing
        t_type = int(np.random.choice([1, 2, 3], p=[0.70, 0.20, 0.10]))
        types_config.append((t_type, f"SYN_{i:04d}_{t_type}"))
    
    import shutil
    # Clean previous directories
    if os.path.exists(BASE_TP_DIR):
        shutil.rmtree(BASE_TP_DIR)
    if os.path.exists(BASE_FT_DIR):
        shutil.rmtree(BASE_FT_DIR)
        
    # Copy limits file to the generic's root and subfolders so the predictor can find it
    os.makedirs(BASE_TP_DIR, exist_ok=True)
    os.makedirs(BASE_FT_DIR, exist_ok=True)
    
    # Copy master limits file to the generic's root as a secondary fallback
    if os.path.exists(LIMITS_FILE):
        generic_root = os.path.dirname(BASE_TP_DIR)
        shutil.copy(LIMITS_FILE, os.path.join(generic_root, LIMITS_FILE))
                
    for type_num, lot_id in types_config:
        # Select a random year for this lot to test multi-year filtering (2022-2026)
        lot_year = random.randint(2022, 2026)
        dfs_probe = []
        dfs_final = []
        yields_probe = []
        yields_final = []
        
        for set_num in [1]:
            # Generate Probe Test natively, pass an int hash of lot_id for deterministic lot-level macro shifting 
            probe_df = generate_base_data(type_num, set_num, lot_id_hash=abs(hash(lot_id)) % 10000)
            probe_yield = (probe_df['Bin'] == 1).mean() * 100.0
            yields_probe.append(probe_yield)
            
            # Generate Final Test purely passing through proper shifting correlated to Probe
            final_df = generate_final_from_probe(probe_df)
            final_yield = (final_df['Bin'] == 1).mean() * 100.0
            yields_final.append(final_yield)
            
            dfs_probe.append((set_num, probe_df))
            dfs_final.append((set_num, final_df))
            
        avg_probe_yield = np.mean(yields_probe)
        avg_final_yield = np.mean(yields_final)
        
        wafer_str = "01"
        
        # Build Probe Folder
        probe_folder = f"{lot_id}_[{wafer_str}]_{avg_final_yield:.2f}%"
        probe_folder_path = os.path.join(BASE_TP_DIR, probe_folder)
        os.makedirs(probe_folder_path, exist_ok=True)
        
        # Build Final Folder
        final_folder = f"{lot_id}_[{wafer_str}]_{avg_final_yield:.2f}%"
        final_folder_path = os.path.join(BASE_FT_DIR, final_folder)
        os.makedirs(final_folder_path, exist_ok=True)
        
        # Create 'limit' subfolders per lot as requested
        tp_limit_dir = os.path.join(probe_folder_path, "limit")
        ft_limit_dir = os.path.join(final_folder_path, "limit")
        os.makedirs(tp_limit_dir, exist_ok=True)
        os.makedirs(ft_limit_dir, exist_ok=True)
        
        for set_num, p_df in dfs_probe:
            now_str_file = datetime.now().strftime(f"%m%d{lot_year}_%H%M%S")
            probe_filename = f"{args.part}_00_{args.part}_{lot_id}_{set_num:02d}_{args.tester}_PRB_{now_str_file}.std_1.csv"
            write_datalog(p_df, probe_folder_path, probe_filename, year_override=lot_year)
            
            # Create matching limits file: {csv_name}_limits.csv
            if os.path.exists(LIMITS_FILE):
                shutil.copy(LIMITS_FILE, os.path.join(tp_limit_dir, f"{probe_filename.replace('.csv', '')}_limits.csv"))
            time.sleep(0.1) 
            
        for set_num, f_df in dfs_final:
            now_str_file = datetime.now().strftime(f"%m%d{lot_year}_%H%M%S")
            final_filename = f"{args.part}_00_{args.part}_{lot_id}_{set_num:02d}_{args.tester}_FIN_{now_str_file}.std_1.csv"
            write_datalog(f_df, final_folder_path, final_filename, year_override=lot_year)
            
            # Create matching limits file: {csv_name}_limits.csv
            if os.path.exists(LIMITS_FILE):
                shutil.copy(LIMITS_FILE, os.path.join(ft_limit_dir, f"{final_filename.replace('.csv', '')}_limits.csv"))
            time.sleep(0.1)

if __name__ == "__main__":
    main()
