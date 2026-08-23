import os
import glob
import pandas as pd
import random

def inject_resistance_sensors():
    print("WaferPulse: Starting Sensor Injection...")
    
    # 1. Point to the folder where your friend's code saves the chips
    target_folder = os.path.join('dataset', 'J750', 'DDR4SDRAM_MT40A1G8', 'T&P_Decrypted')
    
    if not os.path.exists(target_folder):
        print(f"❌ ERROR: Cannot find {target_folder}. Did you run your friend's generator first?")
        return

    # 2. Find all the CSV files his factory just generated
    csv_files = glob.glob(os.path.join(target_folder, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} factory logs. Scanning dies...")

    # 3. Loop through every single file and inject the WaferPulse sensors
    for file in csv_files:
        df = pd.read_csv(file)
        
        # Safety Check: Skip if we already injected sensors into this file
        if "Resistance_mean" in df.columns:
            continue

        num_rows = len(df)
        res_mean, res_median, res_std = [], [], []

        for _ in range(num_rows):
            # 15% of dies have bad probe marks
            if random.random() < 0.15:
                res_mean.append(round(random.uniform(0.8, 3.5), 3))
                res_median.append(round(random.uniform(0.5, 2.0), 3))
                res_std.append(round(random.uniform(1.5, 5.0), 3)) 
            # 85% of dies are perfectly healthy
            else:
                res_mean.append(round(random.uniform(0.1, 0.4), 3))
                res_median.append(round(random.uniform(0.1, 0.3), 3))
                res_std.append(round(random.uniform(0.05, 0.4), 3)) 

        # Add the WaferPulse columns to his dataframe
        df["Resistance_mean"] = res_mean
        df["Resistance_median"] = res_median
        df["Resistance_std"] = res_std

        # Overwrite the CSV file with the new data
        df.to_csv(file, index=False)

    print("✅ WaferPulse Sensor Injection Complete!")

if __name__ == "__main__":
    inject_resistance_sensors()