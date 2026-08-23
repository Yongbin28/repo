import pandas as pd
import numpy as np

def compute_spatial_risk(df: pd.DataFrame) -> dict:
    """
    Analyzes the wafer spatial distribution for cluster defects and GDBN density.
    Requires 'X' and 'Y' columns, and assumes devices with Bin != 1 or
    having Fails are failing dies.
    """
    # Normalize col names
    n_map = {c.strip().lower(): c for c in df.columns}
    
    if 'x' not in n_map or 'y' not in n_map:
        return {
            "gdbn_count": 0,
            "gdbn_ratio_all_dies": 0.0,
            "gdbn_rate_good_dies": 0.0,
            "total_dies": 0,
            "total_good_dies": 0,
            "spatial_defect_density": 0.0,
            "edge_cluster_count": 0,
            "wafer_map_df": pd.DataFrame(),
        }

    x_col = n_map['x']
    y_col = n_map['y']
    
    # Carry forward metadata columns for plotting (hover data)
    keep_cols = [x_col, y_col]
    for extra in ['Device #', 'device #', 'Site', 'site', 'Bin', 'bin']:
        if extra in df.columns and extra not in keep_cols:
            keep_cols.append(extra)
    df_spatial = df[keep_cols].copy()
    
    # Determine pass/fail
    # Default to failing unless proven passing
    is_pass = pd.Series(True, index=df.index)
    
    if 'bin' in n_map:
        is_pass = is_pass & (pd.to_numeric(df[n_map['bin']], errors='coerce') == 1)
        
    for k in ['fails', 'alarms']:
        if k in n_map:
            col_val = df[n_map[k]].astype(str).str.strip()
            # If col_val is not empty and not NaN, it is a fail/alarm
            is_pass = is_pass & (df[n_map[k]].isna() | (col_val == "") | (col_val == "nan"))
            
    df_spatial['is_pass'] = is_pass
    
    # Drop rows without valid coordinates
    df_spatial[x_col] = pd.to_numeric(df_spatial[x_col], errors='coerce')
    df_spatial[y_col] = pd.to_numeric(df_spatial[y_col], errors='coerce')
    df_spatial = df_spatial.dropna(subset=[x_col, y_col])
    
    if df_spatial.empty:
         return {
             "gdbn_count": 0,
             "gdbn_ratio_all_dies": 0.0,
             "gdbn_rate_good_dies": 0.0,
             "total_dies": 0,
             "total_good_dies": 0,
             "spatial_defect_density": 0.0,
             "edge_cluster_count": 0,
             "wafer_map_df": df_spatial,
         }
    
    # Wafer dimensions
    x_min, x_max = df_spatial[x_col].min(), df_spatial[x_col].max()
    y_min, y_max = df_spatial[y_col].min(), df_spatial[y_col].max()
    
    # Create coordinate matrix mapping
    df_spatial['coord'] = list(zip(df_spatial[x_col], df_spatial[y_col]))
    pass_map = dict(zip(df_spatial['coord'], df_spatial['is_pass']))
    
    gdbn_count = 0
    total_good = 0
    df_spatial['gdbn_flag'] = False
    
    # Identify GDBN (Good Die fully surrounded by Bad Neighbors)
    # Define "fully surrounded" as having at least N bad neighbors and very few/no good neighbors
    # For a classic 8-neighborhood (Moore)
    for idx, row in df_spatial.iterrows():
        if row['is_pass']:
            total_good += 1
            x, y = row[x_col], row[y_col]
            neighbors = [
                (x-1, y-1), (x, y-1), (x+1, y-1),
                (x-1, y),             (x+1, y),
                (x-1, y+1), (x, y+1), (x+1, y+1)
            ]
            
            bad_neighbors = 0
            valid_neighbors = 0
            
            for nx, ny in neighbors:
                if (nx, ny) in pass_map:
                    valid_neighbors += 1
                    if not pass_map[(nx, ny)]:
                        bad_neighbors += 1
            
            # GDBN logic: BNR >= 0.75 with at least 4 valid neighbours.
            if valid_neighbors >= 4 and (bad_neighbors / valid_neighbors) >= 0.75:
                gdbn_count += 1
                df_spatial.at[idx, 'gdbn_flag'] = True

    # Spatial defect density
    # Percentage of dies that are failing overall
    total_dies = len(df_spatial)
    total_good_dies = int(df_spatial['is_pass'].sum())
    fail_dies = total_dies - df_spatial['is_pass'].sum()
    defect_density = (fail_dies / total_dies) if total_dies > 0 else 0.0
    gdbn_ratio_all_dies = (gdbn_count / total_dies) if total_dies > 0 else 0.0
    gdbn_rate_good_dies = (
        gdbn_count / total_good_dies
    ) if total_good_dies > 0 else 0.0
    
    # Edge cluster detection
    # Dies on the extreme bounds of the X/Y axes that failed
    edge_fails = df_spatial[
        ((df_spatial[x_col] <= x_min + 2) | (df_spatial[x_col] >= x_max - 2) |
         (df_spatial[y_col] <= y_min + 2) | (df_spatial[y_col] >= y_max - 2)) &
        (~df_spatial['is_pass'])
    ]
    edge_cluster_count = len(edge_fails)
    
    # Incorporating GDBN and Edge defects into an abstract spatial score mapping
    return {
        "gdbn_count": gdbn_count,
        "gdbn_ratio_all_dies": gdbn_ratio_all_dies,
        "gdbn_rate_good_dies": gdbn_rate_good_dies,
        "total_dies": total_dies,
        "total_good_dies": total_good_dies,
        "spatial_defect_density": defect_density,
        "edge_cluster_count": edge_cluster_count,
        "wafer_map_df": df_spatial
    }
