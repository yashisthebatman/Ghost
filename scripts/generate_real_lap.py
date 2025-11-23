import pandas as pd
import numpy as np
import sys
from pathlib import Path

# --- CONFIG ---
RAW_CSV_PATH = Path("/data/raw/road_america/race2/R2_road_america_telemetry_data.csv")
OUTPUT_DIR = Path("/data/processed/ghost_laps")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"--- 🔍 DATA TRANSFORMATION & EXTRACTION (FIXED) ---")
    
    if not RAW_CSV_PATH.exists():
        print(f"❌ Error: Raw file not found at {RAW_CSV_PATH}")
        return

    # 1. LOAD RAW DATA
    print("1. Loading CSV...")
    try:
        df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
        return

    # 2. FILTER VEHICLE
    target_id = "GR86-002-2"
    if 'vehicle_id' in df.columns:
        df = df[df['vehicle_id'] == target_id].copy()
        print(f"   Filtered for {target_id}: {len(df)} rows.")
    
    if df.empty:
        print("❌ No data found.")
        return

    # 3. PIVOT (Long -> Wide)
    print("2. Pivoting Data...")
    df['telemetry_name'] = df['telemetry_name'].astype(str).str.strip()
    
    try:
        df_wide = df.pivot_table(
            index='timestamp', 
            columns='telemetry_name', 
            values='telemetry_value',
            aggfunc='first'
        )
        # Recover 'lap' column (it gets lost in pivot if not included)
        # We group by timestamp and take the first lap value found
        lap_series = df.groupby('timestamp')['lap'].first()
        df_wide['lap'] = lap_series
        df_wide = df_wide.reset_index()
    except Exception as e:
        print(f"❌ Pivot Failed: {e}")
        return

    # 4. NORMALIZE COLUMNS
    col_map = {}
    for col in df_wide.columns:
        c_low = col.lower()
        if 'speed' in c_low: col_map[col] = 'speed'
        elif 'throttle' in c_low or 'ath' in c_low: col_map[col] = 'ath'
        elif 'brake' in c_low and 'pressure' in c_low: col_map[col] = 'pbrake_f'
        elif 'pbrake_f' in c_low: col_map[col] = 'pbrake_f'
        elif 'steering' in c_low: col_map[col] = 'Steering_Angle'
    
    df_wide.rename(columns=col_map, inplace=True)
    print(f"   Columns: {list(df_wide.columns)}")

    # 5. FIX TIME (CRITICAL STEP)
    print("3. Normalizing Time...")
    
    # Attempt 1: Convert to Datetime
    try:
        df_wide['ts_obj'] = pd.to_datetime(df_wide['timestamp'])
        # If successful, convert to seconds from start
        start_time = df_wide['ts_obj'].min()
        df_wide['seconds'] = (df_wide['ts_obj'] - start_time).dt.total_seconds()
    except:
        # Attempt 2: It might be string numbers ("100.1", "100.2")
        try:
            df_wide['seconds'] = pd.to_numeric(df_wide['timestamp'])
            df_wide['seconds'] = df_wide['seconds'] - df_wide['seconds'].min()
        except:
            print("❌ Could not convert timestamp to numeric seconds. Aborting.")
            return

    # 6. EXTRACT LAPS
    print("4. Extracting Laps...")
    laps = []
    
    if 'lap' in df_wide.columns:
        unique_laps = df_wide['lap'].dropna().unique()
        unique_laps.sort()
        
        for ln in unique_laps:
            # Slice the lap
            lap_df = df_wide[df_wide['lap'] == ln].copy()
            lap_df = lap_df.sort_values('seconds')
            
            if lap_df.empty: continue

            # Calculate Duration
            duration = lap_df['seconds'].iloc[-1] - lap_df['seconds'].iloc[0]
            
            print(f"   > Lap {int(ln)}: {duration:.2f}s ({len(lap_df)} rows) ", end="")

            # VALIDATION: Road America is ~135s - 160s
            if 130 < duration < 180:
                # 7. RESAMPLE TO 100Hz (0.01s)
                # Create clean time index: 0.00, 0.01, 0.02 ... Duration
                new_time = np.arange(0, duration, 0.01)
                
                # Relativize existing time
                lap_df['rel_time'] = lap_df['seconds'] - lap_df['seconds'].iloc[0]
                lap_df = lap_df.set_index('rel_time')
                
                # Select numeric columns only
                valid_cols = [c for c in ['speed', 'ath', 'pbrake_f', 'Steering_Angle'] if c in lap_df.columns]
                
                # Interpolate
                # We reindex to the new clean timeline and fill values
                resampled = lap_df[valid_cols].reindex(lap_df.index.union(new_time)).interpolate(method='index').reindex(new_time)
                resampled = resampled.reset_index().rename(columns={'index': 'time'})
                
                laps.append({"n": int(ln), "df": resampled, "time": duration})
                print("✅ Valid")
            else:
                print("❌ Ignored (Duration)")
    
    if not laps:
        print("❌ No valid laps found. Check duration logic.")
        return

    # 8. SAVE
    laps.sort(key=lambda x: x['time'])
    
    selection = []
    if len(laps) >= 1: selection.append({"file": "real_lap_3.parquet", "data": laps[0], "label": "PB"})
    if len(laps) >= 2: selection.append({"file": "real_lap_2.parquet", "data": laps[1], "label": "Median"})
    if len(laps) >= 3: selection.append({"file": "real_lap_1.parquet", "data": laps[-1], "label": "Slow"})
    
    # Generic Best
    selection.append({"file": "real_lap.parquet", "data": laps[0], "label": "Best"})

    print(f"\n   📊 SAVING FILES:")
    for item in selection:
        d = item['data']['df'].copy()
        
        # UNIT SANITY CHECK (Raw -> Metric)
        # Speed: if > 500, it's raw. Scale to ~265 kph max
        if 'speed' in d.columns and d['speed'].max() > 500:
             d['speed'] = d['speed'] / (d['speed'].max() / 265.0)
             
        # Throttle: if > 200, scale to 100
        if 'ath' in d.columns and d['ath'].max() > 200:
             d['ath'] = d['ath'] / (d['ath'].max() / 100.0)
             
        # Brake: if > 200, scale to 100
        if 'pbrake_f' in d.columns and d['pbrake_f'].max() > 200:
             d['pbrake_f'] = d['pbrake_f'] / (d['pbrake_f'].max() / 100.0)

        # Ensure 100Hz time column
        d['time'] = np.arange(len(d)) * 0.01

        d.to_parquet(OUTPUT_DIR / item['file'])
        print(f"   -> {item['file']} ({item['data']['time']:.2f}s)")

    print("\n✅ Done.")

if __name__ == "__main__":
    main()