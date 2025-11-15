# This script is the first step. It processes the large telemetry file.
# It should be run BEFORE process_context.py
import os, sys, pandas as pd, numpy as np, joblib
from sqlalchemy.orm import Session
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, '/app')
from app.database import SessionLocal
from app.models import Sessions, MicroSectors

# Configuration is identical to the last working version
CONFIG = {
    "env": {
        "target_vehicle_id": os.getenv("TARGET_VEHICLE_ID"),
        "telemetry_file_path": os.getenv("TELEMETRY_FILE_PATH"),
    },
    "paths": {
        "processed": Path("/data/processed"),
        "snippets": Path("/data/processed/snippets"),
        "scaler": Path("/data/processed/scaler.joblib"),
    },
    "required_channels": ["speed", "ath", "pbrake_f", "Steering_Angle"],
    "feature_engineering": {"roc_channels": ["Steering_Angle", "ath", "pbrake_f"]},
    "model_channels": [
        "speed", "ath", "pbrake_f", "Steering_Angle",
        "Steering_Angle_roc", "ath_roc", "pbrake_f_roc"
    ],
    "outlier_clipping": {"speed": (0, 350), "ath": (0, 100), "pbrake_f": (0, 120), "Steering_Angle": (-540, 540)},
    "segmentation_states": {"braking_threshold": 10, "turn_steer_threshold": 20, "accel_throttle_threshold": 15, "straight_steer_threshold": 5}
}

# All helper functions (load_and_prepare_data, clip_outliers, etc.) are the same...
# For brevity, we will just show the main function's logic flow.
# [Note: The full code for helper functions is the same as the previous response's script]

def load_and_prepare_data(file_path: str, vehicle_id: str) -> pd.DataFrame:
    print(f"1. Loading data for vehicle {vehicle_id}...")
    df = pd.read_csv(file_path)
    df_vehicle = df[df["vehicle_id"] == vehicle_id].copy()
    if df_vehicle.empty: raise ValueError(f"No data for vehicle_id: {vehicle_id}")
    print("   Pivoting data to wide format...")
    df_wide = df_vehicle.pivot_table(index="timestamp", columns="telemetry_name", values="telemetry_value").reset_index()
    df_wide["timestamp"] = pd.to_datetime(df_wide["timestamp"])
    df_wide = df_wide.sort_values("timestamp").ffill().bfill()
    df_wide["time_delta"] = df_wide["timestamp"].diff().dt.total_seconds().fillna(0)
    df_wide["cumulative_time"] = df_wide["time_delta"].cumsum()
    for channel in CONFIG["required_channels"]:
        if channel not in df_wide.columns: raise ValueError(f"Required channel '{channel}' not found.")
    return df_wide

def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    print("2. Clipping outliers...")
    for channel, (min_val, max_val) in CONFIG["outlier_clipping"].items():
        if channel in df.columns: df[channel] = df[channel].clip(min_val, max_val)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("3. Engineering Rate-of-Change features...")
    df_eng = df.copy()
    time_diff = df_eng["cumulative_time"].diff().replace(0, 1e-6).fillna(1e-6)
    for channel in CONFIG["feature_engineering"]["roc_channels"]:
        roc_col_name = f"{channel}_roc"
        df_eng[roc_col_name] = df_eng[channel].diff().fillna(0) / time_diff
        df_eng[roc_col_name] = df_eng[roc_col_name].rolling(window=5, min_periods=1, center=True).mean()
    return df_eng.ffill().bfill()

def get_driving_state(row: pd.Series) -> str:
    s = CONFIG["segmentation_states"]
    if row["pbrake_f"] > s["braking_threshold"]: return "Braking"
    if abs(row["Steering_Angle"]) < s["straight_steer_threshold"]: return "Straight"
    if row["pbrake_f"] < s["braking_threshold"] and row["ath"] < s["accel_throttle_threshold"] and abs(row["Steering_Angle"]) > s["turn_steer_threshold"]: return "Turn"
    if row["ath"] > s["accel_throttle_threshold"] and abs(row["Steering_Angle"]) > s["turn_steer_threshold"]: return "Acceleration"
    return "Transition"

def segment_and_save(db: Session, session_id: int, df_engineered: pd.DataFrame, df_clipped: pd.DataFrame):
    print("5. Segmenting by driving state and saving snippets...")
    df_clipped["state"] = df_clipped.apply(get_driving_state, axis=1)
    snippets, current_state, start_index = [], None, 0
    for i, state in tqdm(enumerate(df_clipped["state"]), total=len(df_clipped), desc="   Scanning states"):
        if state != "Transition" and state != current_state:
            if current_state is not None and (i - 1 > start_index):
                snippets.append({"state": current_state, "start_idx": start_index, "end_idx": i - 1})
            current_state, start_index = state, i
    if current_state is not None: snippets.append({"state": current_state, "start_idx": start_index, "end_idx": len(df_clipped) - 1})
    print(f"   Found {len(snippets)} snippets. Analyzing and saving...")
    scaler = joblib.load(CONFIG["paths"]["scaler"])
    snippet_count = 0
    for snip_info in tqdm(snippets, desc="   Saving snippets"):
        snippet_df_raw = df_clipped.iloc[snip_info['start_idx']:snip_info['end_idx'] + 1]
        if len(snippet_df_raw) < 10: continue
        time_delta = snippet_df_raw['cumulative_time'].iloc[-1] - snippet_df_raw['cumulative_time'].iloc[0]
        entry_speed, exit_speed, min_speed = snippet_df_raw['speed'].iloc[0], snippet_df_raw['speed'].iloc[-1], snippet_df_raw['speed'].min()
        snippet_df_eng = df_engineered.iloc[snip_info['start_idx']:snip_info['end_idx'] + 1]
        normalized_data = snippet_df_eng[CONFIG["model_channels"]].values
        file_name = f"session_{session_id}_snippet_{snippet_count}.npy"
        np.save(CONFIG["paths"]["snippets"] / file_name, normalized_data)
        db.add(MicroSectors(
            session_id=session_id, sector_type=snip_info["state"], lap_number=0, time_delta=time_delta,
            entry_speed=entry_speed, exit_speed=exit_speed, min_speed=min_speed, snippet_path=str(CONFIG["paths"]["snippets"] / file_name)
        ))
        snippet_count += 1
    db.commit()
    print(f"   Saved {snippet_count} valid snippets.")

def main():
    cfg = CONFIG["env"]
    paths = CONFIG["paths"]
    paths["snippets"].mkdir(parents=True, exist_ok=True)
    db = SessionLocal()
    try:
        df_raw = load_and_prepare_data(cfg["telemetry_file_path"], cfg["target_vehicle_id"])
        df_clipped = clip_outliers(df_raw.copy())
        df_engineered = engineer_features(df_clipped.copy())

        print("4. Normalizing data and saving scaler...")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(df_engineered[CONFIG["model_channels"]])
        joblib.dump(scaler, paths["scaler"])

        session_name = Path(cfg["telemetry_file_path"]).stem
        # Check if session already exists
        session = db.query(Sessions).filter(Sessions.session_name == session_name).first()
        if not session:
            session = Sessions(session_name=session_name, vehicle_id=cfg["target_vehicle_id"])
            db.add(session)
            db.commit()
            db.refresh(session)
            print(f"   Created new session record with ID: {session.id}")
        else:
            print(f"   Found existing session record with ID: {session.id}")
            # Optional: Clean up old snippets for this session before re-processing
            db.query(MicroSectors).filter(MicroSectors.session_id == session.id).delete()
            db.commit()
            print("   Deleted old microsector data for this session.")

        segment_and_save(db, session.id, df_engineered, df_clipped)
        print("\n--- Telemetry processing complete! ---")
        print("--- Next step: run process_context.py ---")

    except Exception as e:
        import traceback; traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()