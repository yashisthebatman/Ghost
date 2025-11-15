import os, sys, pandas as pd, json
from sqlalchemy.orm import Session
from pathlib import Path

sys.path.insert(0, '/app')
from app.database import SessionLocal
from app.models import Sessions, LapSummaries

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strips leading/trailing whitespace from column names."""
    df.columns = df.columns.str.strip()
    return df

def time_str_to_seconds(time_str):
    """
    Converts a time string in 'M:S.ms' or 'S.ms' format to total seconds.
    Returns None if conversion is not possible.
    """
    if pd.isna(time_str) or not isinstance(time_str, str):
        return None
    try:
        if ':' in time_str:
            minutes, seconds = time_str.split(':')
            return float(minutes) * 60 + float(seconds)
        else:
            return float(time_str)
    except (ValueError, TypeError):
        # Handles cases where the data is malformed
        return None

def main():
    print("--- Starting Contextual Data Processing ---")
    db = SessionLocal()
    
    telemetry_file = os.getenv("TELEMETRY_FILE_PATH")
    if not telemetry_file:
        sys.exit("Error: TELEMETRY_FILE_PATH environment variable not set.")
    
    session_name = Path(telemetry_file).stem
    target_vehicle_id = os.getenv("TARGET_VEHICLE_ID")

    session = db.query(Sessions).filter(Sessions.session_name == session_name).first()
    if not session:
        sys.exit(f"Error: No session found for '{session_name}'. Please run process_telemetry.py first.")
        
    print(f"Found session ID {session.id} for vehicle {target_vehicle_id}. Enriching data...")
    
    data_root = Path(telemetry_file).parent
    lap_analysis_file = data_root / "23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV"
    weather_file = data_root / "26_Weather_Race 2_Anonymized.CSV"
    
    try:
        # 1. Process Weather Data
        if weather_file.exists():
            print(f"Processing weather file: {weather_file.name}")
            df_weather = pd.read_csv(weather_file, sep=';')
            weather_data = df_weather.iloc[0].to_dict()
            session.weather_data = weather_data
            print("   Updated session with weather data.")
        else:
            print(f"Warning: Weather file not found at {weather_file}")

        # 2. Process Lap Summary Data
        if lap_analysis_file.exists():
            print(f"Processing lap analysis file: {lap_analysis_file.name}")
            df_laps_raw = pd.read_csv(lap_analysis_file, sep=';')
            df_laps_raw = clean_column_names(df_laps_raw)
            
            car_number_str = target_vehicle_id.split('-')[1]
            car_number_int = int(car_number_str)
            
            df_vehicle_laps = df_laps_raw[df_laps_raw['NUMBER'] == car_number_int].copy()

            if not df_vehicle_laps.empty:
                # --- THIS IS THE CRITICAL FIX ---
                print("   Converting LAP_TIME from string to seconds...")
                df_vehicle_laps['LAP_TIME_SECONDS'] = df_vehicle_laps['LAP_TIME'].apply(time_str_to_seconds)
                
                db.query(LapSummaries).filter(LapSummaries.session_id == session.id).delete()

                for _, row in df_vehicle_laps.iterrows():
                    lap_summary = LapSummaries(
                        session_id=session.id,
                        lap_number=row.get('LAP_NUMBER'),
                        lap_time=row.get('LAP_TIME_SECONDS'), # Use the converted value
                        s1_time=row.get('S1_SECONDS'),
                        s2_time=row.get('S2_SECONDS'),
                        s3_time=row.get('S3_SECONDS'),
                        top_speed_kph=row.get('TOP_SPEED')
                    )
                    db.add(lap_summary)
                print(f"   Added {len(df_vehicle_laps)} lap summaries to the database.")
            else:
                print(f"   Warning: No lap data found for car number {car_number_int} in analysis file.")
        else:
            print(f"Warning: Lap analysis file not found at {lap_analysis_file}")

        db.commit()
        print("\n--- Contextual data processing complete! ---")

    except Exception as e:
        import traceback; traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()