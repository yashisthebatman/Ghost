import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '/app')
from app.database import SessionLocal, engine
from app.models import Base, Sessions, MicroSectors

METADATA_PATH = Path("/data/processed/snippets_metadata.csv")

def populate():
    print("--- 1. POPULATING DB (SMART LAP DETECTION) ---")
    
    if not METADATA_PATH.exists():
        print(f"❌ Error: {METADATA_PATH} not found.")
        return

    df = pd.read_csv(METADATA_PATH)
    print(f"   Loaded {len(df)} sectors from stream.")

    # --- SMART LAP SEGMENTATION ---
    # 1. Find the "Signature" Sector (The Main Straight)
    # We look for 'Straight' or 'Acceleration' sectors.
    # The Main Straight at Road America is fast and long.
    
    straights = df[
        (df['sector_type'].isin(['Straight', 'Acceleration'])) & 
        (df['time_delta'] > 4.0) # Filter out short straights
    ].copy()

    if straights.empty:
        print("❌ Could not detect any long straights to split laps. Using fallback.")
        target_df = df.iloc[:50] # Fallback
    else:
        # We assume the sector with the Highest Entry Speed + Duration combination is the Main Straight
        # We calculate a 'score' to find the most distinct straight
        straights['score'] = straights['time_delta'] * straights['entry_speed']
        
        # Pick the specific sector that represents the Main Straight
        # We take the top 10% of fast straights to find the recurring pattern
        threshold = straights['score'].quantile(0.95)
        main_straights = straights[straights['score'] >= threshold]
        
        # Get the INDICES of these straights in the main dataframe
        # These are our "Lap Markers"
        lap_markers = main_straights.index.sort_values().tolist()
        
        print(f"   Detected {len(lap_markers)} potential lap markers (Main Straight crossings).")

        laps = []
        # Slice the dataframe between markers
        for i in range(len(lap_markers) - 1):
            start_idx = lap_markers[i]
            end_idx = lap_markers[i+1]
            
            # A valid lap must have enough sectors (e.g., >30) and not too many (>100)
            if 30 < (end_idx - start_idx) < 100:
                lap_df = df.iloc[start_idx:end_idx].copy()
                lap_time = lap_df['time_delta'].sum()
                
                # Road America Check: Lap must be between 120s and 160s
                if 120 < lap_time < 160:
                    laps.append({
                        'df': lap_df,
                        'time': lap_time,
                        'idx': i
                    })

        if not laps:
            print("   ⚠️ Smart segmentation found markers but no valid laps (check time thresholds). Fallback to first 140s.")
            target_df = df[df['time_delta'].cumsum() < 140]
        else:
            # Pick the FASTEST valid lap
            best_lap = min(laps, key=lambda x: x['time'])
            target_df = best_lap['df']
            print(f"   ✅ Identified {len(laps)} valid laps.")
            print(f"   🏆 Selected Fastest Lap: {best_lap['time']:.2f}s (Lap Index {best_lap['idx']})")
            print(f"      Sector Count: {len(target_df)}")

    # --- DB INGESTION ---
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        session = Sessions(
            session_name="Road_America_Optimal",
            vehicle_id="GR86-AI-Ghost",
            weather_data={"track_temp": 28, "condition": "OPTIMAL"}
        )
        db.add(session)
        db.commit()

        sectors = []
        # Re-index lap_number to 1 for the AI
        for _, row in target_df.iterrows():
            filename = Path(row['snippet_path']).name
            full_path = f"/data/processed/snippets/{filename}"
            
            s = MicroSectors(
                session_id=session.id,
                sector_type=row.get('sector_type', 'Straight'),
                lap_number=1, 
                time_delta=float(row.get('time_delta', 1.0)),
                entry_speed=float(row.get('entry_speed', 100.0)),
                exit_speed=0.0, 
                min_speed=0.0,
                snippet_path=full_path
            )
            sectors.append(s)
        
        db.add_all(sectors)
        db.commit()
        print("✅ DB Populated. Ready for synthesis.")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    populate()