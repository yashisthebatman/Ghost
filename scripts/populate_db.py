# ==========================================
# FILE: scripts/populate_db.py
# ==========================================
import pandas as pd
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/app')
from app.database import SessionLocal, engine
from app.models import Base, Sessions, MicroSectors

# Paths
METADATA_PATH = Path("/data/processed/snippets_metadata.csv")
SNIPPETS_DIR = Path("/data/processed/snippets")

def populate():
    print("--- POPULATING DATABASE FROM CSV ---")
    
    if not METADATA_PATH.exists():
        print(f"❌ Metadata CSV not found at {METADATA_PATH}")
        return

    # 1. Load CSV
    df = pd.read_csv(METADATA_PATH)
    print(f"   Loaded {len(df)} rows from CSV.")

    # 2. Reset & Init DB
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    try:
        # 3. Create Session
        session = Sessions(
            session_name="Restored_Session",
            vehicle_id="GR86-002-2",
            processed_at=datetime.utcnow(),
            weather_data={"track_temp": 32, "condition": "OPTIMAL"}
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        print(f"✅ Created Session ID: {session.id}")

        # 4. Insert MicroSectors
        # We assume the CSV has columns: snippet_path, sector_type, time_delta, entry_speed
        # If your CSV has different names, we map them here.
        
        count = 0
        for idx, row in df.iterrows():
            # Fix path: Ensure it points to the docker path /data/processed/snippets/...
            filename = Path(row['snippet_path']).name
            full_path = str(SNIPPETS_DIR / filename)
            
            # Default to 'Straight' if type is missing
            s_type = row.get('sector_type', 'Straight')
            
            # Default lap 1 if missing
            lap_num = row.get('lap_number', 1) 

            sector = MicroSectors(
                session_id=session.id,
                sector_type=s_type,
                lap_number=lap_num,
                time_delta=row.get('time_delta', 1.0),
                entry_speed=row.get('entry_speed', 100.0),
                exit_speed=0.0, # Optional
                min_speed=0.0,  # Optional
                snippet_path=full_path
            )
            db.add(sector)
            count += 1

        db.commit()
        print(f"✅ Successfully inserted {count} sectors into Database.")
        print("   You can now run synthesize_lap.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    populate()