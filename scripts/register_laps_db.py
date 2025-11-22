# Usage: docker-compose exec backend python /scripts/register_laps_db.py
import sys
sys.path.insert(0, '/app')
from app.database import SessionLocal
from app.models import Sessions, LapSummaries

def register():
    print("--- REGISTERING GENERATED LAPS IN DB ---")
    db = SessionLocal()
    session = db.query(Sessions).first()
    
    # Clear existing
    db.query(LapSummaries).filter(LapSummaries.session_id == session.id).delete()
    
    # Insert the 3 laps matching your parquet files
    laps = [
        {"n": 1, "t": 70.54, "s1": 38.2, "s2": 45.1, "speed": 210},
        {"n": 2, "t": 65.12, "s1": 36.5, "s2": 43.8, "speed": 235},
        {"n": 3, "t": 60.00, "s1": 35.9, "s2": 43.2, "speed": 260}, # PB
    ]
    
    for l in laps:
        db.add(LapSummaries(
            session_id=session.id,
            lap_number=l['n'],
            lap_time=l['t'],
            s1_time=l['s1'],
            s2_time=l['s2'],
            s3_time=l['t'] - l['s1'] - l['s2'],
            top_speed_kph=l['speed']
        ))
    
    db.commit()
    print(f"✅ Registered {len(laps)} laps. Sidebar will now show 3 items.")
    db.close()

if __name__ == "__main__":
    register()