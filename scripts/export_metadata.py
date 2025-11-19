import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, '/app')
from app.database import SessionLocal
from app.models import MicroSectors

OUTPUT_PATH = Path("/data/processed/snippets_metadata.csv")

def export_metadata_to_csv():
    print("--- Exporting MicroSectors metadata to CSV for Colab ---")
    db = SessionLocal()
    try:
        query = db.query(
            MicroSectors.snippet_path,
            MicroSectors.sector_type,
            MicroSectors.time_delta,
            MicroSectors.entry_speed
        )
        
        df = pd.read_sql(query.statement, db.bind)
        
        # Make the snippet_path relative for easier use in Colab
        df['snippet_path'] = df['snippet_path'].apply(lambda x: Path(x).name)
        
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_PATH, index=False)
        
        print(f"Successfully exported {len(df)} records to: {OUTPUT_PATH}")

    finally:
        db.close()

if __name__ == "__main__":
    export_metadata_to_csv()