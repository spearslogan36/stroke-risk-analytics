"""

Builds a normalized SQLite database from the raw .csv file.

Run with: python src/build_database.py
Produces: data/processed/stroke.db

"""

from pathlib import Path

import pandas as pd
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
DB_PATH = PROJECT_ROOT / "data" / "processed" / "stroke.db"
SCHEMA_SQL = PROJECT_ROOT / "sql" / "01_create_schema.sql"


def build_database(raw_csv: Path = RAW_CSV, db_path: Path = DB_PATH, schema_sql: Path = SCHEMA_SQL) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv)

    patients = df[["id", "gender", "age", "ever_married", "work_type", "Residence_type"]].copy()
    patients.columns = ["patient_id", "gender", "age", "ever_married", "work_type", "residence_type"]

    clinical = df[["id", "hypertension", "heart_disease", "avg_glucose_level", "bmi",
                    "smoking_status", "stroke"]].copy()
    clinical.columns = ["patient_id", "hypertension", "heart_disease", "avg_glucose_level",
                         "bmi", "smoking_status", "stroke"]

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema_sql.read_text())
        patients.to_sql("patients", conn, if_exists="append", index=False)
        clinical.to_sql("clinical_records", conn, if_exists="append", index=False)
        conn.commit()
    finally:
        conn.close()

    print(f"Built {db_path} with {len(patients)} patients and {len(clinical)} clinical records.")


if __name__ == "__main__":
    build_database()