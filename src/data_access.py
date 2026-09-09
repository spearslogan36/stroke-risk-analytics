from pathlib import Path
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "processed" / "stroke.db"

FULL_DATASET_QUERY = """
SELECT
    p.patient_id,
    p.gender,
    p.age,
    p.ever_married,
    p.work_type,
    p.residence_type,
    c.hypertension,
    c.heart_disease,
    c.avg_glucose_level,
    c.bmi,
    c.smoking_status,
    c.stroke
FROM patients p
JOIN clinical_records c ON p.patient_id = c.patient_id;
"""


def get_full_dataset(db_path: Path = DB_PATH) -> pd.DataFrame:
    # Runs the patients/clinical_records JOIN and returns one row per patient
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(FULL_DATASET_QUERY, conn)
    finally:
        conn.close()
    return df


if __name__ == "__main__":
    result = get_full_dataset()
    print(result.shape)
    print(result.head())