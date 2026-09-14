# Exports a single, denormalized CSV from the SQLite database, formatted for connecting to Tableau Public.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_access import get_full_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stroke_for_tableau.csv"


def main():
    df = get_full_dataset()

    df["stroke_label"] = df["stroke"].map({0: "No Stroke", 1: "Stroke"})
    df["hypertension_label"] = df["hypertension"].map({0: "No", 1: "Yes"})
    df["heart_disease_label"] = df["heart_disease"].map({0: "No", 1: "Yes"})

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()