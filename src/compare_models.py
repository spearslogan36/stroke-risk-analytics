"""
Pulls the three supervised models' saved metrics together into one
comparison table.

Run with: python src/compare_models.py
Requires: train_logistic.py, train_naive_bayes.py, and train_decision_tree.py
          to have already been run at least once, so their outputs/*.json
          files exist.
Produces: outputs/model_comparison.csv
"""
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_SOURCES = {
    "Logistic Regression": ("logistic_regression_metrics.json", "tuned_balanced_default_threshold"),
    "Decision Tree": ("decision_tree_metrics.json", "balanced"),
    "Naive Bayes": ("naive_bayes_metrics.json", "tuned_prior"),
}

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def main():
    rows = []
    for model_name, (filename, key) in MODEL_SOURCES.items():
        with open(OUTPUT_DIR / filename) as f:
            data = json.load(f)
        metrics = data[key]
        row = {"model": model_name}
        row.update({m: metrics[m] for m in METRIC_COLUMNS})
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("model")
    comparison = comparison.sort_values("roc_auc", ascending=False)

    pd.set_option("display.width", 120)
    print(comparison.round(3))

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv")
    print(f"\nSaved to {OUTPUT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()