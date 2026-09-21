
# Gaussian Naive Bayes classifier for stroke prediction.

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_access import get_full_dataset
from preprocessing import build_preprocessing_pipeline, filter_unmodelable_records

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
NUMERIC_COLS = ["age", "avg_glucose_level", "bmi"]


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw = get_full_dataset()
    filtered = filter_unmodelable_records(raw)

    print("Correlation matrix -- Naive Bayes assumes these are all ~0:")
    print(filtered[NUMERIC_COLS].corr().round(3))
    print()

    X = filtered.drop(columns=["stroke"])
    y = filtered["stroke"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Default: natural, empirical class prior
    default_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline(scale_numeric=False)),
        ("classifier", GaussianNB()),
    ])
    default_pipeline.fit(X_train, y_train)
    default_pred = default_pipeline.predict(X_test)
    default_proba = default_pipeline.predict_proba(X_test)[:, 1]

    # Tuned: let cross-validation search artificially adjusted priors too,
    # to see whether the natural prior is actually the best choice.
    tuned_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline(scale_numeric=False)),
        ("classifier", GaussianNB()),
    ])
    param_grid = {
        "classifier__priors": [None, [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]],
    }
    grid = GridSearchCV(tuned_pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned_pipeline = grid.best_estimator_
    tuned_pred = tuned_pipeline.predict(X_test)
    tuned_proba = tuned_pipeline.predict_proba(X_test)[:, 1]

    results = {
        "default_prior": evaluate(y_test, default_pred, default_proba),
        "tuned_prior": {
            "best_params": grid.best_params_,
            "cv_roc_auc": grid.best_score_,
            **evaluate(y_test, tuned_pred, tuned_proba),
        },
    }

    with open(OUTPUT_DIR / "naive_bayes_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    # Interpretability payoff: the per-class mean each numeric feature learned
    nb = tuned_pipeline.named_steps["classifier"]
    feature_names = tuned_pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [f.split("__", 1)[1] for f in feature_names]

    print("\nLearned per-class means (numeric features):")
    for i, name in enumerate(NUMERIC_COLS):
        idx = feature_names.index(name)
        print(f"  {name}: No Stroke = {nb.theta_[0][idx]:.2f}, Stroke = {nb.theta_[1][idx]:.2f}")


if __name__ == "__main__":
    main()