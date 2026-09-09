# End-to-end predictive pipeline for the stroke dataset.

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_access import get_full_dataset
from preprocessing import build_preprocessing_pipeline, filter_unmodelable_records

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                      cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


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

    X = filtered.drop(columns=["stroke"])
    y = filtered["stroke"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Baseline: unweighted, default settings, default 0.5 threshold.
    baseline_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    baseline_pipeline.fit(X_train, y_train)
    baseline_proba = baseline_pipeline.predict_proba(X_test)[:, 1]
    baseline_pred = baseline_pipeline.predict(X_test)

    # Tuned, class-balanced pipeline
    tuned_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", LogisticRegression(random_state=RANDOM_STATE)),
    ])
    param_grid = [
        {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100], "classifier__solver": ["lbfgs"],
         "classifier__penalty": ["l2"], "classifier__class_weight": ["balanced"],
         "classifier__max_iter": [2000]},
        {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100], "classifier__solver": ["liblinear"],
         "classifier__penalty": ["l1"], "classifier__class_weight": ["balanced"],
         "classifier__max_iter": [2000]},
    ]
    grid = GridSearchCV(tuned_pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    # Leakage-safe threshold selection: out-of-fold probabilities on
    # TRAINING data only. The test set plays no role in this decision.
    oof_proba = cross_val_predict(
        best_pipeline, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    thresholds = np.linspace(0.05, 0.95, 19)
    f1_scores = [f1_score(y_train, (oof_proba >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1_scores))])

    # Final, one-time evaluation on the untouched test set.
    tuned_proba = best_pipeline.predict_proba(X_test)[:, 1]
    tuned_pred_default = (tuned_proba >= 0.5).astype(int)
    tuned_pred_selected_threshold = (tuned_proba >= best_threshold).astype(int)

    results = {
        "baseline": evaluate(y_test, baseline_pred, baseline_proba),
        "tuned_balanced_default_threshold": evaluate(y_test, tuned_pred_default, tuned_proba),
        "tuned_balanced_selected_threshold": {
            "threshold": best_threshold,
            "selected_via": "out-of-fold cross-validated probabilities on training data",
            **evaluate(y_test, tuned_pred_selected_threshold, tuned_proba),
        },
        "grid_search_best_params": grid.best_params_,
        "grid_search_best_cv_roc_auc": grid.best_score_,
    }

    with open(OUTPUT_DIR / "logistic_regression_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()