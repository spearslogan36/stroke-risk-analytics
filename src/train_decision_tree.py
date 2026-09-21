"""
Decision tree classifier for stroke prediction.

Reuses the same data pipeline as train_logistic.py, but with
scaling turned off (build_preprocessing_pipeline(scale_numeric=False)):
a tree's predictions are identical either way, but unscaled thresholds
(e.g. "age <= 47.5") are actually readable

Fits two variants, same structure as train_logistic.py: an unweighted
baseline and a class-balanced version
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_access import get_full_dataset
from preprocessing import build_preprocessing_pipeline, filter_unmodelable_records

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def fit_and_tune(X_train, y_train, class_weight):
    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline(scale_numeric=False)),
        ("classifier", DecisionTreeClassifier(class_weight=class_weight, random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "classifier__max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "classifier__min_samples_leaf": [5, 10, 20, 40],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    raw = get_full_dataset()
    filtered = filter_unmodelable_records(raw)
    X = filtered.drop(columns=["stroke"])
    y = filtered["stroke"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    baseline_grid = fit_and_tune(X_train, y_train, class_weight=None)
    baseline_pipeline = baseline_grid.best_estimator_
    baseline_pred = baseline_pipeline.predict(X_test)
    baseline_proba = baseline_pipeline.predict_proba(X_test)[:, 1]

    balanced_grid = fit_and_tune(X_train, y_train, class_weight="balanced")
    balanced_pipeline = balanced_grid.best_estimator_
    balanced_pred = balanced_pipeline.predict(X_test)
    balanced_proba = balanced_pipeline.predict_proba(X_test)[:, 1]

    results = {
        "baseline": {
            "best_params": baseline_grid.best_params_,
            "cv_roc_auc": baseline_grid.best_score_,
            **evaluate(y_test, baseline_pred, baseline_proba),
        },
        "balanced": {
            "best_params": balanced_grid.best_params_,
            "cv_roc_auc": balanced_grid.best_score_,
            **evaluate(y_test, balanced_pred, balanced_proba),
        },
    }

    with open(OUTPUT_DIR / "decision_tree_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    tree_model = balanced_pipeline.named_steps["classifier"]
    feature_names = balanced_pipeline.named_steps["preprocessor"].get_feature_names_out()
    feature_names = [f.split("__", 1)[1] for f in feature_names]

    rules_text = export_text(tree_model, feature_names=feature_names, max_depth=3)
    print("\nDecision rules (top 3 levels):\n")
    print(rules_text)
    with open(OUTPUT_DIR / "decision_tree_rules.txt", "w") as f:
        f.write(rules_text)

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(tree_model, feature_names=feature_names, class_names=["No Stroke", "Stroke"],
              filled=True, rounded=True, fontsize=9, ax=ax)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "decision_tree.png", dpi=150)
    print(f"\nSaved tree diagram to {FIGURES_DIR / 'decision_tree.png'}")


if __name__ == "__main__":
    main()