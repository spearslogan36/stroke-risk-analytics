import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_access import get_full_dataset
from preprocessing import build_preprocessing_pipeline, filter_unmodelable_records

from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def subgroup_metrics(y_true, y_pred, n_total):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    return {"n": int(n_total), "actual_positives": int(tp + fn), "tp": int(tp), "fn": int(fn),
            "tn": int(tn), "fp": int(fp), "recall": recall, "false_positive_rate": fpr}


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

    pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", LogisticRegression(random_state=RANDOM_STATE)),
    ])
    param_grid = [
        {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100], "classifier__solver": ["lbfgs"],
         "classifier__penalty": ["l2"], "classifier__class_weight": ["balanced"], "classifier__max_iter": [2000]},
        {"classifier__C": [0.001, 0.01, 0.1, 1, 10, 100], "classifier__solver": ["liblinear"],
         "classifier__penalty": ["l1"], "classifier__class_weight": ["balanced"], "classifier__max_iter": [2000]},
    ]
    grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    # X_test still holds the raw, un-encoded 'gender' column -- the pipeline
    # only ever sees a transformed COPY when we call .predict(), so we can
    # still slice the original DataFrame by group afterward.
    y_pred = best_pipeline.predict(X_test)

    results = {}
    for group in ["Male", "Female"]:
        mask = (X_test["gender"] == group).values
        results[group] = subgroup_metrics(y_test[mask], y_pred[mask], mask.sum())

    contingency_table = [
        [results["Male"]["tp"], results["Male"]["fn"]],
        [results["Female"]["tp"], results["Female"]["fn"]],
    ]
    odds_ratio, p_value = fisher_exact(contingency_table)
    results["recall_gap_significance_test"] = {
        "test": "Fisher's exact test on TP/FN counts by gender",
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "interpretation": (
            "not statistically significant (p >= 0.05): no evidence of a real "
            "recall disparity by gender, though the sample is too small to "
            "rule one out with confidence"
        ) if p_value >= 0.05 else (
            "statistically significant (p < 0.05): evidence of a real "
            "recall disparity by gender worth investigating further"
        ),
    }

    with open(OUTPUT_DIR / "fairness_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()