
# This file relies on ColumnTransformer.set_output(transform="pandas"), which
# changes its actual return type to a DataFrame at runtime. Pylance's type
# stubs for scikit-learn don't know about that call, so they report the
# older, generic return type (numpy array / sparse matrix) and flag every
# `.columns` access here as an error, even though the code has been verified
# to run correctly. Firth's `feature_names_in_` has the same underlying
# issue: it's set dynamically during `.fit()`, and this package's type
# stubs don't declare it. Both are confirmed false positives, not real bugs.
# pyright: reportAttributeAccessIssue=false, reportArgumentType=false


import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_access import get_full_dataset
from preprocessing import build_preprocessing_pipeline, filter_unmodelable_records

import numpy as np
import pandas as pd
from firthmodels import FirthLogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    raw = get_full_dataset()
    filtered = filter_unmodelable_records(raw)

    X = filtered.drop(columns=["stroke"])
    y = filtered["stroke"].astype(int)

    # Same split, same random_state, as train_logistic.py -- this model is
    # fit on the identical training population, so both stories (predictive
    # performance and coefficient interpretation) describe the same data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessing_pipeline()
    X_train_ready = preprocessor.fit_transform(X_train)
    # ColumnTransformer prefixes every column with its transformer's name
    # (num__, cat__, bin__) to prevent name clashes; strip that prefix here
    # purely for a readable report -- it has no effect on the numbers.
    X_train_ready.columns = [c.split("__", 1)[1] for c in X_train_ready.columns]

    model = FirthLogisticRegression()
    model.fit(X_train_ready, y_train)

    print(f"Converged: {model.converged_} ({model.n_iter_} iterations)")
    print(f"Never_worked category present: "
          f"{'work_type_Never_worked' in X_train_ready.columns}")

    ci = model.conf_int()  # Wald 95% CIs; intercept is the last row
    names = list(model.feature_names_in_) + ["Intercept"]
    coefs = np.append(model.coef_, model.intercept_)

    summary = pd.DataFrame({
        "coef": coefs,
        "std_err": model.bse_,
        "ci_lower": ci[:, 0],
        "ci_upper": ci[:, 1],
        "p_value": model.pvalues_,
    }, index=names)
    summary["odds_ratio"] = np.exp(summary["coef"])
    summary["or_ci_lower"] = np.exp(summary["ci_lower"])
    summary["or_ci_upper"] = np.exp(summary["ci_upper"])
    summary = summary.sort_values("p_value")

    pd.set_option("display.width", 160)
    print()
    print(summary.round(4))

    summary.to_csv(OUTPUT_DIR / "inferential_model_coefficients.csv")
    print(f"\nSaved to {OUTPUT_DIR / 'inferential_model_coefficients.csv'}")


if __name__ == "__main__":
    main()