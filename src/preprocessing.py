"""
Preprocessing for the stroke dataset, built as a leakage-safe scikit-learn
pipeline.

"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_COLS = ["age", "avg_glucose_level", "bmi"]
BINARY_COLS = ["hypertension", "heart_disease"]
CATEGORICAL_COLS = ["gender", "ever_married", "work_type", "residence_type", "smoking_status"]


def filter_unmodelable_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["patient_id"])
    df = df[df["gender"] != "Other"].reset_index(drop=True)
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Returns an unfitted ColumnTransformer. Call .fit_transform() on
    training data, and .transform() (never .fit_transform()) on test data
    or any new data."""
    numeric_transformer = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_transformer = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_COLS),
        ("cat", categorical_transformer, CATEGORICAL_COLS),
        ("bin", "passthrough", BINARY_COLS),
    ])
    preprocessor.set_output(transform="pandas")
    return preprocessor


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_access import get_full_dataset
    from sklearn.model_selection import train_test_split

    raw = get_full_dataset()
    filtered = filter_unmodelable_records(raw)
    print("After filtering:", filtered.shape)
    print("work_type values:", sorted(filtered["work_type"].unique()))

    X = filtered.drop(columns=["stroke"])
    y = filtered["stroke"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessing_pipeline()
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)

    print("\nTraining BMI missing before transform:", X_train["bmi"].isna().sum())
    print("Transformed training shape:", X_train_ready.shape)
    print("Transformed columns:", list(X_train_ready.columns))
    print("Any missing values after transform:", X_train_ready.isna().sum().sum())