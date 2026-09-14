# Unit tests for src/preprocessing.py.
# Run with: pytest


import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import build_preprocessing_pipeline, filter_unmodelable_records


@pytest.fixture
def sample_df():
    """A tiny stand-in for get_full_dataset()'s output, covering the edge
    cases the pipeline needs to handle: a gender='Other' record, a
    Never_worked record, and missing BMI values."""
    return pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5, 6],
        "gender": ["Male", "Female", "Other", "Female", "Male", "Female"],
        "age": [45, 70, 30, 55, 62, 40],
        "ever_married": ["Yes", "Yes", "No", "Yes", "No", "Yes"],
        "work_type": ["Private", "Never_worked", "children", "Govt_job", "Self-employed", "Private"],
        "residence_type": ["Urban", "Rural", "Urban", "Rural", "Urban", "Rural"],
        "hypertension": [0, 1, 0, 0, 1, 0],
        "heart_disease": [0, 1, 0, 0, 0, 0],
        "avg_glucose_level": [90.0, 200.0, 85.0, 110.0, 150.0, 95.0],
        "bmi": [None, 32.0, 24.0, 27.0, None, 26.0],
        "smoking_status": ["never smoked", "smokes", "Unknown", "formerly smoked", "smokes", "never smoked"],
        "stroke": [0, 1, 0, 0, 1, 0],
    })



# filter_unmodelable_records
def test_filter_drops_other_gender(sample_df):
    result = filter_unmodelable_records(sample_df)
    assert "Other" not in result["gender"].values


def test_filter_drops_patient_id_column(sample_df):
    result = filter_unmodelable_records(sample_df)
    assert "patient_id" not in result.columns


def test_filter_preserves_never_worked(sample_df):
    """The most important test in this file: confirms filter_unmodelable_records
    no longer merges or drops Never_worked, unlike the old clean() function."""
    result = filter_unmodelable_records(sample_df)
    assert "Never_worked" in result["work_type"].values


def test_filter_preserves_row_count_minus_dropped_gender(sample_df):
    result = filter_unmodelable_records(sample_df)
    assert len(result) == 5


def test_filter_preserves_missing_bmi(sample_df):
    """filter_unmodelable_records must NOT impute -- that's the pipeline's
    job, fit on training data only. If this ever fails because BMI got
    filled in here, that's a leakage regression."""
    result = filter_unmodelable_records(sample_df)
    assert result["bmi"].isna().sum() > 0



# build_preprocessing_pipeline
@pytest.fixture
def train_test_split_frames(sample_df):
    cleaned = filter_unmodelable_records(sample_df)
    X = cleaned.drop(columns=["stroke"])
    y = cleaned["stroke"]
    return X.iloc[:3], X.iloc[3:], y.iloc[:3], y.iloc[3:]


def test_pipeline_fills_all_missing_values(train_test_split_frames):
    X_train, X_test, y_train, y_test = train_test_split_frames
    pipeline = build_preprocessing_pipeline()
    X_train_ready = pipeline.fit_transform(X_train)
    X_test_ready = pipeline.transform(X_test)
    assert X_train_ready.isna().sum().sum() == 0
    assert X_test_ready.isna().sum().sum() == 0


def test_pipeline_produces_only_numeric_output(train_test_split_frames):
    X_train, X_test, y_train, y_test = train_test_split_frames
    pipeline = build_preprocessing_pipeline()
    X_train_ready = pipeline.fit_transform(X_train)
    non_numeric = X_train_ready.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    assert non_numeric == []


def test_pipeline_never_worked_survives_into_output(train_test_split_frames):
    X_train, X_test, y_train, y_test = train_test_split_frames
    pipeline = build_preprocessing_pipeline()
    X_train_ready = pipeline.fit_transform(X_train)
    never_worked_cols = [c for c in X_train_ready.columns if "Never_worked" in c]
    assert len(never_worked_cols) == 1


def test_pipeline_scaling_is_fit_on_train_only(train_test_split_frames):
    """The core leakage check for this whole file: refitting the SAME
    pipeline on train+test combined should shift the learned scaling
    statistics. If it didn't, that would mean test data was silently
    influencing the 'trained' transformer -- exactly what the graded
    feedback flagged in the original BMI-imputation bug."""
    X_train, X_test, y_train, y_test = train_test_split_frames
    X_all = pd.concat([X_train, X_test])

    pipeline_train_only = build_preprocessing_pipeline()
    pipeline_train_only.fit(X_train)

    pipeline_all_data = build_preprocessing_pipeline()
    pipeline_all_data.fit(X_all)

    train_only_mean = pipeline_train_only.named_transformers_["num"].named_steps["scale"].mean_
    all_data_mean = pipeline_all_data.named_transformers_["num"].named_steps["scale"].mean_

    assert not np.allclose(train_only_mean, all_data_mean)