"""
K-means clustering: discovering patient risk segments using only clinical
features (age, glucose, BMI, hypertension, heart disease) WITHOUT ever
using the stroke label. The stroke rate per discovered cluster is computed
only after fitting, purely as a validation check: do groups formed with
zero knowledge of stroke outcomes still differ meaningfully in how often
stroke actually occurred?

Run with: python src/cluster_patients.py
Produces: outputs/cluster_profile.csv, outputs/figures/patient_clusters.png
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_access import get_full_dataset
from preprocessing import filter_unmodelable_records

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

CLUSTER_NUMERIC_COLS = ["age", "avg_glucose_level", "bmi"]
CLUSTER_BINARY_COLS = ["hypertension", "heart_disease"]


def build_cluster_features(df: pd.DataFrame) -> np.ndarray:
    numeric_imputed = SimpleImputer(strategy="median").fit_transform(df[CLUSTER_NUMERIC_COLS])
    numeric_scaled = StandardScaler().fit_transform(numeric_imputed)
    return np.hstack([numeric_scaled, df[CLUSTER_BINARY_COLS].values])


def choose_k(X, k_range=range(2, 9)):
    print("K | inertia | silhouette")
    scores = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        scores[k] = sil
        print(f"{k} | {km.inertia_:.1f} | {sil:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"\nBest silhouette score at K={best_k}")
    return best_k


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    raw = get_full_dataset()
    filtered = filter_unmodelable_records(raw)

    X = build_cluster_features(filtered)

    best_k = choose_k(X)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(X)
    filtered = filtered.copy()
    filtered["cluster"] = km.labels_

    profile = filtered.groupby("cluster").agg(
        n_patients=("stroke", "size"),
        avg_age=("age", "mean"),
        avg_glucose=("avg_glucose_level", "mean"),
        avg_bmi=("bmi", "mean"),
        hypertension_rate=("hypertension", "mean"),
        heart_disease_rate=("heart_disease", "mean"),
        stroke_rate=("stroke", "mean"),
    ).round(3)
    print("\nCluster profile (stroke_rate computed AFTER clustering, for validation only):")
    print(profile)
    profile.to_csv(OUTPUT_DIR / "cluster_profile.csv")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)
    print(f"\nPCA captures {pca.explained_variance_ratio_.sum()*100:.0f}% of total variance in 2 dimensions")

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(best_k):
        mask = km.labels_ == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5,
                   color=colors[c % len(colors)], label=f"Cluster {c}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% of variance)")
    ax.set_title("Patient clusters (PCA projection)")
    ax.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "patient_clusters.png", dpi=150)
    print(f"Saved cluster visualization to {FIGURES_DIR / 'patient_clusters.png'}")


if __name__ == "__main__":
    main()