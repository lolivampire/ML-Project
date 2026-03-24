"""
W03D02 — Missing Value Strategies
Demonstrasi: SimpleImputer (mean/median/mode), KNNImputer,
dan missing indicator column. Semua dalam Pipeline yang benar.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import make_classification

def generate_missing_data(n_samples: int = 500, missing_rate: float = 0.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Menghasilkan dataset sintetis dengan missing values (NaN) secara acak.
    """
    X, y = make_classification(
        n_samples=n_samples, n_features=4, n_informative=3, n_redundant=0, random_state=42
    )
    
    # Injeksi missing values
    rng = np.random.default_rng(42)
    mask = rng.random(X.shape) < missing_rate
    X_missing = X.astype(float)
    X_missing[mask] = np.nan
    
    return X_missing, y

def get_pipelines() -> Dict[str, Pipeline]:
    """
    Mendefinisikan berbagai strategi imputasi dalam format Scikit-Learn Pipeline.
    """
    # Menggunakan add_indicator=True adalah cara modern menggantikan FeatureUnion
    # untuk menangani 'Missing Indicator' secara otomatis.
    
    pipelines = {
        "Mean Imputation": Pipeline([
            ("imputer", SimpleImputer(strategy="mean", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42))
        ]),
        
        "Median Imputation": Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42))
        ]),
        
        "KNN Imputation": Pipeline([
            # KNNImputer butuh data ter-scale tapi StandardScaler tidak suka NaN.
            # Kita bisa pakai SimpleImputer ringan dulu atau langsung KNN jika range data mirip.
            ("imputer", KNNImputer(n_neighbors=5, add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42))
        ])
    }
    return pipelines

def evaluate_models(X: np.ndarray, y: np.ndarray, pipelines: Dict[str, Pipeline]):
    """
    Melakukan evaluasi performa setiap pipeline menggunakan Cross-Validation.
    """
    print(f"{'Strategy':<20} | {'Mean Accuracy':<15} | {'Std Dev'}")
    print("-" * 50)
    
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        print(f"{name:<20} | {scores.mean():.4f}          | {scores.std():.4f}")

# ── MAIN EXECUTION ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Persiapan Data
    X_missing, y = generate_missing_data()
    
    # 2. Split Data (Mencegah Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset Info: {X_train.shape[0]} samples, {np.isnan(X_train).sum()} total NaNs\n")

    # 3. Definisi Strategi
    model_strategies = get_pipelines()

    # 4. Evaluasi
    evaluate_models(X_train, y_train, model_strategies)