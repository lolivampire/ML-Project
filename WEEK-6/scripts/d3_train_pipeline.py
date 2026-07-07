"""
week-06/scripts/d3_train_pipeline.py

Skrip ini mendemonstrasikan pembuatan, pelatihan, dan penyimpanan scikit-learn Pipeline.
Menyimpan tahap Preprocessing (StandardScaler) dan Model (LogisticRegression) 
sebagai satu bundle (.joblib) agar inferensi di API cukup memanggil satu fungsi .predict().
"""

import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def build_pipeline() -> Pipeline:
    """
    Membangun arsitektur scikit-learn Pipeline.
    
    Urutan di dalam list menentukan urutan eksekusi secara berurutan.
    Data akan masuk ke 'scaler' untuk dinormalisasi, lalu diteruskan ke 'model'.

    Returns:
        Pipeline: Objek pipeline yang belum dilatih (untrained).
    """
    return Pipeline([
        ("scaler", StandardScaler()),                 # Step 1: Normalisasi fitur
        ("model", LogisticRegression(max_iter=200))   # Step 2: Klasifikasi
    ])


def train_pipeline(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
    """
    Melatih pipeline menggunakan data training.
    
    Pemanggilan `pipeline.fit()` secara otomatis akan mengeksekusi:
    1. scaler.fit_transform(X_train)
    2. model.fit(X_train_scaled, y_train)

    Args:
        pipeline: Objek pipeline yang akan dilatih.
        X_train: Fitur training.
        y_train: Label/target training.

    Returns:
        Pipeline: Objek pipeline yang sudah dilatih (fitted).
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_pipeline(pipeline: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Mengevaluasi performa pipeline pada data testing.

    Pemanggilan `pipeline.predict()` secara otomatis mengeksekusi:
    1. scaler.transform(X_test) -> TIDAK fit ulang, hanya menggunakan mean/std dari data train.
    2. model.predict(X_test_scaled)

    Returns:
        float: Nilai akurasi (0.0 - 1.0).
    """
    predictions = pipeline.predict(X_test)
    return float(accuracy_score(y_test, predictions))


def save_pipeline_bundle(pipeline: Pipeline, accuracy: float, version: str = "v2") -> Path:
    """
    Menyimpan pipeline beserta metadatanya sebagai satu file bundle (.joblib).
    Path akan di-resolve secara dinamis relatif terhadap lokasi file skrip ini.
    """
    # Menyiapkan metadata untuk traceability di Production/API
    bundle: Dict[str, Any] = {
        "pipeline": pipeline,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "accuracy": round(accuracy, 4),
            "version": version,
            # Ambil n_features dari scaler (langkah pertama)
            "n_features": int(pipeline.named_steps["scaler"].n_features_in_), 
            # Ambil classes dari model (langkah terakhir)
            "classes": pipeline.named_steps["model"].classes_.tolist(), 
            "steps": [name for name, _ in pipeline.steps],
        }
    }

    # Resolusi path cerdas (menyimpan ke week-06/scripts/models/)
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"pipeline_{version}_{date_str}.joblib"
    filepath = models_dir / filename

    joblib.dump(bundle, filepath, compress=3)
    return filepath


# ── BLOK EKSEKUSI UTAMA ───────────────────────────────────────────────────────

def main() -> None:
    print("1. Menyiapkan dataset sintetis (4 fitur)...")
    X, y = make_classification(
        n_samples=500,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("2. Membangun dan melatih Pipeline...")
    pipeline = build_pipeline()
    pipeline = train_pipeline(pipeline, X_train, y_train)

    print("3. Mengevaluasi Pipeline...")
    acc = evaluate_pipeline(pipeline, X_test, y_test)
    print(f"➜ Akurasi Test: {acc:.4f}")

    print("4. Menjalankan Sanity Check (Uji Konsistensi)...")
    # Memastikan pipeline.predict() memberikan hasil yang sama persis 
    # dengan melakukan scaling dan prediksi secara manual.
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]
    
    manual_pred = model.predict(scaler.transform(X_test[:3]))
    pipe_pred = pipeline.predict(X_test[:3])
    
    assert np.array_equal(manual_pred, pipe_pred), "Sanity Check Gagal: Mismatch antara Pipeline dan Manual!"
    print("➜ Sanity Check Lulus: Logika Pipeline terbukti konsisten.")

    print("\n5. Menyimpan Pipeline Bundle...")
    saved_path = save_pipeline_bundle(pipeline, accuracy=acc, version="pipe_v1")
    print(f"✅ Pipeline tersimpan di: {saved_path}")


if __name__ == "__main__":
    print("=== Memulai Training Pipeline ===\n")
    main()