"""
model_serialization.py
W06D01 — Model Serialization (OOP, Versioning, & Size Comparison)
"""

import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def generate_and_train_model() -> Tuple[LogisticRegression, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load dataset Breast Cancer dan latih model Logistic Regression."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test


# ── PENDEKATAN OOP UNTUK MODEL MANAGEMENT ─────────────────────────────────────

class ModelManager:
    """Class untuk mengelola penyimpanan, pemuatan, dan versioning model ML."""
    
    def __init__(self, base_dir: str | Path = "models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_file_size_kb(filepath: str | Path) -> float:
        """Utility method (Static) untuk menghitung ukuran file dalam KB."""
        return Path(filepath).stat().st_size / 1024

    def save_model(self, model: Any, filename: str, metadata: Optional[Dict[str, Any]] = None, method: str = "joblib") -> Path:
        """Menyimpan model ke dalam disk (mendukung joblib dan pickle)."""
        filepath = self.base_dir / filename
        
        final_metadata = {
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now().isoformat(),
        }
        
        if metadata:
            final_metadata.update(metadata)

        model_bundle = {
            "model": model,
            "metadata": final_metadata
        }

        # Logika pemilihan metode penyimpanan
        if method == "joblib":
            joblib.dump(model_bundle, filepath, compress=3)
        elif method == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(model_bundle, f)
        else:
            raise ValueError("Metode tidak didukung. Pilih 'joblib' atau 'pickle'.")

        print(f"[INFO] Model disimpan dengan {method} di: {filepath}")
        return filepath

    def load_model(self, filename: str) -> Tuple[Any, Dict[str, Any]]:
        """Memuat model dari disk."""
        filepath = self.base_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File model tidak ditemukan: {filepath}")

        # Kita asumsikan load utamanya selalu pakai joblib untuk kemudahan
        # (Joblib bisa membaca file .pkl juga)
        bundle = joblib.load(filepath)

        if isinstance(bundle, dict) and "model" in bundle and "metadata" in bundle:
            return bundle["model"], bundle["metadata"]
        else:
            return bundle, {}


# ── BLOK EKSEKUSI UTAMA ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("1. Membuat dan melatih model (Logistic Regression)...")
    model, X_train, X_test, y_train, y_test = generate_and_train_model()
    
    original_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Akurasi training: {original_accuracy:.4f}\n")

    registry = ModelManager(base_dir="models")

    custom_metadata = {
        "accuracy": float(original_accuracy),
        "n_features": X_train.shape[1],
        "model_type": "Logistic Regression",
        "max_iter": model.max_iter,
        "author": "Yotsubae"
    }

    print("2. Menyimpan model untuk perbandingan ukuran...")
    # Simpan dengan joblib
    joblib_path = registry.save_model(model, "bc_model_v1.joblib", metadata=custom_metadata, method="joblib")
    # Simpan dengan pickle (untuk komparasi saja)
    pickle_path = registry.save_model(model, "bc_model_v1.pkl", metadata=custom_metadata, method="pickle")

    # Ambil ukuran file menggunakan static method
    size_joblib = ModelManager.get_file_size_kb(joblib_path)
    size_pickle = ModelManager.get_file_size_kb(pickle_path)

    print("\n3. Memuat model (dari joblib)...")
    loaded_model, metadata = registry.load_model("bc_model_v1.joblib")
    
    print("\n--- METADATA DITEMUKAN ---")
    for key, value in metadata.items():
        print(f" {key.ljust(15)} : {value}")
    
    print("\n4. Verifikasi Prediksi...")
    preds_original = model.predict(X_test)
    preds_loaded   = loaded_model.predict(X_test)
    print(f"Prediksi identik: {np.array_equal(preds_original, preds_loaded)}")
    file_size = (registry.base_dir / "bc_model_v1.joblib").stat().st_size / 1024
    print(f"Ukuran file     : {file_size:.1f} KB")

    print("\n--- PERBANDINGAN UKURAN FILE --- ")
    print(f"  joblib : {size_joblib:.1f} KB")
    print(f"  pickle : {size_pickle:.1f} KB")
    
    # Menghindari hasil negatif jika ukuran kebetulan sama/berbeda sedikit
    selisih = abs(size_pickle - size_joblib)
    if size_joblib < size_pickle:
        print(f"  Selisih: {selisih:.1f} KB (joblib lebih kecil)")
    else:
        print(f"  Selisih: {selisih:.1f} KB (pickle lebih kecil)")