"""
week-06/scripts/d2_prepare_model.py

Skrip ini digunakan untuk melatih model Machine Learning (Random Forest) pada 
dataset Iris dan menyimpannya sebagai bundle (.joblib) beserta metadatanya.
Model yang dihasilkan akan di-load oleh aplikasi utama melalui lifespan/dependency.
"""

import joblib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_and_save_model(version: str = "v1") -> Path:
    """
    Melatih model klasifikasi Iris dan menyimpannya ke folder 'models/'.
    
    Path tujuan akan di-resolve secara dinamis berdasarkan lokasi skrip ini,
    sehingga tidak bergantung pada direktori terminal (Current Working Directory).

    Args:
        version (str): Versi model yang akan disimpan. Default: 'v1'.

    Returns:
        Path: Lokasi absolut tempat file model disimpan.
    """
    print("1. Memuat dataset Iris...")
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("2. Melatih model Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluasi
    accuracy = float(accuracy_score(y_test, model.predict(X_test)))

    print("3. Menyiapkan bundle dan metadata...")
    bundle: Dict[str, Any] = {
        "model": model,
        "metadata": {
            "trained_at": datetime.now().isoformat(),
            "accuracy": round(accuracy, 4),
            "n_features": int(X.shape[1]),
            "target_names": list(data.target_names),
            "version": version
        }
    }

    print("4. Menyimpan model...")
    # RESOLUSI PATH DINAMIS
    # __file__ adalah lokasi dari file skrip ini (d2_prepare_model.py)
    # .resolve().parent mengambil folder 'scripts/' tempat file ini berada
    script_dir = Path(__file__).resolve().parent
    models_dir = script_dir / "models"
    
    # Buat folder 'models/' jika belum ada
    models_dir.mkdir(parents=True, exist_ok=True)

    # Buat nama file dinamis berdasarkan tanggal hari ini (Contoh: model_v1_20260414.joblib)
    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"model_{version}_{date_str}.joblib"
    filepath = models_dir / filename

    # Simpan bundle
    joblib.dump(bundle, filepath, compress=3)
    
    print(f"\n[SUCCESS] Model berhasil disimpan!")
    print(f"➜ Lokasi  : {filepath}")
    print(f"➜ Akurasi : {accuracy:.4f}")
    
    return filepath


# ── BLOK EKSEKUSI UTAMA ───────────────────────────────────────────────────────

if __name__ == "__main__":
    # Script ini hanya dijalankan sekali secara manual sebelum server app berjalan.
    print("=== Memulai Persiapan Model ===")
    train_and_save_model(version="v1")