# create_dummy_model.py
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import os

print("Membuat model dummy...")
# Buat data sintetis: 100 sampel, 5 fitur
X, y = make_classification(n_samples=100, n_features=5, random_state=42)

# Latih model regresi logistik sederhana
model = LogisticRegression()
model.fit(X, y)

# Simpan ke folder yang benar (sesuaikan path jika berbeda)
MODEL_PATH =  Path(__file__).parent / "model"
os.makedirs(MODEL_PATH, exist_ok=True)
joblib.dump(model, MODEL_PATH / "model.pkl")
print("Model dummy berhasil disimpan di ", MODEL_PATH / "/model.pkl")