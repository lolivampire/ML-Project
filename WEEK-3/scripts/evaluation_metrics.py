"""
W03D04 — Precision, Recall, F1
Simulasi confusion matrix + interpretasi mendalam
Dataset: Titanic (lanjutan dari minggu sebelumnya)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

# ── 1. BUAT DATASET IMBALANCED (simulasi deteksi penipuan) ────────────
# 90% kelas 0 (normal), 10% kelas 1 (fraud)
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.9, 0.1],   # 900 normal, 100 fraud
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    # stratify=y → proporsi kelas dijaga sama di train & test
)

# ── 2. LATIH MODEL ────────────────────────────────────────────────────
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── 3. CONFUSION MATRIX ───────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
# cm[0,0] = TN, cm[0,1] = FP
# cm[1,0] = FN, cm[1,1] = TP
TN, FP, FN, TP = cm.ravel()  # unpack semua 4 sel

print("=" * 45)
print("CONFUSION MATRIX")
print("=" * 45)
print(f"          Pred Neg    Pred Pos")
print(f"Actual Neg   {TN:>5}       {FP:>5}   (TN, FP)")
print(f"Actual Pos   {FN:>5}       {TP:>5}   (FN, TP)")

# ── 4. HITUNG METRIK ──────────────────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)

print("\n" + "=" * 45)
print("METRIK EVALUASI")
print("=" * 45)
print(f"Accuracy  : {acc:.3f}  ← menyesatkan di imbalanced!")
print(f"Precision : {prec:.3f}  ← dari yang kamu tuduh fraud, berapa yang beneran?")
print(f"Recall    : {rec:.3f}  ← dari semua fraud nyata, berapa yang ketangkap?")
print(f"F1 Score  : {f1:.3f}  ← balance precision + recall")

# ── 5. SIMULASI: KONSEKUENSI PILIHAN METRIK ───────────────────────────
print("\n" + "=" * 45)
print("INTERPRETASI BISNIS")
print("=" * 45)
print(f"Total transaksi test : {len(y_test)}")
print(f"Total fraud nyata    : {y_test.sum()}")
print(f"Fraud terdeteksi     : {TP}  (Recall = {TP}/{y_test.sum()})")
print(f"Fraud yang lolos     : {FN}  ← setiap ini = kerugian nyata")
print(f"False alarm          : {FP}  ← blokir transaksi sah, user kesal")

# ── 6. CLASSIFICATION REPORT (ringkasan lengkap) ─────────────────────
print("\n" + "=" * 45)
print("CLASSIFICATION REPORT (sklearn)")
print("=" * 45)
print(classification_report(y_test, y_pred,
target_names=["Normal", "Fraud"]))