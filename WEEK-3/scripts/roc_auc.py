"""
W03D05 — ROC-AUC & Threshold Tuning
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    classification_report,
    roc_auc_score
)

# ── 1. DATA ──────────────────────────────────────────────────

# Buat dataset imbalanced (realistic): 80% class 0, 20% class 1
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    weights=[0.8, 0.2],   # 80/20 split — imbalanced
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# stratify=y wajib agar proporsi kelas sama di train/test

# ── 2. TRAIN MODEL ───────────────────────────────────────────

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# predict_proba → probabilitas, BUKAN label biner
# kolom [1] = probabilitas class positif (yang kita butuhkan)
y_proba = model.predict_proba(X_test)[:, 1]

# ── 3. ROC CURVE ─────────────────────────────────────────────

# roc_curve mengembalikan FPR, TPR, dan threshold
# untuk SETIAP kemungkinan threshold (n_samples threshold)
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba)

# auc() hitung luas area di bawah kurva
roc_auc = auc(fpr, tpr)
# Alternatif: roc_auc_score(y_test, y_proba) — hasil sama

print(f"ROC AUC Score: {roc_auc:.4f}")
# AUC = 0.5 → random, AUC = 1.0 → sempurna
# AUC > 0.8 sudah dianggap bagus untuk banyak use case

# ── 4. PLOT ROC CURVE ────────────────────────────────────────

plt.figure(figsize=(8, 6))

# Kurva model kita
plt.plot(fpr, tpr, color='steelblue', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')

# Garis diagonal = random classifier (AUC 0.5)
plt.plot([0, 1], [0, 1], color='gray', lw=1,
         linestyle='--', label='Random classifier')

# Titik threshold 0.5 default — tandai di kurva
idx_05 = np.argmin(np.abs(thresholds_roc - 0.5))
plt.scatter(fpr[idx_05], tpr[idx_05],
            color='red', s=100, zorder=5,
            label=f'Threshold 0.5 (FPR={fpr[idx_05]:.2f}, TPR={tpr[idx_05]:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 5. THRESHOLD TUNING ──────────────────────────────────────

print("\n── Eksperimen threshold ──")
print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>8}")
print("-" * 46)

# precision_recall_curve memberi P dan R di setiap threshold
precisions, recalls, thresholds_pr = precision_recall_curve(
    y_test, y_proba
)
# Catatan: len(thresholds_pr) = len(precisions) - 1
# karena kurva ini menambah satu titik di awal (P=1, R=0)

for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    # Konversi probabilitas ke label biner pakai threshold custom
    y_pred_custom = (y_proba >= thresh).astype(int)

    # Ambil precision dan recall dari array
    idx = np.argmin(np.abs(thresholds_pr - thresh))
    p = precisions[idx]
    r = recalls[idx]
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(f"{thresh:>10.1f} | {p:>10.3f} | {r:>10.3f} | {f1:>8.3f}")

# ── 6. PILIH THRESHOLD OPTIMAL ───────────────────────────────

# Strategi: cari threshold yang maksimalkan F1
f1_scores = (2 * precisions[:-1] * recalls[:-1] /
             (precisions[:-1] + recalls[:-1] + 1e-9))
# +1e-9 untuk hindari division by zero

best_idx = np.argmax(f1_scores)
best_threshold = thresholds_pr[best_idx]
print(f"\nBest threshold (max F1): {best_threshold:.3f}")
print(f"F1 at best threshold: {f1_scores[best_idx]:.3f}")

# ── 7. REPORT FINAL ──────────────────────────────────────────

y_pred_best = (y_proba >= best_threshold).astype(int)
print("\n── Classification Report (threshold optimal) ──")
print(classification_report(y_test, y_pred_best))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")