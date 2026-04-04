# Week 03 — Feature Engineering & Evaluation Metrics

**ML Engineer Journey | Phase 1 — ML Fundamentals**  
**Periode:** 24–29 Mar 2026  
**Difficulty:** ★★☆

---

## 🎯 Tujuan Minggu Ini

Minggu ketiga berfokus pada dua area yang sering diremehkan pemula:
**Feature Engineering** (bagaimana menyiapkan data agar model bisa belajar lebih baik)
dan **Evaluation Metrics** (bagaimana mengukur performa model secara jujur, bukan hanya accuracy).

---

## 📅 Daily Breakdown

| Hari | Topik | Goals |
|------|-------|-------|
| D01 | Feature Scaling & Encoding | StandardScaler, MinMaxScaler, OneHotEncoder — kapan pakai yang mana |
| D02 | Missing Value Strategies | SimpleImputer (mean/median/mode), KNNImputer — dampak ke model performance |
| D03 | Feature Creation & Selection | Buat fitur baru dari fitur existing, SelectKBest + feature importance |
| D04 | Precision, Recall, F1 | Confusion matrix, classification report, kapan Recall > Precision |
| D05 | ROC-AUC & Threshold Tuning | roc_curve, pr_curve, Youden's J, threshold experiment 0.1–0.9 |
| D06 | Improved Notebook + Polish | Pipeline terintegrasi end-to-end, siap push ke GitHub |

---

## 📁 Struktur Folder

```
week-03/
├── notebooks/
│   └── week03_feature_engineering_evaluation.ipynb   ← pipeline utama
├── scripts/
│   ├── feature_engineering.py    ← scaling, encoding, imputation utils
│   ├── model_evaluation.py       ← ModelEvaluator class (D04)
│   └── roc_auc.py                ← roc_curve, pr_curve, threshold tuning (D05)
├── data/                         ← tidak di-push (.gitignore)
└── README.md                     ← file ini
```

---

## 🔬 Dataset

**Breast Cancer Wisconsin** (bawaan sklearn)  
- 569 sampel, 30 numeric features  
- Binary classification: Malignant (0) vs Benign (1)  
- Class ratio: ~37% Malignant / ~63% Benign → ringan imbalanced  
- Cocok untuk demonstrasi ROC vs PR curve tradeoff

---

## 🛠️ Pipeline End-to-End (W03D06)

```
Raw Data
    │
    ▼
[1] Train-Test Split (stratify=y)    ← WAJIB SEBELUM fit preprocessor
    │
    ▼
[2] Missing Value Imputation         ← fit di X_train, transform X_test
    │   SimpleImputer(strategy='median')
    │
    ▼
[3] Feature Scaling                  ← fit di X_train SAJA
    │   StandardScaler
    │
    ▼
[4] Feature Creation                 ← fitur baru dari kombinasi existing
    │   interaction, ratio, log transform
    │
    ▼
[5] Feature Selection                ← fit di X_train, transform X_test
    │   SelectKBest(f_classif, k=15)
    │
    ▼
[6] Model Training
    │   LogisticRegression + RandomForest
    │
    ▼
[7] Evaluation
        Confusion Matrix + Classification Report
        ROC Curve + PR Curve
        Threshold Tuning (Youden's J)
```

---

## 💡 Key Concepts

### Feature Scaling

| Scaler | Formula | Kapan Pakai |
|--------|---------|-------------|
| `StandardScaler` | z = (x − μ) / σ | Default — fitur approx normal |
| `MinMaxScaler` | x' = (x − min) / (max − min) | Neural network, tanpa outlier ekstrem |
| `RobustScaler` | x' = (x − Q2) / (Q3 − Q1) | Ada outlier serius |

**Aturan wajib:** `fit_transform` hanya di training set. `transform` saja di test set.  
Melanggar ini = **data leakage**.

---

### Missing Value Strategies

| Strategi | Kapan Pakai |
|----------|-------------|
| `mean` | Distribusi normal, tidak ada outlier |
| `median` | Distribusi skewed, ada outlier — **lebih robust** |
| `most_frequent` | Fitur kategorikal |
| `KNNImputer(k=5)` | Perlu akurasi tinggi, dataset tidak terlalu besar |
| Drop row | Missing < 1%, data sangat banyak |

---

### Precision, Recall, F1

```
Precision  = TP / (TP + FP)   → Dari semua yang diprediksi positif, berapa yang benar?
Recall     = TP / (TP + FN)   → Dari semua yang benar positif, berapa yang berhasil ditangkap?
F1         = 2 × P × R / (P + R)  → Harmonic mean keduanya
```

**Kapan prioritaskan Recall:** Medical screening, fraud detection — false negative mahal.  
**Kapan prioritaskan Precision:** Spam filter, recommendation — false positive mengganggu.

---

### ROC-AUC & Threshold Tuning

**AUC** mengukur kemampuan ranking model, bukan accuracy.  
AUC = 0.95 → ada 95% kemungkinan model memberi skor lebih tinggi ke sampel positif daripada negatif yang dipilih secara acak.

**AUC tidak berubah saat threshold berubah.** Ini properti intrinsik model, bukan cut-off decision.

| Metode | Cara | Optimal Untuk |
|--------|------|---------------|
| Default | threshold = 0.5 | General purpose |
| Youden's J | `argmax(TPR - FPR)` | Medical screening |
| F1 Maximum | `argmax(F1)` over thresholds | Imbalanced dataset |
| High Recall | threshold rendah (~0.3) | Tidak boleh miss positive |
| High Precision | threshold tinggi (~0.7) | Tidak boleh false alarm |

**ROC vs PR Curve:**  
ROC curve optimistis pada imbalanced data karena TN yang banyak menekan FPR.  
Gunakan **PR curve** untuk gambaran lebih jujur pada dataset imbalanced.

---

## ⚠️ Common Gotchas

```python
# ❌ SALAH — data leakage
scaler.fit_transform(X)        # lalu train_test_split
selector.fit_transform(X, y)   # sebelum split

# ✅ BENAR
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
X_train_scaled = scaler.fit_transform(X_train)   # fit di train
X_test_scaled  = scaler.transform(X_test)        # transform saja

# ❌ SALAH — predict() bukan probabilitas
fpr, tpr, _ = roc_curve(y_test, model.predict(X_test))

# ✅ BENAR — gunakan predict_proba
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)

# ⚠️ GOTCHA — precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
# precision & recall panjang n+1, thresholds panjang n
# Gunakan [:-1] saat plot bersama thresholds:
plt.plot(thresholds, precision[:-1], recall[:-1])
```

---

## 📊 Hasil Evaluasi

| Model | AUC-ROC | Avg Precision | F1 (default) | Recall (Youden) |
|-------|---------|---------------|--------------|-----------------|
| Logistic Regression | ~0.995 | ~0.997 | ~0.97 | ~0.97 |
| Random Forest | ~0.998 | ~0.998 | ~0.97 | ~0.98 |

*Angka aktual muncul setelah notebook dijalankan.*

---

## 📦 Output Files

| File | Keterangan |
|------|-----------|
| `notebooks/week03_feature_engineering_evaluation.ipynb` | Pipeline utama — siap run |
| `scripts/roc_auc.py` | ROC, PR curve, ModelEvaluator class |
| `W03D06_Improved_Notebook_Polish.pdf` | Ringkasan materi lengkap |

---

## 🔗 Referensi

- [scikit-learn: Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn: Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [scikit-learn: Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Understanding ROC Curves — Google Developers](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

---

## 📌 Next — Week 04

**XGBoost, RandomForest & Tuning Serius** ★★★

- RandomForest: bagging & feature importance
- XGBoost: boosting concept mendalam
- Hyperparameter tuning: GridSearchCV, RandomizedSearch
- SHAP values — explainability dasar
- Mini Project: Tuned ML Model end-to-end

---

*github.com/lolivampire/ML-Project*