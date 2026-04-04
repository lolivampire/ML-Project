# Week 04 Mini Project — Breast Cancer Classification

**Phase 1 — ML Fundamentals | Week 04**

End-to-end ML pipeline dengan hyperparameter tuning dan SHAP explainability
pada Breast Cancer Wisconsin Dataset.

---

## Tujuan

Membangun pipeline klasifikasi biner yang:
1. Membandingkan performa RandomForest vs XGBoost
2. Melakukan hyperparameter tuning dengan GridSearchCV dan RandomizedSearchCV
3. Menganalisis feature importance menggunakan SHAP values

---

## Dataset

| Property | Value |
|---|---|
| Source | `sklearn.datasets.load_breast_cancer` |
| Samples | 569 |
| Features | 30 (numeric) |
| Target | 0 = Malignant, 1 = Benign |
| Missing values | None |
| Class distribution | 212 Malignant / 357 Benign |

---

## Metode

### Pipeline

Data Loading → EDA → Preprocessing → Baseline → Tuning → Evaluation → SHAP

### Preprocessing
- `StandardScaler` via `sklearn.Pipeline`
- `train_test_split` dengan `stratify=y`, `test_size=0.2`, `random_state=42`
- `StratifiedKFold(n_splits=5)` untuk cross-validation

### Tuning Strategy
| Model | Method | Search Space |
|---|---|---|
| RandomForest | GridSearchCV | n_estimators, max_depth, min_samples_split, max_features |
| XGBoost | RandomizedSearchCV (n_iter=50) | n_estimators, max_depth, learning_rate, subsample, colsample_bytree |

---

## Hasil

### Model Performance

| Model | Baseline AUC | Tuned AUC | Delta |
|---|---|---|---|
| RandomForest | 0.9939 | 0.9939 | +0.0000 (baseline already optimal) |
| XGBoost | 0.9901 | 0.9924 | +0.0023 |

### Confusion Matrix — XGBoost (Best Model)

| | Predicted Malignant | Predicted Benign |
|---|---|---|
| **Actual Malignant** | TN (correct) | FN: 0 |
| **Actual Benign** | FP: 4 | TP (correct) |

**FN = 0** — tidak ada kasus Malignant yang terlewat.

### Top Predictive Features (SHAP)
1. `worst concave points`
2. `worst perimeter`
3. `mean concave points`

---

## Cara Run

### Requirements
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost shap
```

### Jalankan Notebook
```bash
cd week-04/notebooks
jupyter notebook w04_mini_project_final.ipynb
```

Jalankan semua cell secara berurutan (Run All).
Output plots tersimpan otomatis ke `week-04/outputs/`.

### Output Files

week-04/
├── notebooks/
│   └── w04_mini_project_final.ipynb   ← main notebook
└── outputs/
├── eda_correlation.png
├── roc_curves.png
├── shap_summary_rf.png
├── shap_summary_xgb.png
└── shap_waterfall_fn.png           ← tidak ada jika FN=0

---

## Key Findings

1. **RandomForest baseline sudah sangat optimal** — GridSearch tidak menemukan
   kombinasi parameter yang meningkatkan AUC dari 0.9939. Ini umum terjadi
   pada dataset yang relatif bersih dan separable seperti Breast Cancer Wisconsin.

2. **XGBoost mendapat manfaat dari tuning** — RandomizedSearch dengan 50 iterasi
   meningkatkan AUC sebesar 0.0023. Ruang parameter XGBoost lebih kompleks,
   sehingga default parameter jarang optimal.

3. **FN = 0 adalah target utama domain medis** — Model tidak melewatkan satu pun
   kasus Malignant di test set. 4 False Positive (Benign diprediksi Malignant)
   jauh lebih dapat ditoleransi dibanding melewatkan diagnosis kanker.

4. **SHAP mengkonfirmasi domain knowledge** — `worst concave points` dan
   `worst perimeter` adalah fitur morfologi sel yang memang digunakan dokter
   patologi untuk menilai keganasan tumor.

---

## Koneksi ke Materi Week 04

| Topik | Implementasi |
|---|---|
| W04D01 — RandomForest & Bagging | `RandomForestClassifier`, feature importance via SHAP |
| W04D02 — XGBoost & Boosting | `XGBClassifier` dengan `eval_metric='logloss'` |
| W04D03 — GridSearchCV | Tuning RF dengan `GridSearchCV` |
| W04D04 — RandomizedSearch | Tuning XGB dengan `RandomizedSearchCV`, `scipy.stats` distributions |
| W04D05 — SHAP Values | `TreeExplainer`, summary plot, waterfall plot |

---

*Week 04 | Phase 1 — ML Fundamentals | github.com/lolivampire/ML-Project*