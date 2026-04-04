# Week 02 — Supervised Learning Fundamentals

**ML Engineer Journey | Phase 1 — ML Fundamentals**  
**Durasi**: 6 hari · 2–3 jam/hari  
**Status**: ✅ DONE

---

## Topik yang Dipelajari

| Hari | Topik | Output |
|------|-------|--------|
| D01 | Linear Regression mendalam | `scripts/linear_regression.py` |
| D02 | Logistic Regression & decision boundary | `scripts/logistic_regression.py` |
| D03 | Decision Trees: splitting criteria | `scripts/decision_tree.py` |
| D04 | Cross-validation (K-Fold, Stratified) | `scripts/cross_validation.py` |
| D05 | Bias-variance tradeoff | `scripts/bias_variance.py` |
| D06 | Mini project: Classification | `notebooks/mini_project_classification.ipynb` |

---

## Mini Project — Classification (W02D06)

**Dataset**: Breast Cancer Wisconsin (sklearn built-in)  
**Task**: Binary classification — malignant (0) vs benign (1)  
**Models**: DummyClassifier, Linear Regression (baseline), Logistic Regression, Decision Tree

### Hasil

| Model | Train Acc | Test Acc | CV Mean | CV Std |
|-------|-----------|----------|---------|--------|
| Dummy (majority) | 0.632 | 0.632 | N/A | N/A |
| Linear Regression | 0.968 | 0.956 | N/A | N/A |
| **Logistic Regression** | **0.991** | **0.982** | **0.978** | **0.008** |
| Decision Tree | 1.000 | 0.912 | 0.916 | 0.015 |

### Model Terpilih: Logistic Regression

Dipilih berdasarkan learning curve — bukan sekadar test accuracy tertinggi.
Training score dan CV score konvergen dengan gap 0.011 (sweet spot: low bias,
low variance). CV score stabil di 0.978.

Decision Tree di-reject meskipun training accuracy 1.0: gap 0.084 yang stagnan
seiring penambahan data mengindikasikan high variance (overfitting) dan model
problem — bukan data problem.

### Plots

```
outputs/plots/
├── class_distribution.png   — distribusi kelas malignant vs benign
├── learning_curves.png      — diagnosis bias-variance tiap model
└── model_comparison.png     — perbandingan test accuracy & CV score
```

---

## Key Learnings Minggu Ini

**Linear Regression**
- Koefisien = seberapa besar pengaruh tiap fitur terhadap output
- Implementasi dari scratch dengan numpy, bukan hanya `fit()`
- RSS, MSE, R² — cara mengukur seberapa baik model fit ke data

**Logistic Regression**
- Output adalah probabilitas (0–1), bukan nilai kontinu
- Decision boundary: garis (atau hyperplane) yang memisahkan dua kelas
- Log-odds: transformasi matematika yang membuat output linear bisa jadi probabilitas

**Decision Trees**
- Splitting criteria: Gini impurity vs Information Gain (Entropy)
- Tanpa batasan depth → pohon tumbuh sampai hafal semua training data (overfitting)
- `max_depth`, `min_samples_split` — parameter untuk kontrol kompleksitas

**Cross-validation**
- K-Fold: data dibagi k bagian, model dilatih dan divalidasi k kali
- StratifiedKFold: wajib untuk dataset imbalanced — proporsi kelas terjaga di setiap fold
- CV score lebih reliable dari single train-test split

**Bias-Variance Tradeoff**
- Bias = error dari asumsi terlalu kuat → underfitting (model terlalu simpel)
- Variance = sensitivitas berlebih terhadap training data → overfitting (model terlalu kompleks)
- Total error = Bias² + Variance + Irreducible Error
- Learning curve adalah alat diagnosis utama — bukan cross_val_score()

**Urutan Diagnosis (wajib diingat)**
```
plot learning curve → baca gap → tentukan bias/variance → pilih aksi → cross_val_score
```

---

## Cara Menjalankan

```bash
# Aktifkan virtual environment
venv\Scripts\Activate.ps1          # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Jalankan mini project
python week-02/scripts/classification_pipeline.py

# Atau buka notebook
jupyter notebook week-02/notebooks/mini_project_classification.ipynb
```

---

## Struktur Folder

```
week-02/
├── notebooks/
│   └── mini_project_classification.ipynb
├── scripts/
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   ├── cross_validation.py
│   ├── bias_variance.py
│   └── classification_pipeline.py
├── outputs/
│   └── plots/
│       ├── class_distribution.png
│       ├── learning_curves.png
│       └── model_comparison.png
└── README.md
```

---

*Next: [Week 03 — Feature Engineering & Evaluation Metrics](../week-03/README.md)*  
*GitHub: [lolivampire/ML-Project](https://github.com/lolivampire/ML-Project)*
