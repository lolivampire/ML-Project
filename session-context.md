# 📍 Session Context — ML Engineer Journey

## Status
- **Week**: 02  **Day**: 06  **Phase**: 1 — ML Fundamentals
- **Tanggal**: [isi tanggal besok]

## Progress
| Week | Tema | Status |
|------|------|--------|
| W01 | Python & Data Foundations | ✅ DONE |
| W02 | Supervised Learning | 5/6 ✅ |
| W03 | Feature Engineering & Evaluation | ○ |
| W04 | XGBoost, RandomForest & Tuning | ○ |

## W02 — Detail Harian
| Hari | Topik | Status |
|------|-------|--------|
| D01 | Linear Regression mendalam | ✅ |
| D02 | Logistic Regression & decision boundary | ✅ |
| D03 | Decision Trees: splitting criteria | ✅ |
| D04 | Cross-validation (K-Fold, Stratified) | ✅ |
| D05 | Bias-variance tradeoff | ✅ |
| D06 | Mini project: Classification | ○ |

## 🧠 Key Learnings — W02D05
- Bias = error dari asumsi terlalu kuat — model terlalu simpel (underfitting)
- Variance = error dari sensitivitas berlebih terhadap data training (overfitting)
- Keduanya tidak bisa diminimalkan bersamaan — ada titik optimal di tengah
- Learning curve adalah alat diagnosis utama — bukan cross_val_score()
- Urutan wajib: plot learning curve → diagnosis → fix → cross_val_score()
- Gap mengecil seiring data = data problem → tambah sampel
- Gap tidak mengecil = model problem → kurangi kompleksitas
- Tambah data ≠ tambah fitur — dua hal berbeda, solusi berbeda
- learning_curve() untuk diagnosis, cross_val_score() untuk evaluasi final
- Total error = Bias² + Variance + Irreducible Error

## ⚠️ Blur / Perlu Review — BACA INI SEBELUM SESI BERIKUTNYA
### [PRIORITAS TINGGI] Urutan diagnosis sebelum aksi
User sempat langsung menyebut solusi (turunkan max_depth, pruning)
sebelum menyebut diagnosis (plot learning curve dulu). Kebiasaan ini
perlu diperkuat: di sesi berikutnya yang melibatkan evaluasi model,
tanya dulu "apa yang sudah kamu lihat dari grafik?" sebelum diskusi solusi.

### [PRIORITAS MEDIUM] Tambah data vs tambah fitur
User belum otomatis membedakan keduanya — perlu dikonfirmasi lagi
di sesi W03 yang akan banyak membahas feature engineering.

### [RESOLVED] clone() dan state estimator
Tidak muncul sebagai masalah di sesi ini. Monitor terus di W02D06.

### [RESOLVED] Membaca output sebelum menarik kesimpulan
User berhasil menjawab semua pertanyaan pengamatan dari output aktual,
bukan dari asumsi. Progress nyata dibanding sesi sebelumnya.

## 📁 Output Files
- `week-02/scripts/decision_tree.py` ✅
- `week-02/scripts/cross_validation.py` ✅
- `week-02/scripts/bias_variance.py` ✅
- `week-02/outputs/plots/decision_tree_analysis.png` ✅
- `week-02/outputs/plots/depth_analysis2.png` ✅
- `week-02/outputs/bias_variance/bias-variance.png` ✅
- `W02D03_Decision_Trees.pdf` ✅
- `W02D04_Cross_Validation.pdf` ✅
- `W02D05_Bias_Variance_Tradeoff.pdf` ✅
- GitHub: https://github.com/lolivampire/ML-Project

## 📌 Next Session
- **W02D06** — Mini Project: Classification
- **Preview**: Semua yang sudah dipelajari minggu ini — Linear Regression,
  Logistic Regression, Decision Tree, Cross-validation, Bias-Variance —
  digabungkan dalam satu project kecil. Kamu akan memilih dataset,
  melatih beberapa model, mengevaluasi dengan CV, mendiagnosa dengan
  learning curve, dan memilih model terbaik dengan justifikasi yang benar.