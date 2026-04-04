## Status Terakhir
- Week: 04
- Day: 04
- Tanggal: 2 Apr 2026

## ✅ Yang Sudah Selesai (tambahkan)
- W04D04: RandomizedSearch + Early Stopping XGBoost

## 🧠 Yang Sudah Dipahami (tambahkan)
- uniform(loc, scale) — parameter kedua adalah lebar range, bukan upper bound
- randint(a, b) — upper bound eksklusif
- RandomizedSearch unggul di grid besar, bukan dataset kecil
- Early stopping dan CV tidak bisa dipakai bersamaan — konflik arsitektur
- best_iteration menyimpan index iterasi val score terbaik
- predict() setelah early stopping otomatis pakai pohon 0 hingga best_iteration
- Solusi industri: RandomizedSearch (n_estimators fixed) → fit ulang dengan early stopping

## 📁 Output Files (tambahkan)
- week-04/notebooks/w04d04_random_early_stop.ipynb ✅
- W04D04_RandomizedSearch_EarlyStopping.pdf ✅

## 📌 Next Session
- Week: 04
- Day: 05
- Topik: SHAP Values — Explainability Dasar
- Preview: Model yang akurat tapi tidak bisa dijelaskan = black box.
  SHAP membuka black box itu — menunjukkan kontribusi tiap fitur
  terhadap setiap prediksi individual, bukan hanya rata-rata global.