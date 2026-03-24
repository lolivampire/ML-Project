"""
W03D02 — Missing Value Strategies Test Section
Diagnosis missing value, bangun 3 pipeline:
Pipeline A — SimpleImputer(strategy="mean") untuk kolom numerik, SimpleImputer(strategy="most_frequent") untuk kolom kategorikal.
Pipeline B — SimpleImputer(strategy="median") untuk kolom numerik, sama untuk kategorikal.
Pipeline C — Tambahkan MissingIndicator sebelum imputasi menggunakan FeatureUnion.
Evaluasi ketiga pipeline dengan cross_val_score, 5-fold, metric accuracy.
dengan dataset Titanic dari seaborn.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# 1. LOAD DATA
def load_titanic_data():
    df = sns.load_dataset('titanic')
    # Sesuai permintaan: Fitur spesifik
    features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    target = 'survived'
    return df[features], df[target]

# 2. DIAGNOSIS
def diagnose_missing(df):
    print("── Diagnosis Dataset ──")
    stats = pd.DataFrame({
        'Missing': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2),
        'Dtype': df.dtypes
    })
    print(stats, "\n")

# 3. PIPELINE BUILDER
def create_pipeline(num_col, cat_col, num_strategy="mean", use_indicator=False):
    # Pipeline Numerik
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=num_strategy)),
        ("scaler", StandardScaler())
    ])
    
    # Pipeline Kategorikal
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # sparse_output=False penting agar bisa digabung di FeatureUnion
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_col),
        ("cat", cat_pipeline, cat_col)
    ])

    # Logika FeatureUnion untuk Pipeline C
    if use_indicator:
        # Menggabungkan data yang sudah diproses DAN indikator missing
        final_features = FeatureUnion([
            ("processed_data", preprocessor), # output: sudah imputed, scaled, encoded
            ("missing_ind", MissingIndicator()) # input: X ASLI dengan NaN
        ])
    else:
        final_features = preprocessor

    return Pipeline([
        ("features", final_features),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main() -> None:
    # A. Load & Diagnosis
    X, y = load_titanic_data()
    diagnose_missing(X)

    #Kolom mana yang punya missing values?
    #dari output yang dihasilkan, kolom 'age', 'embarked', 'deck', dan 'embark_town' punya missing value
    # apakah missing di kolom age itu MCAR, MAR, atau MNAR? Kenapa?
    # jawaban: ada 177 nilai usia hilang (sekitar 20% data), kemungkinan besar data missing pada age adalah MAR, karena kemungkinan petugas lebih teliti mencatat usia pada penumpang kelas 1 daripada kelas bawahnya.
    #apakah missing di kolom deck itu MCAR, MAR, atau MNAR? Kenapa?
    # jawaban: pada kolom cabin/deck 70% data di kolom ini hilang. kemungkinan besar data cabin missing adalah MNAR, Data kabin hilang karena penumpang tersebut memang tidak punya kabin (mereka tidur di ruang umum atau kelas bawah).

    num_col = ['pclass','age', 'sibsp', 'parch', 'fare']
    cat_col = ['sex', 'embarked']

    # B. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # C. Build Strategies
    strategies = {
        "A: Mean Impute": create_pipeline(num_col, cat_col, "mean"),
        "B: Median Impute": create_pipeline(num_col, cat_col, "median"),
        "C: Median + Indicator": create_pipeline(num_col, cat_col, "median", use_indicator=True)
    }

    # D. Evaluation
    print("── Titanic Survival Prediction Performance ──")
    print(f"{'Strategy':<25} | {'Mean CV Accuracy':<15}")
    print("-" * 45)

    for name, pipe in strategies.items():
        # Gunakan X_train untuk menghindari data leakage
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy')
        print(f"{name:<25} | {scores.mean():.4f} ± {scores.std():.4f}")
        # Tampilkan std deviation — ini penting untuk tahu stabilitas mode

    #REFLECTION
    #Mengapa fit imputer harus di X_train saja?
    #karena fit itu seperti proses belajar, fit dilakukan pada X_train seperti belajar dengan materi, jika fit dilakukan dengan X_Test juga itu seperti belajar dengan materi + soal, nantinya imputer akan tau informasi yang ada di data latih dan digunakan untuk mengisi kekosongan pada saat memproses X_test
    # Apa yang terjadi kalau kamu fit di seluruh data sebelum split?
    # -> akan terjadi data leakage, Data X_test (yang seharusnya menjadi data rahasia/masa depan) ikut mempengaruhi perhitungan rata-rata dan standar deviasi pada data latih, sehingga dapat mempengaruhi performa model. ini adalah kesalahan besar karena seperti mencotek pada saat ujian.
    # Kolom deck (77% Missing): Impute atau Drop?
    # -> kedua argumen memiliki kelebihan dan kekurangan masing masing, jika DROP data yang tersisa mungkin tidak akan mewakili populasi namun jika Impute kolom ini adalah kolom penting karena menunjukkan nomor kabin punya peluang selamat lebih tinggi. lebih baik jika Isi semua NaN dengan kategori baru bernama "Unknown" atau "No_Cabin" atau lakukan categorical_encoding agar model tetap bisa menggunakan informasi "Tidak punya kabin".

if __name__ == "__main__":
    main()