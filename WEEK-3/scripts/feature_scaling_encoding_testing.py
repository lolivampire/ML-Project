import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ============================================================================
# CONSTANTS
# ============================================================================

DATA = {
    'tinggi_cm':  [158, 172, 165, 180, 155, 190, 163, 175],
    'berat_kg':   [52, 70, 61, 85, 48, 95, 58, 78],
    'pendapatan': [4_500_000, 12_000_000, 7_200_000, 18_000_000,
                   3_800_000, 25_000_000, 5_500_000, 15_000_000],
    'kota':       ['Surabaya', 'Jakarta', 'Bandung', 'Jakarta',
                   'Surabaya', 'Jakarta', 'Bandung', 'Surabaya'],
    'pendidikan': ['SMA', 'S1', 'S2', 'S1', 'SMA', 'S2', 'S1', 'S2'],
    'label':      [0, 1, 0, 1, 0, 1, 0, 1]
}

# Urutan ordinal yang nyata: SMA < S1 < S2
# LabelEncoder aman karena urutan ini benar-benar bermakna di dunia nyata
ORDINAL_MAPPING = {'SMA': 0, 'S1': 1, 'S2': 2}

NUMERIC_STD   = ['tinggi_cm', 'berat_kg']  # distribusi mendekati normal → StandardScaler
NUMERIC_MM    = ['pendapatan']              # skala besar, tidak perlu asumsi distribusi → MinMaxScaler
NOMINAL       = ['kota']                   # tidak ada urutan → OneHotEncoder
ORDINAL       = 'pendidikan'               # ada urutan nyata → encode manual sebelum split
TARGET        = 'label'

TEST_SIZE     = 0.25
RANDOM_STATE  = 42

# ============================================================================
# FUNCTIONS
# ============================================================================

def encode_ordinal(df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
    """
    Encode kolom ordinal dengan mapping eksplisit.

    Dilakukan SEBELUM split agar aman — ini bukan fitting statistik,
    hanya penggantian label teks → angka yang sudah kita tentukan sendiri.
    Tidak ada informasi dari distribusi data yang bocor.
    """
    df = df.copy()
    df[column] = df[column].map(mapping)
    return df


def print_scaling_stats(pipeline: Pipeline, X_train: pd.DataFrame) -> None:
    """
    Ambil statistik scaler langsung dari pipeline yang sudah di-fit.
    Lebih bersih dari fit ulang scaler terpisah.
    """
    preprocessor = pipeline.named_steps['preprocessor']

    # StandardScaler — ambil mean_ dan scale_ dari pipeline
    std_scaler = preprocessor.named_transformers_['standard']
    print("\n=== StandardScaler (fit dari training data) ===")
    for name, mean, std in zip(NUMERIC_STD, std_scaler.mean_, std_scaler.scale_):
        print(f"  {name:12} → mean={mean:.2f}, std={std:.2f}")

    # MinMaxScaler — ambil data_min_ dan data_max_ dari pipeline
    mm_scaler = preprocessor.named_transformers_['minmax']
    print("\n=== MinMaxScaler (fit dari training data) ===")
    for name, mn, mx in zip(NUMERIC_MM, mm_scaler.data_min_, mm_scaler.data_max_):
        print(f"  {name:12} → min={mn:,.0f}, max={mx:,.0f}")

    # OneHotEncoder — tampilkan kategori yang dikenali
    ohe = preprocessor.named_transformers_['onehot']
    print("\n=== OneHotEncoder — kategori yang dikenali (dari training) ===")
    for feature, categories in zip(NOMINAL, ohe.categories_):
        print(f"  {feature}: {list(categories)} — drop: '{categories[0]}'")


def build_pipeline() -> Pipeline:
    """
    Buat pipeline preprocessing + classifier.

    ColumnTransformer memproses kolom berbeda secara paralel:
    - standard : StandardScaler  → tinggi_cm, berat_kg
    - minmax   : MinMaxScaler    → pendapatan
    - onehot   : OneHotEncoder   → kota
    remainder='passthrough' agar kolom ordinal (pendidikan) ikut masuk tanpa diubah.
    """
    preprocessor = ColumnTransformer(transformers=[
        ('standard', StandardScaler(), NUMERIC_STD),
        ('minmax',   MinMaxScaler(),   NUMERIC_MM),
        ('onehot',   OneHotEncoder(drop='first', sparse_output=False), NOMINAL),
    ], remainder='passthrough')

    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier',   LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ])


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    df = pd.DataFrame(DATA)

    # Encode ordinal SEBELUM split — aman karena tidak melibatkan statistik data
    df = encode_ordinal(df, ORDINAL, ORDINAL_MAPPING)

    features = NUMERIC_STD + NUMERIC_MM + NOMINAL + [ORDINAL]
    X = df[features]
    y = df[TARGET]

    # Split: stratify=y wajib untuk dataset kecil dengan class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} baris | Test: {X_test.shape[0]} baris")

    # Fit pipeline — scaler dan encoder belajar HANYA dari X_train
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Statistik diambil dari scaler yang sudah fit di dalam pipeline
    print_scaling_stats(pipeline, X_train)

    # Evaluasi
    score = pipeline.score(X_test, y_test)
    print(f"\n{'='*45}")
    print(f"Accuracy pada test set : {score:.4f}")
    print(f"{'='*45}")

    # kenapa kamu pilih scaler yang berbeda untuk pendapatan vs tinggi/berat?
    # tinggi_cm dan berat_kg mendekati distribusi normal → StandardScaler cocok. pendapatan memiliki skala yang jauh lebih besar tapi tidak ada asumsi distribusi normal — MinMaxScaler mengkompresnya ke [0,1] tanpa asumsi apapun tentang bentuk distribusinya.
    # kenapa LabelEncoder aman dipakai untuk pendidikan tapi tidak untuk kota?
    # karena kota adalah nominal — tidak ada urutan bermakna antara Jakarta, Bandung, Surabaya. Kalau LabelEncoder, model akan asumsikan Surabaya (2) > Jakarta (1) — relasi yang tidak ada di dunia nyata dan akan menyesatkan model.

if __name__ == "__main__":
    main()