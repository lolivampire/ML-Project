import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Dataset sederhana: usia dan gaji
data = {
    'usia': [22, 35, 28, 45, 30, 52, 24, 41],
    'gaji': [3_000_000, 8_500_000, 5_200_000, 12_000_000,
             6_000_000, 15_000_000, 3_500_000, 10_000_000],
    'churn': [1, 0, 1, 0, 0, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df[['usia', 'gaji']]
y = df['churn']

# WAJIB: split dulu sebelum scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ── StandardScaler ──────────────────────────────────────────
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)  # fit + transform sekaligus di train
X_test_std  = std_scaler.transform(X_test)        # HANYA transform di test — tanpa fit

# ── MinMaxScaler ────────────────────────────────────────────
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm  = mm_scaler.transform(X_test)          # sama — jangan fit ulang

# Lihat hasilnya
print("=== StandardScaler (train) ===")
print(pd.DataFrame(X_train_std, columns=['usia', 'gaji']).round(3))

print("\n=== MinMaxScaler (train) ===")
print(pd.DataFrame(X_train_mm, columns=['usia', 'gaji']).round(3))

# ── Feature Encoding ────────────────────────────────────────────

le = LabelEncoder()
kota = ['Jakarta', 'Bandung', 'Surabaya', 'Jakarta', 'Bandung']
encoded = le.fit_transform(kota)
print(encoded)        # [1 0 2 1 0]
print(le.classes_)    # ['Bandung' 'Jakarta' 'Surabaya']

# ── OneHotEncoder — kolom terpisah per kategori ────────────────────────────────────────────

ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' hindari dummy trap
kota = pd.DataFrame({'kota': ['Jakarta', 'Bandung', 'Surabaya', 'Jakarta', 'Bandung']})

encoded = ohe.fit_transform(kota[['kota']])
kolom   = ohe.get_feature_names_out(['kota'])
df_encoded = pd.DataFrame(encoded, columns=kolom)
print(df_encoded)