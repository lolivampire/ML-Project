"""
pandas_basics.py
Week 01 - Day 03: Pandas DataFrame Basics
"""

import pandas as pd
import numpy as np

# ── DATA ──────────────────────────────────────────

raw_data = {
    "nama": ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"],
    "departemen": ["Engineering", "Marketing", "Engineering",
                   "Marketing", "Engineering", "HR"],
    "gaji": [8500, 6200, 9100, 5800, 7750, None],
    "tahun_kerja": [3, 5, 2, 7, 4, None],
    "performa": [88.0, 72.5, 95.0, 68.0, 81.5, 77.0]
}

# ── TASK 1 ────────────────────────────────────────
# Buat DataFrame dari raw_data
# Print: shape, dtypes, dan info()

# ── TASK 2 ────────────────────────────────────────
# Deteksi missing values
# Print: jumlah missing per kolom

# ── TASK 3 ────────────────────────────────────────
# Handle missing values:
# - gaji → fill dengan rata-rata gaji
# - tahun_kerja → fill dengan median tahun_kerja
# Verifikasi: tidak ada missing values setelah fill

# ── TASK 4 ────────────────────────────────────────
# Filter: tampilkan karyawan dari departemen Engineering
# dengan performa > 85

# ── TASK 5 ────────────────────────────────────────
# GroupBy: rata-rata gaji per departemen
# Sort hasilnya dari tertinggi ke terendah

# ── TASK 6 ────────────────────────────────────────
# Tambah kolom baru: "gaji_normalized"
# Gunakan min-max normalization pada kolom gaji
# (gunakan cara vectorized, bukan loop)
  
# ── MAIN ──────────────────────────────────────────
def main() -> None:
    # Jalankan semua task di sini
    print("======================TASK 1============================")
    data = pd.DataFrame(raw_data)
    print("Shape:", data.shape)
    print("Data types:\n", data.dtypes)
    print("Info:", data.info())

    print("=====================TASK 2=============================")
    missing_values = data.isnull().sum()
    print("Jumlah missing values per kolom:\n", missing_values)

    print("=====================TASK 3=============================")
    data["gaji"] = data["gaji"].fillna(data["gaji"].mean())
    data["tahun_kerja"] = data["tahun_kerja"].fillna(data["tahun_kerja"].median())
    # periksa apakah semua nilai sudah terisi
    missing_after = data.isnull().sum()
    if (missing_after == 0).all():
        print("Data null kosong setelah fill")
    else:
        print("Jumlah missing values setelah fill:\n", missing_after)

    print("=====================TASK 4=============================")
    engineering_top = data[(data["departemen"] == "Engineering") & (data["performa"] > 85)]
    print("Karyawan dari Engineering dengan performa > 85:")
    print(engineering_top)

    print("=====================TASK 5=============================")
    avg_salary_by_dept = data.groupby("departemen")["gaji"].mean().sort_values(ascending=False)
    print("Rata-rata gaji per departemen (tertinggi ke terendah):")
    print(avg_salary_by_dept)

    print("=====================TASK 6=============================")
    def min_max_normalize(series: pd.Series) -> pd.Series:
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val)

    data["gaji_normalized"] = min_max_normalize(data["gaji"])
    print("DataFrame dengan kolom gaji_normalized:") 
    print(data.head(10))

    pass

if __name__ == "__main__":
    main()