from unicodedata import normalize

import numpy as np
import time

# TASK 1: Verifikasi bahwa operasi NumPy lebih cepat daripada loop manual untuk operasi pada array besar.

#PAKAI MANUAL
start = time.time()
nilai = list(range(1, 1000001))  # Membuat daftar angka dari 1 hingga 1 juta
hasil = []
for x in nilai:
    hasil.append((x + 10)* x)
end = time.time()
print(f"Manual sum of squares elapsed {end - start:.6f} seconds")

#pakai numpy
start = time.time()
np_arr = np.array(nilai)
hasil_numpy = (np_arr + 10) * np_arr
end = time.time()
print(f"NumPy sum of squares elapsed {end - start:.6f} seconds")

# TASK 2: Hitung Suhu
suhu = np.array([28, 31, 27, 35, 33, 29, 30, 26, 34, 32, 28, 31, 36, 29])
suhu_panas = suhu[suhu > 31]
rata_rata_suhupanas = np.mean(suhu_panas)
banyak_hari_panas = len(suhu_panas)

print(f"Suhu rata-rata: {rata_rata_suhupanas} °C, Hari dimana suhu lebih dari 31°C: {banyak_hari_panas} hari")

# TASK 3: Normalize jarak motor
jarak_motor = np.array([12.5, 8.3, 20.1, 5.0, 17.8, 9.4, 15.6])
# Min-max scaling ke range [0, 1]
def normalize_jarak(arr: np.ndarray) -> np.ndarray:
    min_jarak = np.min(arr)
    max_jarak = np.max(arr)
    return (arr - min_jarak) / (max_jarak - min_jarak)

print("Jarak motor sebelum normalisasi:", jarak_motor)
print("Jarak motor setelah normalisasi:", normalize_jarak(jarak_motor))