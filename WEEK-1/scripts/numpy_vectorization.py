import numpy as np
import time

# ── CONSTANTS ──────────────────────────────────────────────────────

nilai_list = list(range(1, 1000001))  # Membuat daftar angka dari 1 hingga 1 juta
nilai_arr = np.arange(1, 1000001)  # Membuat array NumPy dari 1 hingga 1 juta
suhu = np.array([28, 31, 27, 35, 33, 29, 30, 26, 34, 32, 28, 31, 36, 29])
jarak_motor = np.array([12.5, 8.3, 20.1, 5.0, 17.8, 9.4, 15.6])

# ── FUNCTIONS ──────────────────────────────────────────────────────


# Min-max scaling ke range [0, 1]
def normalize_jarak(arr: np.ndarray) -> np.ndarray:
    """Normalize array values to range [0, 1] using min-max scaling."""
    min_jarak = np.min(arr)
    max_jarak = np.max(arr)
    return (arr - min_jarak) / (max_jarak - min_jarak)

# TASK 1: Verifikasi bahwa operasi NumPy lebih cepat daripada loop manual untuk operasi pada array besar.
def task1_verifikasi_kecepatan():
    """Verify that NumPy operations are faster than manual loops for large arrays."""
    #PAKAI MANUAL
    start = time.time()
    
    hasil = []
    for x in nilai_list:
        hasil.append((x + 10)* x)
    end = time.time()
    print(f"Manual sum of squares elapsed {end - start:.6f} seconds")

    #pakai numpy
    start = time.time()
    hasil_numpy = (nilai_arr + 10) * nilai_arr
    end = time.time()
    print(f"NumPy sum of squares elapsed {end - start:.6f} seconds")

# TASK 2: Hitung Suhu
def task2_hitung_suhu():
    """Calculate and display average temperature and count of hot days."""
    suhu_panas = suhu[suhu > 31]
    rata_rata_suhu_panas = np.mean(suhu_panas)
    banyak_hari_panas = len(suhu_panas)

    print(f"Suhu rata-rata: {rata_rata_suhu_panas} °C, Hari dimana suhu lebih dari 31°C: {banyak_hari_panas} hari")

# TASK 3: Normalize jarak motor
def task3_normalize_jarak():
    """Normalize motor distances and display before and after values."""
    print("Jarak motor sebelum normalisasi:", jarak_motor)
    print("Jarak motor setelah normalisasi:", normalize_jarak(jarak_motor))

# ── MAIN ──────────────────────────────────────────────────────
def main() -> None:
    """Main function to execute all tasks."""
    task1_verifikasi_kecepatan()
    task2_hitung_suhu()
    task3_normalize_jarak()

if __name__ == "__main__":
    main()