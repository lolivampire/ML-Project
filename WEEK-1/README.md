# Week 01 — Python & Data Foundations

## Topik yang Dipelajari

1. **Python Best Practices**: Penerapan type hints, docstring, single responsibility principle, exception handling dengan try/except, dan konvensi penamaan untuk kode yang bersih dan maintainable.
2. **NumPy Vektorisasi**: Demonstrasi kecepatan operasi vektorisasi NumPy dibanding loop manual, operasi array untuk analisis data seperti filtering suhu, dan teknik normalisasi min-max.
3. **Pandas DataFrame Basics**: Pembuatan DataFrame dari dictionary, handling missing values (fill dengan mean/median), filtering data, groupby untuk agregasi, dan normalisasi kolom secara vektorisasi.
4. **Data Cleaning & EDA**: Teknik pembersihan data termasuk penghapusan duplikat, deteksi outlier dengan metode IQR, imputasi missing values, serta analisis eksplorasi dasar seperti shape, dtypes, dan statistik deskriptif.
5. **Visualisasi Data**: Pembuatan plot menggunakan Matplotlib dan Seaborn, termasuk histogram dengan KDE, boxplot, scatter plot dengan hue, serta penyimpanan plot ke file dengan resolusi tinggi.

## Output Files

- `hello_clean.py`: Script sederhana untuk memahami struktur kode Python yang bersih.
- `numpy_vectorization.py`: Demonstrasi kecepatan vektorisasi NumPy dibandingkan loop manual, serta operasi pada array suhu dan normalisasi jarak.
- `pandas_basic.py`: Pengenalan dasar penggunaan Pandas untuk manipulasi data.
- `data_cleaning_eda.py`: Script untuk membersihkan data dan melakukan analisis eksplorasi awal.
- `visualization.py`: Contoh pembuatan visualisasi data menggunakan Matplotlib.

## Cara Menjalankan

1. Aktivasi virtual environment: Jalankan `& venv\Scripts\Activate.ps1` (untuk Windows PowerShell).
2. Install dependencies: Jalankan `pip install -r requirements.txt` untuk menginstall semua package yang diperlukan.
3. Jalankan script: Gunakan `python WEEK-1/scripts/nama_script.py` untuk menjalankan script tertentu, misalnya `python WEEK-1/scripts/numpy_vectorization.py`.

## Key Learnings

1. Python loop lambat karena tiap elemen harus dicek tipenya di runtime — NumPy menghindari ini dengan menyimpan semua elemen dalam satu blok memori fixed-type dan mengeksekusi operasi di level C, sehingga vektorisasi bisa 10-100x lebih cepat untuk array besar.
2. Pandas DataFrame memungkinkan operasi vektorisasi pada kolom, seperti fillna dengan mean/median atau normalisasi min-max tanpa loop, yang lebih efisien dan readable dibanding list comprehension.
3. Deteksi outlier dengan IQR (Q3-Q1) membantu mengidentifikasi data ekstrem secara statistik, dan imputasi missing values dengan median lebih robust daripada mean untuk data skewed.
4. Matplotlib adalah dasar, dibuat dari nol tapi user memiliki akses penuh modifikasi dan pembuatan, sedangkan Seaborn dibangun di atas Matplotlib dan menyederhanakan pembuatan plot statistik, namun kontrol detail tetap membutuhkan akses ke objek Matplotlib di bawahnya melalui parameter ax=
5. Type hints dan docstring tidak hanya membuat kode lebih readable, tapi juga membantu IDE memberikan autocomplete dan mendeteksi error type lebih awal, mengurangi bug di production.
6. Struktur file sangatlah penting agar kode menjadi clean, reusable, dan mudah dipahami gunakan struktur constants → functions → main `if __name__ == "__main__":` pada bagian main untuk menjalankan kode