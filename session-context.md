# 📍 Session Context — ML Engineer Journey

## Status Terakhir
- **Week**: 01
- **Day**: 05
- **Phase**: 1 — ML Fundamentals
- **Tanggal**: 13 Mar 2026

## ✅ Yang Sudah Selesai
- Setup project structure via terminal (bukan File Explorer)
- Setup virtual environment (venv) + install dependencies
- Buat dan isi .gitignore, README.md, session_context.md
- Tulis hello_clean.py dengan Python best practices
- Push pertama ke GitHub berhasil
- Numpy vectorization: loop vs numpy, boolean indexing, normalization
- Pandas DataFrame basics: struktur, inspect, missing values, filter, groupby, normalization
- Data Cleaning & EDA: duplicates, outlier IQR, imputation via select_dtypes, generate_eda_report
- Matplotlib & Seaborn: Figure/Axes hierarchy, 5 chart inti, fungsi reusable, savefig pipeline

## 🧠 Yang Sudah Dipahami
- Kenapa terminal lebih penting dari GUI untuk engineer
- Virtual environment: isolasi dependency antar project
- Type hints: membuat kontrak fungsi yang jelas
- Docstring: dokumentasi yang hidup di dalam kode
- Single responsibility: satu fungsi, satu tanggung jawab
- try/except: harus membungkus kode yang berpotensi error
- if __name__ == "__main__": agar script bisa diimport tanpa autorun
- np.median() return numpy.float64 — perlu di-wrap float()
- Python loop lambat karena type-checking tiap elemen di runtime
- Numpy array = fixed-type contiguous memory block, operasi di level C
- Vectorization: operasi ke seluruh array sekaligus tanpa loop
- Boolean indexing: suhu[suhu > 31] mengembalikan nilai, np.sum() mengembalikan hitungan
- Fungsi yang hardcoded ke variabel luar tidak bisa ditest secara independen
- Normalisasi min-max: (x - min) / (max - min)
- DataFrame = tabel, Series = satu kolom
- .loc inklusif di kedua batas, .iloc eksklusif di batas akhir
- df.info() dipanggil langsung, bukan di-wrap print()
- fillna() dan dropna() tidak mengubah DataFrame asli tanpa inplace=True atau re-assign
- Boolean indexing di Pandas pakai & bukan and, tiap kondisi dibungkus ()
- Operator precedence: alasan tanda kurung wajib di boolean indexing
- groupby().mean().sort_values() untuk agregasi per grup
- Global mean fill bisa memengaruhi agregasi per grup
- Pipeline cleaning harus berurutan: inspect → drop duplicates → outlier → imputation → fix dtypes
- df.copy() wajib di awal fungsi cleaning — tanpa ini modifikasi bisa merembet ke df asli via view
- select_dtypes untuk imputation massal — scalable untuk banyak kolom
- isnull().sum() = jumlah absolut, isnull().mean()*100 = persentase (lebih informatif saat beda skala kolom)
- median lebih robust dari mean saat distribusi skewed atau ada outlier ekstrim
- IQR method: Q1 - 1.5*IQR dan Q3 + 1.5*IQR sebagai batas outlier (konvensi Tukey)
- Boolean filter outlier (|) vs data bersih (&) — keduanya komplemen
- Return value fungsi harus di-assign: df = func(df), bukan func(df) saja
- EDA sebelum cleaning harus menggunakan df_raw yang belum disentuh sama sekali
- Figure = kanvas, Axes = panel grafik di dalam kanvas — satu Figure bisa banyak Axes
- fig, ax = plt.subplots() adalah cara yang benar — bukan plt.plot() langsung
- Seaborn duduk di atas Matplotlib: setiap plot Seaborn adalah objek Matplotlib di bawahnya
- Selalu pass ax=ax ke fungsi Seaborn — tanpa ini Seaborn buat axes baru sendiri
- savefig() HARUS sebelum show() — jika dibalik, file yang tersimpan akan kosong
- Histogram: lihat shape distribusi (normal/skewed/bimodal) satu variabel
- Boxplot: deteksi outlier + perbandingan distribusi antar grup secara ringkas
- Scatter plot: lihat hubungan dua variabel numerik, hue= untuk color grouping
- Bar chart: bandingkan agregat (mean/median) antar kategori, estimator= bisa diganti
- Heatmap korelasi: lihat korelasi semua variabel numerik sekaligus, vmin=-1 vmax=1 untuk scaling proper
- Fungsi plot yang generik: parameter kolom/x/y harus dipakai di body — bukan hardcoded
- os.makedirs(output_dir, exist_ok=True) di setiap fungsi plot — jangan asumsi folder sudah ada

## ⚠️ Yang Masih Blur
-

## 📁 Output Files
- `week-01/scripts/hello_clean.py` ✅
- `week-01/scripts/numpy_vectorization.py` ✅
- `week-01/scripts/pandas_basics.py` ✅
- `week-01/scripts/data_cleaning_eda.py` ✅
- `week-01/scripts/visualization.py` ✅
- `week-01/outputs/*.png` ✅
- `requirements.txt` ✅
- `.gitignore` ✅
- `README.md` (root) ✅
- `week-01/README.md` ✅
- GitHub: https://github.com/lolivampire/ML-Project

## ❓ Pertanyaan Pending
-

## 📌 Next Session
- **Week**: 01
- **Day**: 06
- **Topik**: Review + Polish Notebook
- **Preview**: Sabtu adalah hari output — bukan belajar topik baru.
  Rapikan semua notebook dan script minggu ini,
  tambahkan markdown explanation di setiap section,
  dan push final commit W01 ke GitHub.