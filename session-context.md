# 📍 Session Context — ML Engineer Journey

## Status Terakhir
- **Week**: 01
- **Day**: 03
- **Phase**: 1 — ML Fundamentals
- **Tanggal**: 10 Mar 2026

## ✅ Yang Sudah Selesai
- Setup project structure via terminal (bukan File Explorer)
- Setup virtual environment (venv) + install dependencies
- Buat dan isi .gitignore, README.md, session_context.md
- Tulis hello_clean.py dengan Python best practices
- Push pertama ke GitHub berhasil
- Numpy vectorization: loop vs numpy, boolean indexing, normalization
- Pandas DataFrame basics: struktur, inspect, missing values, filter, groupby, normalization

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
- fillna() dan dropna() tidak mengubah DataFrame asli tanpa inplace=True
- Boolean indexing di Pandas pakai & bukan and, tiap kondisi dibungkus ()
- Operator precedence: alasan tanda kurung wajib di boolean indexing
- groupby().mean().sort_values() untuk agregasi per grup
- Global mean fill bisa memengaruhi agregasi per grup

## ⚠️ Yang Masih Blur
-

## 📁 Output Files
- `week-01/scripts/hello_clean.py` ✅
- `week-01/scripts/numpy_vectorization.py` ✅
- `week-01/scripts/pandas_basics.py` ✅
- `requirements.txt` ✅
- `.gitignore` ✅
- `README.md` (root) ✅
- `week-01/README.md` ✅
- GitHub: https://github.com/lolivampire/ML-Project

## ❓ Pertanyaan Pending
-

## 📌 Next Session
- **Week**: 01
- **Day**: 04
- **Topik**: Pandas — Data Cleaning & EDA
- **Preview**: Data kotor adalah kenyataan. Missing values,
  duplicates, tipe data yang salah, outlier — semua ini
  harus dibersihkan sebelum model apapun bisa dijalankan.
  EDA adalah cara kita memahami data sebelum memodelkannya.