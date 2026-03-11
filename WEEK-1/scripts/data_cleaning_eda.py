data = {
    'nama':   ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Bob', 'Frank', None],
    'umur':   [25, 30, None, 22, 28, 30, 999, 26],
    'gaji':   [50000, 60000, 55000, None, 62000, 60000, 58000, 51000],
    'kota':   ['Jakarta', 'Bandung', 'Jakarta', 'Surabaya', None, 'Bandung', 'Jakarta', 'Medan'],
    'skor':   [85, 90, 78, 88, 92, 90, 45, 83]
}

import pandas as pd 

df = pd.DataFrame(data)

#tampilkan shape, dtypes, isnull().sum()
def show_basic_info(df):
    print("Shape:", df.shape)
    print("Dtypes:", df.dtypes)
    print("Null values:", df.isnull().sum())

#Hapus baris duplikat
def remove_duplicates(df):
    print("Jumlah baris duplikat:", df.duplicated().sum())
    print(df[df.duplicated()]) 
    df = df.drop_duplicates()
    print("Jumlah baris duplikat setelah drop:", df.duplicated().sum())
    return df

#Deteksi outlier di kolom umur menggunakan IQR method — cetak jumlahnya
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline data cleaning: duplicates → outlier → imputation."""
    df = df.copy()  # work on a copy to avoid modifying original dataframe

    # remove duplicates via helper
    before = len(df)
    df = remove_duplicates(df)
    print(f"[clean] Duplicates removed: {before - len(df)}")

    # outlier detection & removal sebelum imputation
    Q1 = df['umur'].quantile(0.25)
    Q3 = df['umur'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['umur'] < lower_bound) | (df['umur'] > upper_bound)]
    print(f"Jumlah outlier: {len(outliers)}")

    df = df[(df['umur'] >= lower_bound) & (df['umur'] <= upper_bound)]
    print(f"Jumlah data setelah clean: {len(df)}")

    # imputation tanpa loop eksplisit
    num_cols = df.select_dtypes(include='number').columns
    if not df[num_cols].empty:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    obj_cols = df.select_dtypes(include='object').columns
    if not df[obj_cols].empty:
        df[obj_cols] = df[obj_cols].fillna('Unknown')

    return df

print("============================generate_eda_report(df)==================================")
def generate_eda_report(df, label: str | None = None):
    # work on a copy so the original dataframe remains untouched
    df = df.copy()
    if label is not None:
        print(f"=== EDA Report {label} ===")
    print("Shape:", df.shape)
    print("\nJumlah missing values per kolom:")
    print(df.isnull().sum())
    print("\nStatistik deskriptif:")
    print(df.describe())
    print("\nValue counts untuk kolom kota:")
    print(df['kota'].value_counts())


# ── MAIN ──────────────────────────────────────────────────────
def main() -> None:
    show_basic_info(df)

    remove_duplicates(df)

    df_raw = pd.DataFrame(data)  # buat ulang dataframe untuk EDA sebelum cleaning

    generate_eda_report(df_raw, label="Sebelum Cleaning")

    df_clean = clean_dataframe(df_raw)

    generate_eda_report(df_clean, label="Setelah Cleaning")


if __name__ == "__main__":
    main()