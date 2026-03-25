"""
Titanic Feature Engineering & Selection Pipeline.
------------------------------------------------
Script ini melakukan preprocessing, penciptaan fitur baru (feature creation), 
dan seleksi fitur menggunakan SelectKBest serta Random Forest Importance.
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Konfigurasi Global
warnings.filterwarnings('ignore')
plt.style.use('ggplot') # Opsional: Agar visualisasi lebih estetik
output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs/plots')
os.makedirs(output_dir, exist_ok=True)

def load_and_preprocess_base(target='survived'):
    """Memuat dataset Titanic dan melakukan encoding dasar."""
    df = sns.load_dataset('titanic')
    
    # Memilih kolom relevan
    cols = [target, 'pclass', 'age', 'sibsp', 'parch', 'fare', 'sex', 'embarked']
    df = df[cols].copy()

    # Encoding Kategorik: Sex & Embarked
    df['sex'] = (df['sex'] == 'male').astype(int)  # 1: Male, 0: Female
    df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    return df

def create_features(df):
    """
    Melakukan Feature Engineering untuk menambah informasi prediktif.
    
    Fitur baru:
    - family_size: Total anggota keluarga di kapal.
    - fare_per_person: Rasio tarif per orang.
    - is_alone: Flag biner jika penumpang bepergian sendiri.
    - age_group: Binning usia menjadi kategori numerik.
    """
    # 1. Family Interaction
    df['family_size'] = df['sibsp'] + df['parch'] + 1
    
    # 2. Financial Ratio (Prevent division by zero)
    df['fare_per_person'] = df['fare'] / df['family_size'].clip(lower=1)
    
    # 3. Binary Flags
    df['is_alone'] = (df['family_size'] == 1).astype(int)
    
    # 4. Age Binning
    # Menggunakan bins: [Child, Teen, Adult, Senior]
    df['age_group'] = pd.cut(
        df['age'], 
        bins=[0, 12, 18, 60, 100], 
        labels=[0, 1, 2, 3]
    ).astype('float64')
    
    return df

def perform_feature_selection(X_train, y_train, k=6):
    """Menyeleksi fitur terbaik menggunakan statistik ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    return selector

def visualize_results(feature_scores, rf_importances):
    """Membandingkan hasil SelectKBest dan Random Forest secara visual."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: SelectKBest
    scores_sorted = feature_scores.sort_values('score')
    colors = ['#3498db' if s else '#bdc3c7' for s in scores_sorted['selected']]
    axes[0].barh(scores_sorted['feature'], scores_sorted['score'], color=colors)
    axes[0].set_title('SelectKBest: ANOVA F-Scores', fontsize=12)
    axes[0].set_xlabel('Score')

    # Plot 2: Random Forest
    rf_importances.sort_values().plot(kind='barh', ax=axes[1], color='#1abc9c')
    axes[1].set_title('Random Forest: Feature Importance', fontsize=12)
    axes[1].set_xlabel('Importance Value')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_creation_selection.png'), dpi=300)
    plt.show()

def main():
    # --- STEP 1: Load & Create Features ---
    df = load_and_preprocess_base()
    df = create_features(df)
    
    # --- STEP 2: Train-Test Split ---
    X = df.drop('survived', axis=1)
    y = df['survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- STEP 3: Imputation (Silently handling missing values) ---
    imputer = SimpleImputer(strategy='median')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # --- STEP 4: Feature Selection (SelectKBest) ---
    selector = perform_feature_selection(X_train_imp, y_train)
    X_train_sel = selector.transform(X_train_imp)  # eksplisit
    X_test_sel  = selector.transform(X_test_imp)   # eksplisit
    
    feature_scores = pd.DataFrame({
        'feature': X_train.columns,
        'score': selector.scores_,
        'selected': selector.get_support()
    })
    print(feature_scores)

    # --- STEP 5: Feature Importance (Random Forest) ---
    model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf.fit(X_train_imp, y_train)
    rf_importances = pd.Series(model_rf.feature_importances_, index=X_train.columns)
    

    # --- FINAL OUTPUT ---
    print("\n[INFO] Top Features (SelectKBest):", selector.k, "features")
    print("\n--- Feature Ranking ---")
    print(feature_scores.sort_values('score', ascending=False).to_string(index=False))
    print("\n--- Feature Importance ---")
    print(rf_importances.sort_values(ascending=False).to_string())
    
    # visualize_results(feature_scores, rf_importances)

    # REFLECTION
    # 1. Fitur baru mana yang masuk top-3 SelectKBest? Mengapa masuk akal?
    # JAWAB: Fitur baru yang terpilih: fare_per_person (rank 4) dan is_alone (rank 5) fare_per_person masuk akal — dia "membersihkan" fare mentah dari efek ukuran keluarga, jadi lebih representatif dari kemampuan ekonomi individu is_alone masuk akal — perjalanan sendirian vs bersama keluarga ternyata punya sinyal survival yang cukup kuat
    # 2. Apakah urutan top-3 RF importance sama dengan SelectKBest? Bedanya apa?
    # JAWAB: berbeda, karena pendekatanya pun berbeda SelectKBest menggunakan metode ANOVA F-test untuk menentukan fitur yang paling penting, sedangkan RF importance menggunakan metode Random Forest untuk menentukan fitur yang paling penting.
    # 3. Fitur mana yang menurutmu paling berguna — dan kenapa?
    # JAWAB: SelectKBest menilai age secara sendirian terhadap survived — hasilnya lemah (score 3.8) karena hubungan age dan survival memang tidak sederhana secara linear. Anak kecil diselamatkan duluan, tapi di luar itu tidak ada pola usia yang jelas kalau dilihat sendirian. RF melihat age dalam konteks fitur lain — dan menemukan bahwa kombinasi age + sex + pclass sangat menentukan. Anak perempuan kelas 1 hampir pasti selamat. Pria dewasa kelas 3 hampir pasti tidak. Interaksi itu tidak tertangkap SelectKBest. Implikasi praktis: Kalau kamu buang age berdasarkan SelectKBest saja, model RF kamu akan kehilangan fitur yang sebetulnya penting. Inilah kenapa untuk feature selection final, lebih baik pakai feature importance dari model yang akan dipakai, bukan hanya SelectKBest.

if __name__ == "__main__":
    main()