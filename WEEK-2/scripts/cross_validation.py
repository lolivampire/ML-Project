"""
W02D03 — Cross Validation
K-Fold, Stratified K-Fold
"""

import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.datasets import make_classification

# ── CONSTANTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# ── FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

#fungsi 
def cross_validate_manual(X, y, model, k=5):
    """K-Fold CV dari scratch — tanpa sklearn CV utilities."""
    n = len(X)
    fold_size = n // k
    scores = []

    for i in range(k):
        # Tentukan index test fold ke-i
        test_start = i * fold_size
        test_end = test_start + fold_size

        # Split manual
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        X_train = np.concatenate([X[:test_start], X[test_end:]])
        y_train = np.concatenate([y[:test_start], y[test_end:]])

        # Train dan evaluate
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)

        print(f"Fold {i+1}: train={len(X_train)}, test={len(X_test)}, score={score:.4f}")

    return np.array(scores)

# ── MAIN ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#main
def main() -> None: 
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    scores = cross_validate_manual(X, y, model, k=5)

    print(f"\nMean CV Score : {scores.mean():.4f}")
    print(f"Std CV Score  : {scores.std():.4f}")

    #Implementasi K-Fold CV
    model_sklearn = DecisionTreeClassifier(max_depth=3, random_state=2)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model_sklearn, X, y, cv=kf, scoring='accuracy')

    #shuffle=True — penting. Tanpa shuffle, fold dibentuk berdasarkan urutan asli data. Kalau data terurut berdasarkan kelas (seperti Iris), fold pertama hanya berisi kelas 0 semua → hasil tidak representatif.

    print("\nK-Fold CV")
    print(f"Scores per fold : {scores.round(4)}")
    print(f"Mean            : {scores.mean():.4f}")
    print(f"Std             : {scores.std():.4f}")
    print(f"95% CI          : {scores.mean():.4f} ± {scores.std()*2:.4f}")

    #Masalah Kfold Biasa pada data imbelence
    x_imb, y_imb = make_classification(
        n_samples=1000,
        weights=[0.9, 0.1],
        random_state=42
    )

    print(f"Kelas 0: {(y_imb == 0).sum()}")  # 900
    print(f"Kelas 1: {(y_imb == 1).sum()}")  # 100

    #Masalah: Kalau pakai KFold biasa dengan k=5, setiap fold berisi ~200 sampel. Tapi karena acak, bisa saja:
    # Fold 1: 195 kelas 0, 5 kelas 1
    # Fold 3: 180 kelas 0, 20 kelas 1
    # Distribusi kelas per fold tidak terjamin konsisten → evaluasi jadi tidak stabil.

    #Solusi: Stratified K-Fold
    # Stratified: proporsi kelas dijaga
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_skf = cross_val_score(model, x_imb, y_imb, cv=skf, scoring='accuracy')

    # Bandingkan dengan KFold biasa
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_kf = cross_val_score(model, x_imb, y_imb, cv=kf, scoring='accuracy')

    print("\n=== KFold Biasa ===")
    print(f"Scores : {scores_kf.round(4)}")
    print(f"Std    : {scores_kf.std():.4f}")

    print("\n=== Stratified KFold ===")
    print(f"Scores : {scores_skf.round(4)}")
    print(f"Std    : {scores_skf.std():.4f}")
    #yang diharapkan adalah kelas 0 dan kelas 1 memiliki proporsi yang sama di setiap fold, sehingga distribusi kelas per fold akan terjamin konsisten.

    #CV vs Train-Test Split — Perbandingan Langsung
    # --- Train-Test Split biasa (3 percobaan random_state berbeda) ---
    for i in [0,49,99]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"=== Train-Test Split 3 perscobaan random state berbeda ===")
        print(f"  random_state={i:>2}: {model.score(X_train, y_train):.4f}")

    # --- K-Fold CV ---
    # --- Stratified K-Fold CV ---
    print("\n=== Stratified K-Fold CV (k=5) ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf)
    print(f"  Scores : {scores.round(4)}")
    print(f"  Mean   : {scores.mean():.4f} ± {scores.std():.4f}")

    # Kalau tidak ada alasan khusus, selalu pakai StratifiedKFold dengan k=5. Ini default yang paling banyak dipakai di industri.

if __name__ == "__main__":    
    main()
