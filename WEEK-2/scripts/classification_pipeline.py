"""
W02D06 — Mini Project: Classification
Dataset: Breast Cancer Wisconsin (sklearn built-in)
Models: Linear Regression (baseline), Logistic Regression, Decision Tree
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, cross_val_score, learning_curve, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD & EDA
# ─────────────────────────────────────────────

def load_and_eda() -> tuple:
    """Load dataset dan print ringkasan EDA."""
    data = load_breast_cancer()
    X, y = data.data, data.target

    print("=" * 50)
    print("BREAST CANCER WISCONSIN — EDA SUMMARY")
    print("=" * 50)
    print(f"Shape         : {X.shape}")          # (569, 30)
    print(f"Features      : {X.shape[1]}")
    print(f"Samples       : {X.shape[0]}")
    print(f"Classes       : {data.target_names}") # ['malignant', 'benign']
    print(f"Class balance : malignant={np.sum(y==0)}, benign={np.sum(y==1)}")
    print(f"Positive rate : {np.mean(y==1):.1%}")
    print("=" * 50)

    return X, y, data


def plot_class_distribution(y: np.ndarray, output_path: str) -> None:
    """Visualisasi distribusi kelas — cek apakah imbalanced."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['Malignant (0)', 'Benign (1)']
    counts = [np.sum(y == 0), np.sum(y == 1)]
    colors = ['#E24B4A', '#1D9E75']

    bars = ax.bar(labels, counts, color=colors, edgecolor='white', width=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', fontsize=11)

    ax.set_title('Class Distribution — Breast Cancer Wisconsin', fontsize=13)
    ax.set_ylabel('Count')
    ax.set_ylim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[saved] {output_path}")

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    Split dan scale.
    Kenapa scale? LinearRegression dan LogisticRegression sensitif terhadap
    skala fitur. Decision Tree tidak — tapi kita pakai scaler yang sama
    untuk konsistensi pipeline.
    """
    # Stratified split — pastikan proporsi kelas sama di train dan test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y          # <-- WAJIB untuk dataset dengan imbalance
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit HANYA di training data
    X_test_sc = scaler.transform(X_test)          # transform test TANPA fit ulang

    print(f"\nTrain size : {X_train_sc.shape[0]} samples")
    print(f"Test size  : {X_test_sc.shape[0]} samples")

    return X_train_sc, X_test_sc, y_train, y_test


# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────

def build_models() -> dict:
    """
    Tiga model + 1 dummy baseline.
    Linear Regression bukan classifier — tapi kita pakai sebagai
    numerical baseline: threshold 0.5 untuk konversi ke kelas.
    Ini memperlihatkan kenapa classifier proper lebih tepat.
    """
    return {
        "Dummy (majority)": DummyClassifier(strategy="most_frequent", random_state=42),
        "Linear Regression": LinearRegression(),    # numerical baseline
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }


def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test) -> dict:
    """Train semua model, hitung accuracy dan CV score."""
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        # Prediksi (handle LinearRegression yang output float)
        if isinstance(model, LinearRegression):
            # Threshold 0.5: nilai >= 0.5 → kelas 1
            y_pred = (model.predict(X_test) >= 0.5).astype(int)
            y_pred_train = (model.predict(X_train) >= 0.5).astype(int)
            cv_scores = None  # Linear Regression tidak pakai CV score standar
        else:
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)

            # StratifiedKFold — pastikan proporsi kelas terjaga di setiap fold
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "cv_mean": cv_scores.mean() if cv_scores is not None else None,
            "cv_std": cv_scores.std() if cv_scores is not None else None,
        }

        # Print ringkasan
        cv_str = (f"CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
                  if cv_scores is not None else "CV: N/A")
        print(f"\n{name}")
        print(f"  Train acc : {train_acc:.4f}")
        print(f"  Test acc  : {test_acc:.4f}")
        print(f"  {cv_str}")

    return results

# ─────────────────────────────────────────────
# 4. LEARNING CURVE — DIAGNOSIS DULU
# ─────────────────────────────────────────────

def plot_learning_curves(models: dict, X_train, y_train, output_path: str) -> None:
    """
    Plot learning curve untuk SETIAP model classifier (bukan LinReg).
    Ingat urutan diagnosis:
      1. Lihat gap antara train dan validation curve
      2. Lihat tren: apakah konvergen atau masih divergen?
      3. BARU tarik kesimpulan underfitting/overfitting
    """
    classifier_names = [k for k in models if k not in ("Dummy (majority)", "Linear Regression")]
    fig, axes = plt.subplots(1, len(classifier_names), figsize=(14, 5))

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 10)

    for ax, name in zip(axes, classifier_names):
        model = models[name]

        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        # Plot kurva
        ax.plot(train_sizes_abs, train_mean, 'o-', color='#E24B4A',
                label='Training score', linewidth=2)
        ax.fill_between(train_sizes_abs,
                        train_mean - train_std, train_mean + train_std,
                        alpha=0.15, color='#E24B4A')

        ax.plot(train_sizes_abs, val_mean, 'o-', color='#1D9E75',
                label='CV score', linewidth=2)
        ax.fill_between(train_sizes_abs,
                        val_mean - val_std, val_mean + val_std,
                        alpha=0.15, color='#1D9E75')

        # Anotasi gap di titik terakhir — ini yang kamu baca untuk diagnosis
        final_gap = train_mean[-1] - val_mean[-1]
        ax.annotate(f"gap={final_gap:.3f}",
                    xy=(train_sizes_abs[-1], (train_mean[-1] + val_mean[-1]) / 2),
                    xytext=(-60, 0), textcoords='offset points',
                    fontsize=9, color='gray')

        ax.set_title(name, fontsize=12)
        ax.set_xlabel('Training samples')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.75, 1.02)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Learning Curves — W02D06 Classification Project', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[saved] {output_path}")
    

# ─────────────────────────────────────────────
# 5. COMPARE & VISUALIZE
# ─────────────────────────────────────────────

def plot_model_comparison(results: dict, output_path: str) -> None:
    """Bar chart perbandingan test accuracy semua model."""
    names = list(results.keys())
    test_accs = [results[n]["test_acc"] for n in names]
    cv_means = [results[n]["cv_mean"] if results[n]["cv_mean"] is not None else 0
                for n in names]
    cv_stds = [results[n]["cv_std"] if results[n]["cv_std"] is not None else 0
               for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, test_accs, width, label='Test accuracy',
                   color='#378ADD', edgecolor='white')
    bars2 = ax.bar(x + width/2, cv_means, width,
                   yerr=cv_stds, capsize=5,
                   label='CV mean ± std', color='#1D9E75', edgecolor='white')

    # Annotate values
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Comparison — Test Accuracy & CV Score', fontsize=13)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"[saved] {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs/plots')
    os.makedirs(output_dir, exist_ok=True)

    # 1. EDA
    X, y, data = load_and_eda()
    plot_class_distribution(y, "../outputs/plots/class_distribution.png")

    # 2. Preprocessing
    X_train, X_test, y_train, y_test = preprocess(X, y)

    # 3. Train
    models = build_models()
    results = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    # 4. Learning curve — diagnosis dulu sebelum kesimpulan
    models_for_lc = {k: v for k, v in models.items()
                     if k not in ("Dummy (majority)", "Linear Regression")}
    plot_learning_curves(models_for_lc, X_train, y_train,
                         "../outputs/plots/learning_curves.png")

    # 5. Compare
    plot_model_comparison(results, "../outputs/plots/model_comparison.png")

    print("\n" + "=" * 50)
    print("DONE. Cek folder week-02/outputs/plots/")
    print("Langkah selanjutnya: baca learning curve dulu,")
    print("baru tarik kesimpulan model mana yang terbaik.")
    print("=" * 50)


if __name__ == "__main__":
    main()