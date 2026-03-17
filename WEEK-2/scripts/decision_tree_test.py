"""
W02D03 — Decision Trees: Splitting Criteria
Implementasi, visualisasi, dan analisis overfitting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score

output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs/plots')
os.makedirs(output_dir, exist_ok=True)

# ── CONSTANTS ────────────────────────────────────────────────────────────
#buat dataset sintetis dengan 2 fitur untuk visualisasi
X, y = make_classification(
    n_samples=200,
    n_features=2,        # 2 fitur agar bisa divisualisasikan
    n_redundant=0,
    n_informative=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── FUNCTIONS ────────────────────────────────────────────────────────────
def plot_decision_boundary(ax, model, X, y, title="Decision Boundary"):
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='k', marker='o', s=40)
    ax.set_xlabel('Fitur 0')
    ax.set_ylabel('Fitur 1')
    ax.set_title(title, fontsize=12, fontweight='bold')

#hitung gini manual dengan pure python
def gini_impurity(counts:  list[int]) -> float:
    """Calculate Gini impurity for a given set of labels."""
    total = sum(counts)
    if total == 0:
        return 0.0
    sum_squared_probs = sum((count / total) ** 2 for count in counts)
    return 1 - sum_squared_probs
# ── MAIN ────────────────────────────────────────────────────────────
def main() -> None:
    """ Main function to train and evaluate a decision tree classifier. """
    # ── 2. TRAIN: TREE TANPA BATAS (OVERFITTING) ─────────────────
    tree_overfit = DecisionTreeClassifier(random_state=42)  # no max_depth!
    tree_overfit.fit(X_train, y_train)

    train_acc_overfit = accuracy_score(y_train, tree_overfit.predict(X_train))
    test_acc_overfit  = accuracy_score(y_test,  tree_overfit.predict(X_test))

    # ── 3. Train: Tree dengan max-depth= 3 (terkontrol) ─────────────────
    tree_controlled = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_controlled.fit(X_train, y_train)
    train_acc_controlled = accuracy_score(y_train, tree_controlled.predict(X_train))
    test_acc_controlled  = accuracy_score(y_test,  tree_controlled.predict(X_test))

    #TASK 1 Hitung gini secara manual dengan pure python
    #Node Kiri: 40 sampel kelas 0, 10 sampel kelas 1
    #Node Kanan: 5 sampel kelas 0, 45 sampel kelas 
    print("\n--- Gini Impurity Calculation ---")
    left_counts = [40, 10]
    right_counts = [5, 45]
    gini_left = gini_impurity(left_counts)
    gini_right = gini_impurity(right_counts)
    print(f"Gini impurity for left node (40 class 0, 10 class 1): {gini_left:.4f}")
    print(f"Gini impurity for right node (5 class 0, 45 class 1): {gini_right:.4f}")

    #weighted gini ambil dari data aktual (gini generik)
    n_left  = sum(left_counts)
    n_right = sum(right_counts)
    n_total = n_left + n_right
    weighted_gini = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right
    print(f"Weighted Gini: {weighted_gini:.4f}")

    #Task 2 — Analisis Depth vs Accuracy
    #Buat loop yang train Decision Tree dengan max_depth dari 1 sampai 10. Untuk setiap depth, simpan train accuracy dan test accuracy. Lalu plot hasilnya sebagai line chart (dua garis: satu untuk train, satu untuk test).
    print("\n--- Depth vs Accuracy Analysis ---")
    depths = range(1, 11)
    train_accuracies = []
    test_accuracies = []
    for depth in depths:
        tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
        tree.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, tree.predict(X_train))
        test_acc = accuracy_score(y_test, tree.predict(X_test))
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f"Depth={depth}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}")
    plt.figure(figsize=(8, 5))
    plt.plot(depths, train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(depths, test_accuracies, marker='s', label='Test Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Depth vs Accuracy Analysis')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_analysis3.png'), dpi=300)
    plt.show()

    #TASK 3 - Dari tree max_depth=3 yang sudah ditrain di Blok 2, gunakan export_text() untuk print struktur tree-nya
    print("\n--- Tree Rules (Controlled) ---")
    tree_text = export_text(tree_controlled, feature_names=['Fitur 0', 'Fitur 1'])
    print(tree_text)

    #output dari task
# --- Gini Impurity Calculation ---
# Gini impurity for left node (40 class 0, 10 class 1): 0.3200
# Gini impurity for right node (5 class 0, 45 class 1): 0.1800
# Weighted Gini impurity for split: 0.2500

# --- Depth vs Accuracy Analysis ---
# Depth=1: Train Acc=0.9000, Test Acc=0.8250
# Depth=2: Train Acc=0.9125, Test Acc=0.8000
# Depth=3: Train Acc=0.9250, Test Acc=0.8000
# Depth=4: Train Acc=0.9437, Test Acc=0.7500
# Depth=5: Train Acc=0.9750, Test Acc=0.7750
# Depth=6: Train Acc=0.9875, Test Acc=0.7750
# Depth=7: Train Acc=0.9938, Test Acc=0.8000
# Depth=8: Train Acc=1.0000, Test Acc=0.8000
# Depth=9: Train Acc=1.0000, Test Acc=0.8000
# Depth=10: Train Acc=1.0000, Test Acc=0.8000

#  --- Tree Rules (Controlled) ---
# |--- Fitur 0 <= -0.15
# |   |--- Fitur 1 <= -2.11
# |   |   |--- Fitur 0 <= -0.38
# |   |   |   |--- class: 1
# |   |   |--- Fitur 0 >  -0.38
# |   |   |   |--- class: 0
# |   |--- Fitur 1 >  -2.11
# |   |   |--- Fitur 0 <= -0.80
# |   |   |   |--- class: 0
# |   |   |--- Fitur 0 >  -0.80
# |   |   |   |--- class: 0
# |--- Fitur 0 >  -0.15
# |   |--- Fitur 1 <= -2.45
# |   |   |--- class: 0
# |   |--- Fitur 1 >  -2.45
# |   |   |--- Fitur 1 <= 0.77
# |   |   |   |--- class: 1
# |   |   |--- Fitur 1 >  0.77
# |   |   |   |--- class: 1

# SPLIT 1: Tree bertanya apakah [fitur 0] <= [nilai -0.15]
#           Jika ya -> LANJUT SPLIT 2 (fokus pada data di mana fitur 0 <= -0.15) | jika tidak -> LANJUT SPLIT 5 (fokus pada data di mana Fitur 0 > -0.15)
# SPLIT 2: Tree bertanya apakah [Fitur 1] <= [-2.11] (cabang kiri SPLIT 1)
#           Jika ya -> Lanjut ke SPLIT 3 (cabang kiri: fokus pada data di mana Fitur 1 ≤ -2.11). | Jika tidak -> Lanjut ke SPLIT 4 (cabang kanan: fokus pada data di mana Fitur 1 > -2.11).
# **SPLIT 3: Tree bertanya apakah [Fitur 0] <= [-0.38]** (ini adalah cabang kiri dari SPLIT 2).
#           Jika ya -> Prediksi kelas 1 (daun: semua data di sini diprediksi sebagai kelas 1).| Jika tidak -> Prediksi kelas 0 (daun: semua data di sini diprediksi sebagai kelas 0).
# **SPLIT 4: Tree bertanya apakah [Fitur 0] <= [-0.80]** (ini adalah cabang kanan dari SPLIT 2).
#           Jika ya -> Prediksi kelas 0 (daun: semua data di sini diprediksi sebagai kelas 0). | Jika tidak -> Prediksi kelas 0 (daun: semua data di sini diprediksi sebagai kelas 0). (Catatan: Kedua cabang di SPLIT 4 menghasilkan kelas 0, menunjukkan bahwa data di cabang ini cukup homogen.)
# **SPLIT 5: Tree bertanya apakah [Fitur 1] <= [-2.45]** (ini adalah cabang kanan dari SPLIT 1).
#           Jika ya -> Prediksi kelas 0 (daun: semua data di sini diprediksi sebagai kelas 0).| Jika tidak -> Lanjut ke SPLIT 6 (cabang kanan: fokus pada data di mana Fitur 1 > -2.45).
# **SPLIT 6: Tree bertanya apakah [Fitur 1] <= [0.77]** (ini adalah cabang kanan dari SPLIT 5).
#           Jika ya -> Prediksi kelas 1 (daun: semua data di sini diprediksi sebagai kelas 1).|Jika tidak -> Prediksi kelas 1 (daun: semua data di sini diprediksi sebagai kelas 1).
if __name__ == "__main__":    
    main()