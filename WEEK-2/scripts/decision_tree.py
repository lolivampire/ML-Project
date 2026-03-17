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

# ── 4. Visualisasi: Plot decision boundary dan tree structure ─────────────────
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
    """Main function to train and evaluate a decision tree classifier."""
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

    # Plot decision boundaries
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Visualisasi boundary overfit
    plot_decision_boundary(
        axes[0], tree_overfit, X_test, y_test,
        f"Overfit Tree (depth={tree_overfit.get_depth()})\n"
        f"Train: {train_acc_overfit:.2%} | Test: {test_acc_overfit:.2%}"
    )

    # Visualisasi boundary controlled
    plot_decision_boundary(
        axes[1], tree_controlled, X_test, y_test,
        f"Controlled Tree (depth={tree_controlled.get_depth()})\n"
        f"Train: {train_acc_controlled:.2%} | Test: {test_acc_controlled:.2%}"
    )

    # Visualisasi struktur tree controlled
    plot_tree(
        tree_controlled,
        filled=True,
        feature_names=['Feature 0', 'Feature 1'],
        class_names=['Class 0', 'Class 1'],
        ax=axes[2],
        fontsize=8
    )
    axes[2].set_title("Tree Structure (depth=3)", fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'decision_tree_analysis.png'), dpi=300)
    plt.show()

    #cetak perbandingan
    print("=" * 45)
    print("PERBANDINGAN: Overfit vs Controlled")
    print("=" * 45)
    print(f"{'':20} {'OVERFIT':>10} {'CONTROLLED':>10}")
    print(f"{'Train Accuracy':20} {train_acc_overfit:>10.4f} {train_acc_controlled:>10.4f}")
    print(f"{'Test Accuracy':20} {test_acc_overfit:>10.4f} {test_acc_controlled:>10.4f}")
    print(f"{'Tree Depth':20} {tree_overfit.get_depth():>10} {tree_controlled.get_depth():>10}")
    print(f"{'Num Leaves':20} {tree_overfit.get_n_leaves():>10} {tree_controlled.get_n_leaves():>10}")
    print("=" * 45)

    #eksplorasi gini vs entropy
    print("\n--- Gini vs Entropy Comparison ---")
    for criterion in ['gini', 'entropy']:
        tree = DecisionTreeClassifier(criterion=criterion, max_depth=3, random_state=42)
        tree.fit(X_train, y_train)
        acc = accuracy_score(y_test, tree.predict(X_test))
        print(f"criterion='{criterion}': test accuracy = {acc:.4f}, depth = {tree.get_depth()}")

    # Cetak struktur tree sebagai teks
    print("\n--- Tree Rules (Controlled) ---")
    tree_text = export_text(tree_controlled, feature_names=['Fitur 0', 'Fitur 1'])
    print(tree_text)

    

if __name__ == "__main__":    main()