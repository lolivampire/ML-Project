import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from datetime import datetime

# ── CONSTANTS ──────────────────────────────────────────────────────
# ── DATASET SEDERHANA ─────────────────────────────────────────────────
# Fitur: [jam_belajar, jam_tidur]
# Label: 1 = lulus, 0 = tidak lulus
# np.random.seed(42) 
# X_lulus = np.random.randn(50, 2) + np.array([2, 2])
# X_tidak = np.random.randn(50, 2) + np.array([-2, -2])
# X = np.vstack([X_lulus, X_tidak])
# y = np.array([1] * 50 + [0] * 50)

#new dataset
np.random.seed(99)
X_pos = np.random.randn(60, 2) + np.array([3, 3])
X_neg = np.random.randn(60, 2) + np.array([-3, -3])
X = np.vstack([X_pos, X_neg])
y = np.array([1] * 60 + [0] * 60)

# ── TRAIN SKLEARN ──────────────────────────────────────────
# Moved to main function for better organization

# ── FUNCTIONS ──────────────────────────────────────────────────────

# ── SIGMOID ───────────────────────────────────────────────
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Terapkan sigmoid function: σ(z) = 1 / (1 + e^{-z})."""
    return 1 / (1 + np.exp(-z))

# ── LOGISTIC REGRESSION FROM SCRATCH ──────────────────────
def logistic_regression_train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    epochs: int = 1000
) -> tuple[np.ndarray, float]:
    """
    Train Logistic Regression dengan Gradient Descent.
    
    Returns:
        w: bobot (weights)
        b: bias
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for epoch in range(epochs):
        # Forward pass: hitung probabilitas
        z = X @ w + b          # dot product: (n_samples,)
        y_pred = sigmoid(z)    # probabilitas: (n_samples,)

        # Gradients (turunan dari Binary Cross-Entropy loss)
        error = y_pred - y                        # (n_samples,)
        dw = (X.T @ error) / n_samples            # (n_features,)
        db = np.mean(error)                       # scalar

        # Update bobot
        w -= lr * dw
        b -= lr * db

    return w, b


def logistic_regression_predict(
    X: np.ndarray,
    w: np.ndarray,
    b: float,
    threshold: float = 0.5
) -> np.ndarray:
    """Prediksi class label berdasarkan threshold."""
    z = X @ w + b
    proba = sigmoid(z)
    return (proba >= threshold).astype(int)

# ── VISUALISASI DECISION BOUNDARY ──────────────────────────
def plot_decision_boundary(w: np.ndarray, b: float, X: np.ndarray, y: np.ndarray) -> None:
    """Visualisasikan decision boundary dari model scratch di 2D feature space."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = (sigmoid(grid @ w + b) > 0.5).astype(int).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.contour(xx, yy, Z, colors='black', linewidths=1.5)

    plt.scatter(X[y == 1, 0], X[y == 1, 1],
                color='blue', label='Lulus (1)', edgecolors='k')
    plt.scatter(X[y == 0, 0], X[y == 0, 1],
                color='red', label='Tidak Lulus (0)', edgecolors='k')

    plt.title("Decision Boundary — Logistic Regression")
    plt.xlabel("Fitur 1 (jam belajar)")
    plt.ylabel("Fitur 2 (jam tidur)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    
    # Cek apakah file sudah ada, jika ya buat nama baru dengan timestamp
    base_filename = 'decision_boundary.png'
    filepath = os.path.join(output_dir, base_filename)
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'decision_boundary_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300)
    print(f"Plot saved to: {filepath}")
    plt.show()

# ── MAIN ────────────────────────────────────────────────────────────
def main() -> None:
    """Main function to fit and evaluate logistic regression models."""
    print("===================================================================")
    print("=======Implementasi fungsi sigmoid dan verifikasi outputny=========")
    #TASK 1: Implementasi fungsi sigmoid dan verifikasi outputnya
    print(f"sigmoid(0) = {sigmoid(0):.6f}, expect: 0.5")
    print(f"sigmoid(100) = {sigmoid(100):.6f}, expect: 1.0")
    print(f"sigmoid(-100) = {sigmoid(-100):.6f}, expect: 0.0")
    print(f"sigmoid(array) = {sigmoid(np.array([0, 1, -1]))}")

    print("===================================================================")
    print("========Implementasi Logicstic Regression dari Scratch=============")
    #TASK 2: Implementasi Logistic Regression dari scratch 
    #fungsi train dengan gradient descent
    w, b = logistic_regression_train(X, y, lr=0.1, epochs=1000)
    #fungsi predict dengan threshold 0.5
    y_pred = logistic_regression_predict(X, w, b)
    accuracy = np.mean(y_pred == y)

    #print hasil bobot, bias, dan akurasi
    print(f"Model parameters: w = {w}, b = {b:.2f}")
    print(f"Accuracy (Scratch): {accuracy:.2%}")   
    
    print("===================================================================")
    print("==========Sklearn dan Visualisasi Decision Boundary================")
    #TASK 3: Sklearn dan Visualisasi Decision Boundary
    #Train LogisticRegression dari sklearn pada dataset yang sama
    model = LogisticRegression()
    model.fit(X, y)

    #Visualisasikan decision boundary dengan matplotlib
    plot_decision_boundary(w, b, X, y)

    #Bandingkan accuracy scratch vs sklearn — print keduanya
    y_pred_sklearn = model.predict(X)
    accuracy_sklearn = np.mean(y_pred_sklearn == y)
    print(f"Accuracy (Scratch): {accuracy:.2%}")
    print(f"Accuracy (Sklearn): {accuracy_sklearn:.2%}")

    """ Rangkuman Pipeline Logistic Regression
    1. Input X (fitur)
    2. z = w·X + b          ← kombinasi linear (log-odds)
    3. σ(z) = 1/(1+e^{-z}) ← sigmoid → probabilitas [0,1]
    4. threshold 0.5        ← keputusan akhir
    5. ŷ ∈ {0, 1}          ← prediksi class label"""

if __name__ == "__main__":
    main()