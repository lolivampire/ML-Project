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

# Q1. Kenapa Linear Regression tidak cocok untuk masalah klasifikasi? Berikan 1 contoh konkret kenapa outputnya bermasalah.
# Jawaban: Linear Regression menghasilkan output kontinu yang bisa berada di luar rentang [0, 1], sehingga tidak bisa langsung diinterpretasikan sebagai probabilitas. Contohnya, jika kita menggunakan Linear Regression untuk memprediksi apakah seseorang akan lulus (1) atau tidak lulus (0) berdasarkan jam belajar, model bisa saja memprediksi nilai seperti -0.5 atau 1.5, yang tidak masuk akal dalam konteks klasifikasi biner.

# Q2. Sigmoid function menerima input z = -50. Kira-kira outputnya berapa, dan itu artinya apa dalam konteks klasifikasi?
# Jawaban: Sigmoid(-50) akan menghasilkan output yang sangat mendekati 0 (sekitar 1.92874985e-22). Dalam konteks klasifikasi, ini berarti model sangat yakin bahwa input tersebut termasuk dalam kelas 0 (tidak lulus), karena probabilitasnya sangat rendah untuk kelas 1 (lulus).
# Q3. Decision boundary muncul dari mana? Apakah kita menggambarnya manual, atau ada proses lain yang menghasilkannya?
# Jawaban: Decision boundary muncul dari model yang kita latih. Untuk Logistic Regression, decision boundary adalah garis (atau hyperplane) di mana probabilitas prediksi adalah 0.5. Kita tidak menggambar decision boundary secara manual; melainkan, kita menghitungnya berdasarkan parameter model (w dan b) yang diperoleh dari proses training. Dalam visualisasi, kita menggunakan grid untuk mengevaluasi model di berbagai titik dan menggambar contour di mana output model berubah dari kelas 0 ke kelas 1 (probabilitas 0.5).
# Q4. Dalam kode kamu, baris ini:
# pythondw = (X.T @ error) / n_samples
# error itu isinya apa? Kenapa kita hitung ini?
# Jawaban: "error" adalah selisih antara probabilitas prediksi (y_pred) dan label sebenarnya (y). Ini menunjukkan seberapa jauh prediksi model dari nilai yang benar. Kita menghitung error ini untuk mendapatkan gradien (dw dan db) yang digunakan dalam proses update parameter (w dan b) selama training. Gradien ini memberi tahu kita arah dan besarnya perubahan yang perlu dilakukan pada parameter untuk meminimalkan loss function (Binary Cross-Entropy) dan meningkatkan akurasi model.

if __name__ == "__main__":
    main()