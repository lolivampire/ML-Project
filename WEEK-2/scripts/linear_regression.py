"""
linear_regression_scratch.py
W02D01 — Linear Regression dari Scratch
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ── CONSTANTS ──────────────────────────────────────────────────────
# !!! argumen pertama = fitur, argumen kedua = target.!!!

# Simulasi: luas rumah (m²) vs harga (juta rupiah)
np.random.seed(42)
X = np.random.uniform(20, 150, size=100)          # luas: 20–150 m²
y = 200 + 5 * X + np.random.normal(0, 30, 100)    # harga = 200 + 5*luas + noise

# Simulasi: jam belajar vs nilai ujian
np.random.seed(21)
jam_belajar = np.random.uniform(0, 10, size=100)          # jam belajar: 0–10 jam
nilai_ujian = 50 + 5 * jam_belajar + np.random.normal (0, 10, 100)    # nilai ujian = 50 + 5*jam belajar + noise

# ── FUNCTIONS ──────────────────────────────────────────────────────

# ── 1.  CLOSED-FORM (Normal Equation) ─────────────────────────────
def fit_closed_form(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
        Hitung w0 (intercept) dan w1 (slope) menggunakan Normal Equation.
        Formula: w = (XᵀX)⁻¹ Xᵀy
        X_b = design matrix dengan kolom bias (kolom 1 di depan)
    """
    # Tambahkan kolom bias (intercept) ke X
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # shape: (n_samples, 2)
    # Hitung parameter w menggunakan Normal Equation
    w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return w[0], w[1]  # w0 (intercept), w1 (slope)

# ── 2. GRADIENT DESCENT ─────────────────────────────────────────
def fit_gradient_descent(X: np.ndarray, y: np.ndarray, learning_rate:float=0.01, n_iterations=1000):
    """
        Hitung w0 (intercept) dan w1 (slope) menggunakan Gradient Descent.
        Update rule: w = w - learning_rate * gradient (MSE)
        Data dinormalisasi untuk konvergensi yang lebih baik.
    """
    # Normalisasi X
    X_mean = np.mean(X)
    X_std = np.std(X)
    X_norm = (X - X_mean) / X_std
    
    m = len(X_norm)
    w1_norm = 0.0  # slope untuk X_norm
    w0 = np.mean(y)  # intercept awal sebagai mean y
    
    for iteration in range(n_iterations):
        y_pred = w0 + w1_norm * X_norm
        error = y_pred - y
        dw0 = (2/m) * np.sum(error)
        dw1_norm = (2/m) * np.sum(error * X_norm)
        w0 -= learning_rate * dw0
        w1_norm -= learning_rate * dw1_norm
    
    # Konversi kembali ke skala asli
    w1 = w1_norm / X_std
    w0 = w0 - w1_norm * X_mean / X_std
    
    return w0, w1

# ── EVALUATE ─────────────────────────────────────────
def evaluate_model(name: str, w0: float, w1: float, X: np.ndarray, y: np.ndarray) -> None:
    """Evaluasi model dengan MSE dan R²."""
    print(f"\n{name} Method:")
    y_pred = w0 + w1 * X
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Model parameters: w0 (intercept) = {w0:.2f}, w1 (slope) = {w1:.2f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

# ── MAIN ──────────────────────────────────────────────────────p
def main() -> None:
    """Main function to fit and evaluate linear regression models."""
    # Linear Regression dengan Closed-Form (Normal Equation)
    w0_cf, w1_cf = fit_closed_form(jam_belajar, nilai_ujian)
    # Linear Regression dengan Gradient Descent
    w0_gd, w1_gd = fit_gradient_descent(jam_belajar, nilai_ujian)
    # sklearn (ground truth)
    model = LinearRegression()
    model.fit(jam_belajar.reshape(-1, 1), nilai_ujian)
    w0_sk, w1_sk = model.intercept_, model.coef_[0]
    
    print("=" * 75)
    print("Linear Regression — Perbandingan 3 Metode")
    print("=" * 75)
    evaluate_model("Closed-Form",      w0_cf, w1_cf, jam_belajar, nilai_ujian)
    evaluate_model("Gradient Descent", w0_gd, w1_gd, jam_belajar, nilai_ujian)
    evaluate_model("Sklearn",          w0_sk, w1_sk, jam_belajar, nilai_ujian)
    print("=" * 75)
    print(f"Ground truth: intercept≈50, slope≈5")

# ===========================================================================
# Linear Regression — Perbandingan 3 Metode
# ===========================================================================

# Closed-Form Method:
# Model parameters: w0 (intercept) = 49.65, w1 (slope) = 4.92
# Mean Squared Error: 101.09
# R² Score: 0.6394

# Gradient Descent Method:
# Model parameters: w0 (intercept) = 49.65, w1 (slope) = 4.92
# Mean Squared Error: 101.09
# R² Score: 0.6394

# Sklearn Method:
# Model parameters: w0 (intercept) = 49.65, w1 (slope) = 4.92
# Mean Squared Error: 101.09
# R² Score: 0.6394
# ===========================================================================
# Ground truth: intercept≈50, slope≈5

    # INTERPRETASI:
    # Setiap tambah 1 jam belajar → nilai naik berapa poin? 
    # -> Setiap tambah 1 jam belajar, nilai ujian naik sekitar 4.92 poin (berdasarkan model Closed-Form dan Sklearn yang sudah mendekati ground truth). Model Gradient Descent belum converge dengan baik, jadi hasilnya jauh dari ground truth.
    # R² model ini artinya apa?
    # -> R² Score sekitar 0.6394 artinya model Linear Regression ini bisa menjelaskan sekitar 63.94% variasi nilai
    # Apakah Gradient Descent converge mendekati Closed-Form?
    # -> Ya. Dengan normalisasi fitur (zero mean, unit variance) dan lr=0.01, GD converge ke w0=49.65, w1=4.92 — identik dengan Closed-Form. Normalisasi adalah kunci: tanpanya GD sensitif terhadap skala data.
# apakah Gradient Descent pasti lebih akurat dari Closed-Form karena dia iterasi berkali-kali?
# -> Tidak. Closed-Form memberikan solusi analitik yang tepat (jika XᵀX invertible), sedangkan GD adalah metode numerik yang bisa converge ke solusi yang sama jika diatur dengan benar (learning rate, normalisasi, iterasi cukup). Namun, GD bisa gagal converge atau converge ke solusi suboptimal jika parameter tidak tepat. Jadi, GD tidak selalu lebih akurat dari Closed-Form; itu tergantung pada implementasi dan kondisi data.
# Kamu punya dataset dengan 2 fitur: jam_belajar (range 0–10) dan uang_jajan (range 500.000–2.000.000). Kamu langsung run Gradient Descent tanpa normalisasi. Apa yang kemungkinan besar terjadi, dan kenapa?
# -> Kemungkinan besar GD tidak akan converge dengan baik karena fitur jam_belajar dan uang_jajan memiliki skala yang sangat berbeda. Tanpa normalisasi, gradien untuk uang_jajan akan jauh lebih besar daripada untuk jam_belajar, sehingga update parameter akan didominasi oleh fitur uang_jajan. Ini bisa menyebabkan GD gagal menemukan solusi optimal atau bahkan divergen. Normalisasi fitur sangat penting untuk memastikan bahwa GD bekerja dengan efektif pada dataset dengan fitur yang memiliki skala berbeda.
# jika intercept = 49.65 dan slope = 4.92, dan r² = 0.6394, interpretasi model jika dijelaskan dengan bahasa non teknis agan seperti apa?
# -> Model ini menunjukkan bahwa nilai ujian seseorang dipengaruhi oleh jumlah jam belajar mereka. Secara rata-rata, setiap tambahan 1 jam belajar akan meningkatkan nilai ujian sekitar 4.92 poin. Namun, model ini hanya bisa menjelaskan sekitar 63.94% dari variasi nilai ujian, yang berarti ada faktor lain (seperti kualitas tidur, stres, atau metode belajar) yang juga mempengaruhi nilai ujian tetapi tidak termasuk dalam model ini. Jadi, meskipun jam belajar penting, itu bukan satu-satunya faktor yang menentukan hasil ujian seseorang.

if __name__ == "__main__":
    main()