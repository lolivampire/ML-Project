import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import learning_curve

# ── CONSTANTS ──────────────────────────────────────────────────────
x, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    random_state=42
    )
# ── FUNCTIONS ──────────────────────────────────────────────────────
def plot_learning_curve(X, y) :
    """
    Plot learning curve for a given model.

    Parameters
    ----------
    model : sklearn estimator
        Model to plot learning curve for.
    title : str
        Title of the plot.
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.

    Returns
    -------
    None
    """

    #define models with their configurations
    models = [
        (DecisionTreeClassifier(max_depth=1, random_state=42), 'Underfitting\n(max_depth=1)'),
        (DecisionTreeClassifier(max_depth=5, random_state=42), 'Sweet spot\n(max_depth=5)'),
        (DecisionTreeClassifier(max_depth=None, random_state=42), 'Overfitting\n(max_depth=None)'),
    ]

    # Create figure with 3 subplots horizontally
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle('Learning Curves: Model Complexity Comparison', fontsize=14, fontweight='bold')
    
    #plot each model
    for i, (model,title) in enumerate(models):
        ax = axes[i]

        #hitung learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy'
        )

        #hitung means dan standard deviasi
        train_mean = train_scores.mean(axis=1)
        val_mean = test_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_std = test_scores.std(axis=1)

        #plot lines
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy', linewidth=2)
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy', linewidth=2)

        #fill uncertainty area
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color='red', alpha=0.2)

        #STYLING
        #set title
        ax.set_title(title, fontsize=12, fontweight='bold')

        #set axis labels
        ax.set_xlabel('Training Set Size', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)

        #set grid
        ax.grid(True, alpha=0.3, linestyle='dashed')

        #set legend
        ax.legend(loc='lower right', fontsize=10)
    
    #adjust space
    plt.tight_layout()
    #save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs/bias_variance')
    os.makedirs(output_dir, exist_ok=True)
    # Cek apakah file sudah ada, jika ya buat nama baru dengan timestamp
    base_filename = 'bias-variance'
    filepath = os.path.join(output_dir, base_filename)
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{base_filename}_{timestamp}.png'
        filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300)
    print(f"Plot saved to: {filepath}")
    plt.show()
        

# ── MAIN ───────────────────────────────────────────────────────────
def main() -> None:
    """Main function."""
    plot_learning_curve(x, y)

    # Di grafik underfitting, berapa kira-kira nilai akhir training score dan validation score-nya? Apakah gap-nya besar atau kecil?
    # -> nilai akhirnya ada diantara 0.73 - 0.75, gap-nya kecil atau disebut High Bias karena Model gagal menangkap pola dasar dalam data, sehingga baik pada data latihan maupun data baru, performanya sama-sama buruk. menambah lebih banyak data tidak akan membantu kondisi ini. Kedua garis sudah plateau di angka rendah yang sama. Solusinya bukan data lebih banyak, tapi model lebih kompleks.
    # Di grafik overfitting, berapa training score-nya? Berapa validation score-nya? Berapa gap-nya?
    # -> training score-nya konsisten di akurasi 1 namun validation scorenya jauh dibawah yaitu 0,85, terdapat gap yang besar atau disebut High Variance karena Model terlalu terikat pada data latihan sehingga performanya tidak baik pada data baru. yang berbahaya adalah gap ini akan terus bertambah ketika data latihan berubah. Kedua garis sudah plateau di angka rendah yang sama. Solusinya bukan data lebih banyak, tapi model lebih kompleks.
    # max_depth berapa yang kamu pilih untuk sweet spot, dan bagaimana bentuk grafiknya?
    # -> max_depth = 5, grafiknya sebagian besar naik berada diantara 0.85-0.86, dan garis train nya sedikit menurun berada di 0,95. gap antara train dan validation cukup kecil. Model berhasil mempelajari pola umum tanpa menghafal noise dengan kondisi varians berada pada keadaan optimal.
if __name__ == '__main__':
    main()
