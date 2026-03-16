import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
#set global seaborn
sns.set_theme(style='whitegrid')

# ── CONSTANTS ──────────────────────────────────────────────────────
np.random.seed(42)
df = pd.DataFrame({
    'age':        np.random.normal(35, 10, 200).astype(int),
    'salary':     np.random.normal(60000, 15000, 200),
    'experience': np.random.normal(8, 4, 200).astype(int),
    'department': np.random.choice(['Engineering', 'Marketing', 'Finance', 'HR'], 200),
    'score':      np.random.uniform(0, 100, 200)
})

# ── FUNCTIONS ──────────────────────────────────────────────────────

def plot_histogram(df, column, output_dir='../outputs/plots'):
    """Membuat histogram untuk kolom tertentu dengan KDE."""
    # make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column], kde=True, ax=ax)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{column}_histogram.png'), dpi=300)
    plt.show()

def plot_boxplot(df, column, output_dir='../outputs/plots'):
    """Membuat boxplot untuk kolom tertentu."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df[column], ax=ax)
    ax.set_title(f'Boxplot of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Value')
    plt.savefig(os.path.join(output_dir, f'{column}_boxplot.png'), dpi=300)
    plt.show()

#plot_scatter(df, x, y, hue, output_dir)
def plot_scatter(df, x, y, hue=None, output_dir='../outputs/plots'):
    """Membuat scatter plot antara dua kolom dengan opsi hue."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    ax.set_title(f'Scatter Plot of {y} vs {x}')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    filename = f'{y}_vs_{x}_scatter.png' if not hue else f'{y}_vs_{x}_scatter_by_{hue}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.show()

#bar plot_categorical(df, column, output_dir)
def plot_categorical(df, column, output_dir='../outputs/plots'):
    """Membuat count plot untuk kolom kategorikal."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(x=df[column], ax=ax)
    ax.set_title(f'Count Plot of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    plt.savefig(os.path.join(output_dir, f'{column}_countplot.png'), dpi=300)
    plt.show()

def bar_charts(df,x,y, output_dir='./WEEK-1/outputs/plots'):
    """Membuat bar chart rata-rata y berdasarkan x."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x=x, y=y, estimator='mean', ax=ax)
    ax.set_title(f'Average {y} by {x}')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{y}_by_{x}_barchart.png'), dpi=300)
    plt.show()

#plot_correlation_heatmap(df, output_dir)
def plot_correlation_heatmap(df, output_dir='../outputs/plots'):
    """Membuat heatmap korelasi untuk kolom numerik."""
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df.select_dtypes(include='number').corr()
    sns.heatmap(corr_matrix, 
                annot=True,  #tampilkan angka korelasi di setiap sel
                cmap='coolwarm',  #gunakan colormap yang kontras untuk korelasi positif dan negatif
                center=0, #set center ke 0 untuk menonjolkan korelasi positif dan negatif
                vmin=-1, #set minimum value to -1 for proper scaling
                vmax=1,#set maximum value to 1 for proper scaling
                ax=ax
            ) 
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.show()

# ── MAIN ──────────────────────────────────────────────────────
def main() -> None:
    """Fungsi utama untuk menjalankan semua plotting."""
    plot_histogram(df, 'age', output_dir='../outputs/plots')
    plot_histogram(df, 'salary', output_dir='../outputs/plots')
    plot_boxplot(df, 'age', output_dir='../outputs/plots')
    plot_boxplot(df, 'salary', output_dir='../outputs/plots')
    plot_scatter(df, 'age', 'salary', output_dir='../outputs/plots')
    bar_charts(df, 'department', 'salary', output_dir='../outputs/plots')
    plot_categorical(df, 'department', output_dir='../outputs/plots')
    plot_correlation_heatmap(df, output_dir='../outputs/plots')

    # INSIGHT: Distribusi age vs salary tidak menunjukkan korelasi kuat.
    # Ada data anomali seperti age ~10 dengan salary <50000 yang kemungkinan
    # noise dari random seed — perlu dicek sebelum modeling.
if __name__ == "__main__":
    main()