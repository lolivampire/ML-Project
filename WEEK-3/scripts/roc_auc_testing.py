import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, 
    classification_report, roc_auc_score, f1_score
)

class ModelEvaluator:
    """
    Utility class untuk mengevaluasi model klasifikasi biner,
    fokus pada analisis ROC-AUC dan optimalisasi threshold.
    """
    
    def __init__(self, y_true: np.ndarray, y_proba: np.ndarray):
        self.y_true = y_true
        self.y_proba = y_proba
        self.fpr, self.tpr, self.thresholds_roc = roc_curve(y_true, y_proba)
        self.roc_auc = auc(self.fpr, self.tpr)

    def plot_roc_curve(self, default_threshold: float = 0.5, save_path: Optional[str] = None):
        """Visualisasi kurva ROC dengan penanda threshold tertentu."""
        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")
        
        plt.plot(self.fpr, self.tpr, color='steelblue', lw=2, 
                 label=f'ROC curve (AUC = {self.roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

        # Tandai titik threshold default
        idx = np.argmin(np.abs(self.thresholds_roc - default_threshold))
        plt.scatter(self.fpr[idx], self.tpr[idx], color='red', s=100, zorder=5,
                    label=f'Threshold {default_threshold} (FPR={self.fpr[idx]:.2f}, TPR={self.tpr[idx]:.2f})')

        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Recall)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def find_best_threshold(self, metric: str = 'f1') -> float:
        """
        Mencari threshold optimal berdasarkan metrik tertentu.
        Args:
            metric: 'f1' untuk F1-Score atau 'youden' untuk Youden's J Statistic.
        """
        if metric == 'f1':
            p, r, t = precision_recall_curve(self.y_true, self.y_proba)
            # Hindari division by zero dengan 1e-9
            f1_scores = (2 * p * r) / (p + r + 1e-9)
            best_threshold = t[np.argmax(f1_scores[:-1])]
        else: # Youden's J: TPR - FPR
            j_stat = self.tpr - self.fpr
            best_threshold = self.thresholds_roc[np.argmax(j_stat)]
            
        return best_threshold
    
    def run_threshold_experiment(self):
        """
        Menjalankan eksperimen threshold dari 0.1 ke 0.9 dan menampilkan
        hasilnya dalam format tabel yang rapi.
        """
        thresholds = np.arange(0.1, 1.0, 0.1)

        print("\n── Eksperimen threshold ──")
        print(f"\n{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
        print("-" * 50)
        
        results = []
        for thresh in thresholds:
            y_pred = (self.y_proba >= thresh).astype(int)
            
            # Menghitung metrik secara manual untuk threshold spesifik
            # Menggunakan f1_score dari sklearn untuk konsistensi
            from sklearn.metrics import precision_score, recall_score
            
            p = precision_score(self.y_true, y_pred, zero_division=0)
            r = recall_score(self.y_true, y_pred, zero_division=0)
            f1 = f1_score(self.y_true, y_pred, zero_division=0)
            
            results.append((thresh, f1))
            print(f"{thresh:>10.1f} | {p:>10.3f} | {r:>10.3f} | {f1:>10.3f}")
        
        # Mencari F1 tertinggi dari hasil loop
        best_t, best_f1 = max(results, key=lambda x: x[1])
        
        print("-" * 50)
        print(f"Threshold {best_t:.1f} memberikan F1 tertinggi ({best_f1:.3f})")
    
    def plot_pr_curve(self, save_path: Optional[str] = None):
        """
        Visualisasi Precision-Recall Curve dan menandai titik F1 optimal.
        
        Komentar Penting: 
        Pilih PR Curve dibanding ROC Curve ketika dataset sangat imbalanced (kelas positif sangat sedikit).
        ROC
        """
        precisions, recalls, thresholds = precision_recall_curve(self.y_true, self.y_proba)
        # Hitung F1 untuk setiap threshold untuk mencari titik optimal
        # f1 = 2 * (P * R) / (P + R)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

        plt.figure(figsize=(8, 6))
        sns.set_style("whitegrid")
        # Plot PR Curve
        plt.plot(recalls, precisions, color='darkorange', lw=2, 
                 label=f'PR Curve (Best F1 = {best_f1:.3f})')
        
        # Tandai titik optimal
        plt.scatter(recalls[best_idx], precisions[best_idx], color='black', s=100, zorder=5,
                    label=f'Optimal Threshold: {best_threshold:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def print_report(self, threshold: float = 0.5):
        """Cetak classification report pada threshold tertentu."""
        y_pred = (self.y_proba >= threshold).astype(int)
        print(f"\n[ Evaluation Report | Threshold: {threshold:.3f} ]")
        print("-" * 50)
        print(classification_report(self.y_true, y_pred))
        print(f"ROC AUC Score: {self.roc_auc:.4f}")


def prepare_data(n_samples: int = 1000) -> Tuple:
    """Generate imbalanced dataset dan split ke train/test."""
    X, y = make_classification(
        n_samples=n_samples, n_features=10, 
        weights=[0.8, 0.2], random_state=42
    )
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


# ── MAIN EXECUTION ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Setup Data
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. Train Model
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]

    # 3. Evaluation
    evaluator = ModelEvaluator(y_test, y_scores)
    
    # Visualisasi
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs/roc_auc/')
    os.makedirs(output_dir, exist_ok=True)
    evaluator.plot_roc_curve(default_threshold=0.5, save_path=os.path.join(output_dir, 'roc_curve.png'))
    
    # Tuning
    best_thresh = evaluator.find_best_threshold(metric='f1')
    
    # Perbandingan Report
    print("BASELINE (Threshold 0.5):")
    evaluator.print_report(threshold=0.5)
    
    print("\nOPTIMIZED (Best F1 Threshold):")
    evaluator.print_report(threshold=best_thresh)

    # Eksperimen Threshold
    evaluator.run_threshold_experiment()

    # Precision-Recall Curve
    evaluator.plot_pr_curve(save_path=os.path.join(output_dir, 'pr_curve.png'))
    