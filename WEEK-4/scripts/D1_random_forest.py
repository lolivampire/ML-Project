import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from typing import List, Dict, Any
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

class RandomForestLab:
    """
    Laboratorium untuk eksperimen Random Forest: membandingkan dengan Decision Tree,
    menganalisis Feature Importance, dan optimasi n_estimators.
    """
    
    def __init__(self, n_samples: int = 2000, n_features: int = 15):
        self.X, self.y = self._generate_data(n_samples, n_features)
        self.feature_names = [f"feature_{i:02d}" for i in range(n_features)]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.results = {}

    def _generate_data(self, n, f) -> tuple:
        """Helper untuk membuat dataset klasifikasi sintetis."""
        X, y = make_classification(
            n_samples=n, n_features=f, n_informative=8, 
            n_redundant=4, random_state=42
        )
        return pd.DataFrame(X), y

    def run_baseline_comparison(self):
        """Membandingkan Single Decision Tree vs Random Forest Default."""
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
        }
        
        print(f"{'Model':<15} | {'Train Acc':<10} | {'Test Acc':<10} | {'AUC':<10}")
        print("-" * 55)
        
        for name, clf in models.items():
            clf.fit(self.X_train, self.y_train)
            train_acc = clf.score(self.X_train, self.y_train)
            test_acc = clf.score(self.X_test, self.y_test)
            auc = roc_auc_score(self.y_test, clf.predict_proba(self.X_test)[:, 1])
            
            print(f"{name:<15} | {train_acc:<10.3f} | {test_acc:<10.3f} | {auc:<10.3f}")
            self.results[name] = clf

    def plot_feature_importance(self):
        """Visualisasi fitur mana yang dianggap paling penting oleh RF."""
        rf = self.results.get("Random Forest")
        if not rf:
            print("Jalankan run_baseline_comparison terlebih dahulu.")
            return

        importances = pd.Series(rf.feature_importances_, index=self.feature_names)
        importances = importances.sort_values(ascending=False)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances.values, y=importances.index, hue=importances.index, 
                    palette="viridis", legend=False)
        plt.title("Feature Importance - Random Forest (Gini Importance)")
        plt.xlabel("Importance Score")
        plt.show()

    def experiment_n_estimators(self, estimator_range: List[int]):
        """Menganalisis dampak jumlah pohon terhadap AUC dan waktu training."""
        stats = []
        
        for n in estimator_range:
            start_time = time.time()
            model = RandomForestClassifier(n_estimators=n, n_jobs=-1, random_state=42)
            model.fit(self.X_train, self.y_train)
            elapsed = time.time() - start_time
            
            auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
            stats.append({"n_estimators": n, "auc": auc, "time": elapsed})
        
        self._plot_n_estimator_results(pd.DataFrame(stats))

    def _plot_n_estimator_results(self, df: pd.DataFrame):
        fig, ax1 = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df, x="n_estimators", y="auc", ax=ax1, marker='o', label='AUC', color='b')
        ax1.set_ylabel("AUC Score", color='b')
        
        ax2 = ax1.twinx()
        sns.lineplot(data=df, x="n_estimators", y="time", ax=ax2, marker='s', label='Time', color='r', linestyle='--')
        ax2.set_ylabel("Training Time (s)", color='r')
        
        plt.title("Trade-off: Akurasi vs Waktu Komputasi")
        plt.show()

# ── EXECUTION ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lab = RandomForestLab()
    
    # 1. Bandingkan Bias-Variance (DT vs RF)
    lab.run_baseline_comparison()
    
    # 2. Analisis Fitur Relevan
    lab.plot_feature_importance()
    
    # 3. Cari 'Sweet Spot' jumlah pohon
    lab.experiment_n_estimators([1, 10, 50, 100, 200, 500])