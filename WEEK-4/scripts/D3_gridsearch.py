import time
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, Any
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, roc_auc_score

class GridSearchEvaluator:
    """
    Utility class untuk menjalankan GridSearchCV pada berbagai model,
    mencatat waktu komputasi, dan menganalisis gejala overfitting.
    """
    def __init__(self):
        # 1. Load Data
        data = load_breast_cancer()
        self.X, self.y = data.data, data.target
        self.feature_names = data.feature_names
        
        # 2. Split Data (Stratified sangat penting untuk dataset medis)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.best_models = {}
        self.search_results = {}

    def run_search(self, model_name: str, estimator: Any, param_grid: Dict) -> None:
        """
        Menjalankan GridSearchCV untuk model yang diberikan dan menyimpan hasilnya.
        """
        print(f"\n🚀 Memulai GridSearch untuk {model_name}...")
        
        # Inisialisasi GridSearch
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=5,                  # 5-Fold Cross Validation
            scoring='roc_auc',     # Optimasi berdasar kemampan membedakan kelas
            n_jobs=-1,             # Pakai semua core CPU
            verbose=1,             # Tampilkan progress bar
            return_train_score=True # WAJIB diaktifkan untuk deteksi overfit
        )

        # Eksekusi dan catat waktu
        start_time = time.time()
        grid_search.fit(self.X_train, self.y_train)
        elapsed_time = time.time() - start_time

        # Simpan hasil ke dalam instance
        self.best_models[model_name] = grid_search.best_estimator_
        self.search_results[model_name] = {
            'grid': grid_search,
            'time_s': elapsed_time,
            'n_combinations': len(grid_search.cv_results_['params'])
        }

        print(f"⏱ Waktu komputasi: {elapsed_time:.1f} detik")
        print(f"✅ Parameter Terbaik: {grid_search.best_params_}")
        print(f"📊 Best CV AUC: {grid_search.best_score_:.4f}")

    def analyze_top_results(self, model_name: str, top_n: int = 5) -> None:
        """
        Menampilkan Top N kombinasi parameter untuk melihat gap antara Train dan Test (CV).
        """
        if model_name not in self.search_results:
            print(f"Model {model_name} belum dijalankan.")
            return

        grid = self.search_results[model_name]['grid']
        results_df = pd.DataFrame(grid.cv_results_)

        cols_to_show = ['params', 'mean_test_score', 'mean_train_score', 'std_test_score']
        top_results = results_df.nlargest(top_n, 'mean_test_score')[cols_to_show]

        print(f"\n📋 Top {top_n} Kombinasi untuk {model_name}:")
        # Iterasi untuk print dictionary params lebih rapi
        for _, row in top_results.iterrows():
            gap = row['mean_train_score'] - row['mean_test_score']
            print(f"Params: {row['params']}")
            print(f"  ↳ CV AUC: {row['mean_test_score']:.4f} | Train AUC: {row['mean_train_score']:.4f} | Gap: {gap:.4f}")
            print("-" * 70)

    def evaluate_on_test(self, model_name: str) -> None:
        """
        Evaluasi final model terbaik menggunakan data Test yang belum pernah dilihat.
        """
        best_model = self.best_models.get(model_name)
        if not best_model:
            return

        y_pred = best_model.predict(self.X_test)
        y_proba = best_model.predict_proba(self.X_test)[:, 1]
        
        print(f"\n🏆 Final Test Report: {model_name}")
        print("=" * 55)
        print(classification_report(self.y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(self.y_test, y_proba):.4f}")


# ── EXECUTION SCRIPT ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    lab = GridSearchEvaluator()

    # 1. Setup Random Forest
    rf_estimator = RandomForestClassifier(random_state=42)
    rf_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, None],
        'min_samples_split': [2, 5]
    }
    lab.run_search("Random Forest", rf_estimator, rf_grid)

    # 2. Setup XGBoost
    xgb_estimator = xgb.XGBClassifier(random_state=42, eval_metric='auc', verbosity=0)
    xgb_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1]
    }
    lab.run_search("XGBoost", xgb_estimator, xgb_grid)

    # 3. Analisis Mendalam (Mengecek Overfitting)
    lab.analyze_top_results("Random Forest", top_n=3)
    lab.analyze_top_results("XGBoost", top_n=3)

    # 4. Final Evaluasi di Test Set
    lab.evaluate_on_test("Random Forest")
    lab.evaluate_on_test("XGBoost")
    

