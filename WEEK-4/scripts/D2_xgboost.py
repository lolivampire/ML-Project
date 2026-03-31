import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict

class XGBoostTitanicLab:
    """
    Pipeline eksperimen untuk mengevaluasi model XGBoost menggunakan dataset Titanic.
    Fokus pada: Baseline, Early Stopping, dan Learning Rate Trade-off.
    """

    def __init__(self):
        self.X_train, self.X_val, self.y_train, self.y_val = self._prepare_data()
        self.results = []

    def _prepare_data(self) -> Tuple:
        """Memuat, membersihkan, dan mengencode dataset Titanic."""
        df = sns.load_dataset('titanic')
        features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        target = 'survived'
        
        # Data Cleaning: Drop rows with missing values
        df = df[features + [target]].dropna()
        
        # Encoding Categorical Features
        le = LabelEncoder()
        df['sex'] = le.fit_transform(df['sex'])
        df['embarked'] = le.fit_transform(df['embarked'])
        
        X = df[features]
        y = df[target]
        
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train_baseline(self) -> XGBClassifier:
        """Melatih model XGBoost dengan parameter default sebagai baseline."""
        model = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            random_state=42, verbosity=0
        )
        model.fit(self.X_train, self.y_train)
        
        auc = roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])
        print(f"✅ Baseline XGBoost AUC: {auc:.4f}")
        return model

    def train_with_early_stopping(self, patience: int = 50) -> XGBClassifier:
        """
        Melatih model dengan Early Stopping untuk mencegah overfitting.
        Berhenti jika logloss pada data validasi tidak membaik selama 'patience' ronde.
        """
        model = XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            early_stopping_rounds=patience, random_state=42, verbosity=0
        )
        
        # Menyimpan history training untuk plotting nanti
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_train, self.y_train), (self.X_val, self.y_val)],
            verbose=False
        )
        
        print(f"✅ Early Stopping Selesai di Iterasi: {model.best_iteration}")
        print(f"✅ Best AUC: {roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1]):.4f}")
        return model

    def run_lr_tradeoff_experiment(self, configs: List[Dict]):
        """Membandingkan berbagai kombinasi Learning Rate dan n_estimators."""
        print(f"\n{'Config':<20} | {'Train AUC':<10} | {'Val AUC':<10} | {'Gap':<8}")
        print("-" * 55)
        
        for cfg in configs:
            model = XGBClassifier(
                n_estimators=cfg['n'], learning_rate=cfg['lr'],
                max_depth=3, eval_metric='logloss', random_state=42, verbosity=0
            )
            model.fit(self.X_train, self.y_train)
            
            train_auc = roc_auc_score(self.y_train, model.predict_proba(self.X_train)[:, 1])
            val_auc = roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])
    
            print(f"LR={cfg['lr']:<10} n={cfg['n']:<4} | {train_auc:<10.4f} | {val_auc:<10.4f} | {train_auc - val_auc:<8.4f}")
    
    def experiment_depth_impact(self, n_estimators: int = 50, lr: float = 0.3):  
        """Membandingkan berbagai kombinasi Learning Rate dan n_estimators."""
        # Bandingkan max_depth=3 (dari Task 3) dengan max_depth=6
        depths_to_test = [3, 6]

        for d in depths_to_test:
            model_depth = XGBClassifier(
                n_estimators=50,     # Sesuai permintaan (n=50)
                learning_rate=0.3,   # Sesuai permintaan (LR=0.3)
                max_depth=d,          # Variabel yang kita uji
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42,
                verbosity=0
            )
            
            # Menggunakan data dari instance self
            model_depth.fit(self.X_train, self.y_train)
            
            # Hitung score
            # Evaluasi
            tr_auc = roc_auc_score(self.y_train, model_depth.predict_proba(self.X_train)[:, 1])
            vl_auc = roc_auc_score(self.y_val, model_depth.predict_proba(self.X_val)[:, 1])
            gap = tr_auc - vl_auc
            
            label = f"Depth {d} (LR 0.01)"
            print(f"{label:<20} | {tr_auc:<10.4f} | {vl_auc:<10.4f} | {gap:<8.4f}")

    def plot_training_curves(self, model: XGBClassifier):
        """Visualisasi pergerakan logloss pada data Train vs Validation."""
        results = model.evals_result()
        train_loss = results['validation_0']['logloss']
        val_loss = results['validation_1']['logloss']
        
        plt.figure(figsize=(10, 5))
        sns.set_style("whitegrid")
        plt.plot(train_loss, label='Train Logloss', color='steelblue')
        plt.plot(val_loss, label='Val Logloss', color='tomato', linestyle='--')
        plt.axvline(model.best_iteration, color='green', linestyle=':', label='Best Iteration')
        
        plt.title('XGBoost Learning Curves (Early Stopping)')
        plt.xlabel('Boosting Rounds')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.show()

# ── EXECUTION ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    lab = XGBoostTitanicLab()
    
    # 1. Baseline
    lab.train_baseline()
    
    # 2. Early Stopping & Visualization
    model_es = lab.train_with_early_stopping()
    lab.plot_training_curves(model_es)
    
    # 3. Learning Rate Trade-off
    experimental_configs = [
        {'lr': 0.3, 'n': 50},
        {'lr': 0.1, 'n': 100},
        {'lr': 0.05, 'n': 300},
        {'lr': 0.01, 'n': 500}
    ]
    lab.run_lr_tradeoff_experiment(experimental_configs)

    # 4. jalankan dengan config terbaik
    # Menjalankan eksperimen depth yang baru saja kita buat
    print("\n" + "=" * 45)
    lab.experiment_depth_impact(n_estimators=50, lr=0.3)