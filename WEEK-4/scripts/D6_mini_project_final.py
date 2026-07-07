import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# Menekan warning agar konsol tetap bersih
warnings.filterwarnings("ignore")

class MLPipelineEvaluator:
    """
    End-to-end Machine Learning Pipeline untuk Klasifikasi Biner.
    Menangani Data Loading, Baseline, Hyperparameter Tuning, dan Explainability (SHAP).
    """
    
    def __init__(self, output_dir: str = './outputs', random_state: int = 42):
        self.output_dir = output_dir
        self.random_state = random_state
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.models = {}          # Menyimpan model baseline
        self.tuned_models = {}    # Menyimpan model hasil tuning
        self.results = {}         # Menyimpan metrik evaluasi
        
    # ── 1. DATA PREPARATION ──────────────────────────────────────────────
    def load_and_prepare_data(self):
        """Memuat dataset Breast Cancer dan melakukan Stratified Split."""
        print("🚀 Memuat Data...")
        data = load_breast_cancer()
        self.X = pd.DataFrame(data.data, columns=data.feature_names)
        self.y = pd.Series(data.target)
        self.feature_names = data.feature_names
        
        # Simpan korelasi (Opsional: bisa dikembangkan jadi fungsi plot terpisah)
        self.correlations = self.X.assign(target=self.y).corr()['target'].drop('target').abs().sort_values(ascending=False)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state, stratify=self.y
        )
        print(f"Data siap. Train: {self.X_train.shape[0]} baris, Test: {self.X_test.shape[0]} baris.")

    # ── 2. BASELINE TRAINING ─────────────────────────────────────────────
    def train_baselines(self, models_dict: dict):
        """Melatih model baseline dengan Pipeline (Scaler + Model)."""
        print("\n🚀 Melatih Model Baseline...")
        for name, model in models_dict.items():
            pipe = Pipeline([
                ('scaler', StandardScaler()), 
                ('model', model)
            ])
            pipe.fit(self.X_train, self.y_train)
            self.models[name] = pipe
            
            y_proba = pipe.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_proba)
            print(f"   ↳ {name} Baseline AUC: {auc:.4f}")

    # ── 3. HYPERPARAMETER TUNING ─────────────────────────────────────────
    def tune_models(self, tuning_configs: dict):
        """Menjalankan Grid/Randomized Search berdasarkan konfigurasi."""
        print("\n🚀 Memulai Hyperparameter Tuning...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        for name, config in tuning_configs.items():
            print(f"   ↳ Tuning {name} menggunakan {config['method']}...")
            base_pipe = self.models.get(name)
            if not base_pipe:
                print(f"      Model {name} belum di-train di baseline. Melewati...")
                continue
                
            if config['method'] == 'GridSearchCV':
                search = GridSearchCV(
                    base_pipe, config['params'], cv=cv, scoring='roc_auc', 
                    n_jobs=-1, verbose=0
                )
            elif config['method'] == 'RandomizedSearchCV':
                search = RandomizedSearchCV(
                    base_pipe, config['params'], n_iter=config.get('n_iter', 20), 
                    cv=cv, scoring='roc_auc', n_jobs=-1, random_state=self.random_state, verbose=0
                )
            
            search.fit(self.X_train, self.y_train)
            self.tuned_models[name] = search.best_estimator_
            self.results[name] = {'best_cv_auc': search.best_score_, 'best_params': search.best_params_}
            print(f"      Best CV AUC: {search.best_score_:.4f}")

    # ── 4. EVALUASI FINAL ────────────────────────────────────────────────
    def evaluate_tuned_models(self):
        """Mengevaluasi model tuned di Test Set dan menyimpan kurva ROC."""
        print("\n📊 Evaluasi Model Tuned di Test Set...")
        plt.figure(figsize=(8, 6))
        
        summary_data = []
        for name, model in self.tuned_models.items():
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_proba)
            
            # Print Report
            print(f"\n{'='*40}\nModel: {name}\n{'='*40}")
            print(classification_report(self.y_test, y_pred, target_names=['Malignant', 'Benign']))
            
            # Kumpulkan data untuk summary
            base_auc = roc_auc_score(self.y_test, self.models[name].predict_proba(self.X_test)[:, 1])
            summary_data.append({'Model': name, 'Baseline AUC': base_auc, 'Tuned AUC': auc})
            
            # Plot ROC
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)

        # Plot Styling
        plt.plot([0,1], [0,1], 'k--', label='Random Chance')
        plt.title('ROC Curves - Tuned Models')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'roc_curves_tuned.png'), dpi=150)
        plt.close()
        
        # Tampilkan Summary Table
        summary_df = pd.DataFrame(summary_data).round(4)
        print("\n" + "="*45 + "\nBASELINE vs TUNED COMPARISON\n" + "="*45)
        print(summary_df.to_string(index=False))

    # ── 5. SHAP EXPLAINABILITY ───────────────────────────────────────────
    def explain_with_shap(self, target_model_name: str):
        """Menghasilkan SHAP Summary dan Waterfall plot untuk model tertentu."""
        print(f"\n🔬 Menganalisa {target_model_name} dengan SHAP...")
        pipe = self.tuned_models.get(target_model_name)
        if not pipe:
            print("Model belum di-tune!")
            return
            
        # Ekstrak Scaler dan Model dari Pipeline
        scaler = pipe.named_steps['scaler']
        model = pipe.named_steps['model']
        X_test_scaled = scaler.transform(self.X_test)
        
        # Inisialisasi Explainer
        explainer = shap.TreeExplainer(model)
        
        # Penanganan perbedaan API untuk output SHAP RF vs XGB
        if isinstance(model, RandomForestClassifier):
            shap_values = explainer.shap_values(X_test_scaled)[1] # Ambil class 1 (Benign)
            base_value = explainer.expected_value[1]
        else:
            shap_values = explainer.shap_values(X_test_scaled)
            base_value = explainer.expected_value

        # 1. Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_scaled, feature_names=self.feature_names, show=False)
        plt.title(f'SHAP Summary — {target_model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'shap_summary_{target_model_name}.png'), dpi=150)
        plt.close()

        # 2. Waterfall Plot (False Negative Analysis)
        y_pred = pipe.predict(self.X_test)
        fn_indices = np.where((self.y_test.values == 0) & (y_pred == 1))[0]
        
        if len(fn_indices) > 0:
            idx = fn_indices[0] # Ambil sampel pertama
            print(f"   ↳ Menganalisa False Negative pada Index {idx}")
            
            shap_explanation = shap.Explanation(
                values=shap_values[idx],
                base_values=base_value,
                data=self.X_test.iloc[idx].values,
                feature_names=self.feature_names
            )
            
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(shap_explanation, show=False)
            plt.title(f'Waterfall — False Negative (Model: {target_model_name})')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'shap_waterfall_fn_{target_model_name}.png'), dpi=150)
            plt.close()
        else:
            print(f"   ↳ Tidak ditemukan prediksi False Negative untuk {target_model_name}.")


# ── EKSEKUSI UTAMA (KONFIGURASI) ─────────────────────────────────────────────
if __name__ == "__main__":
    
    # [A] DEFINISI MODEL BASELINE
    MODELS = {
        'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
    }

    # [B] KONFIGURASI TUNING (Sangat mudah diubah/ditambah)
    TUNING_CONFIG = {
        'RandomForest': {
            'method': 'GridSearchCV',
            'params': {
                'model__n_estimators': [100, 200],
                'model__max_depth': [5, 10, None],
                'model__min_samples_split': [2, 5],
            }
        },
        'XGBoost': {
            'method': 'RandomizedSearchCV',
            'n_iter': 30,
            'params': {
                'model__n_estimators': randint(100, 300),
                'model__max_depth': randint(3, 8),
                'model__learning_rate': uniform(0.01, 0.2),
                'model__subsample': uniform(0.6, 0.4),
            }
        }
    }

    # [C] JALANKAN PIPELINE
    lab = MLPipelineEvaluator(output_dir='./outputs', random_state=42)
    lab.load_and_prepare_data()
    lab.train_baselines(MODELS)
    lab.tune_models(TUNING_CONFIG)
    lab.evaluate_tuned_models()
    
    # Ekstrak SHAP untuk model terbaik (misal: XGBoost)
    lab.explain_with_shap(target_model_name='XGBoost')
    print("\n Seluruh proses selesai. Silakan cek folder './outputs/'.")