import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

class ShapExplainerLab:
    """
    Pipeline untuk mengekstraksi dan memvisualisasikan penjelasan model (Explainability)
    menggunakan SHAP (SHapley Additive exPlanations) pada model XGBoost.
    """
    def __init__(self, output_dir: str = "./outputs/shap_values"):
        # 1. Setup Output Directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 2. Load Data
        data = load_breast_cancer()
        self.X = pd.DataFrame(data.data, columns=data.feature_names)
        self.y = pd.Series(data.target) # 0 = Malignant, 1 = Benign
        
        # 3. Split Data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        # Reset index agar sejalan dengan array numpy dari SHAP
        self.X_test = self.X_test.reset_index(drop=True)
        self.y_test = self.y_test.reset_index(drop=True)
        
        # Inisialisasi atribut kosong untuk model dan explainer
        self.model = None
        self.explainer = None
        self.shap_values = None

    def train_model(self) -> None:
        """Melatih model XGBoost."""
        print(" Melatih model XGBoost...")
        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42,
            eval_metric="logloss",
            verbosity=0
        )
        self.model.fit(self.X_train, self.y_train)
        acc = self.model.score(self.X_test, self.y_test)
        print(f" Test accuracy: {acc:.4f}")

    def generate_shap_values(self) -> None:
        """Membuat explainer dan menghitung SHAP values untuk data test."""
        if self.model is None:
            raise ValueError("Model belum dilatih! Jalankan train_model() dulu.")
            
        print("\n🔍 Menghitung SHAP values dengan TreeExplainer...")
        # TreeExplainer sangat efisien untuk algoritma berbasis pohon (RF, XGBoost)
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(self.X_test)
        
        print(f" Base value (Rata-rata Prediksi): {self.explainer.expected_value:.4f}")

    def plot_summary(self) -> None:
        """Membuat dan menyimpan Summary Plot."""
        print("\n Membuat Summary Plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_test, 
            show=False
        )
        plt.title("SHAP Summary Plot — Breast Cancer Dataset", pad=15)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, "shap_summary_plot.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f" Summary plot tersimpan di: {save_path}")

    def analyze_specific_cases(self) -> dict:
        """Menganalisa dan mengembalikan detail dari 3 kasus prediksi menarik."""
        print("\nMenganalisa 3 Prediksi Individual...")
        
        proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Mencari indeks kasus menarik
        idx_high_conf = np.argmax(proba)                       # Paling yakin Benign (1)
        idx_low_conf = np.argmin(np.abs(proba - 0.5))          # Paling meragukan (Borderline)
        idx_malignant = np.argmin(proba)                       # Paling yakin Malignant (0)
        idx_wrong_pred = np.argmax(np.abs(proba - 0.5))        # Paling salah (Borderline)

        cases = {
            "High Confidence Benign": idx_high_conf,
            "Borderline Case": idx_low_conf,
            "High Confidence Malignant": idx_malignant,
            "Wrong Prediction": idx_wrong_pred
        }

        for case_name, idx in cases.items():
            actual_label = 'Benign' if self.y_test.iloc[idx] == 1 else 'Malignant'
            print(f"\n{'='*50}")
            print(f"Kasus: {case_name} (Index: {idx})")
            print(f" ↳ Aktual         : {actual_label}")
            print(f" ↳ Prediksi Prob  : {proba[idx]:.4f}")
            print(f" ↳ Base Value     : {self.explainer.expected_value:.4f}")
            
            # Menampilkan 5 fitur paling berpengaruh untuk pasien ini
            shap_row = self.shap_values[idx]
            top_indices = np.argsort(np.abs(shap_row))[::-1][:5]
            
            print(" ↳ Top 5 Fitur Berpengaruh:")
            for rank, fi in enumerate(top_indices, 1):
                direction = "Mendorong ke Benign (▲)" if shap_row[fi] > 0 else "Mendorong ke Malignant (▼)"
                feat_name = self.X_test.columns[fi]
                feat_val = self.X_test.iloc[idx, fi]
                shap_val = shap_row[fi]
                print(f"    {rank}. {feat_name:<25} | {direction:<27} | SHAP: {shap_val:+.4f} | Nilai Aktual: {feat_val:.2f}")

        return cases

    def plot_waterfall(self, idx: int, filename: str = "shap_waterfall_plot.png") -> None:
        """Membuat dan menyimpan Waterfall Plot untuk satu pasien spesifik menggunakan API modern."""
        print(f"\n Membuat Waterfall Plot untuk Index {idx}...")
        
        # Membungkus data ke dalam object shap.Explanation (API terbaru SHAP)
        shap_explanation = shap.Explanation(
            values=self.shap_values[idx],
            base_values=self.explainer.expected_value,
            data=self.X_test.iloc[idx].values,
            feature_names=self.X_test.columns.tolist()
        )

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_explanation, show=False)
        plt.title(f"Waterfall Plot — Pasien Index {idx}", pad=15)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, filename)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Waterfall plot tersimpan di: {save_path}")


# ── EXECUTION SCRIPT ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Inisialisasi JS opsional (hanya berguna jika dijalankan di Jupyter Notebook)
    # shap.initjs()
    
    lab = ShapExplainerLab()
    
    # Eksekusi Pipeline
    lab.train_model()
    lab.generate_shap_values()
    lab.plot_summary()
    
    # Analisa detail dan visualisasi individu
    cases_dict = lab.analyze_specific_cases()
    
    # Menggambar waterfall untuk kasus paling yakin
    # lab.plot_waterfall(idx=cases_dict["High Confidence Benign"], filename="waterfall_high_benign.png")
    lab.plot_waterfall(idx=cases_dict["Wrong Prediction"], filename="waterfall_wrong_prediction.png")