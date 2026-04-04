import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, accuracy_score
)

class TitanicEvaluator:
    """
    Class untuk memproses dataset Titanic dan mengevaluasi performa model
    dengan fokus pada metrik Precision, Recall, dan F1-Score.
    """
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=500, random_state=42)
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_and_preprocess(self):
        """Memuat data Titanic dan melakukan preprocessing sederhana."""
        # Mengambil data dari seaborn library
        data = sns.load_dataset('titanic')
        
        # Preprocessing: Pilih fitur relevan & hapus missing values
        cols = ['survived', 'pclass', 'sex', 'age', 'fare', 'sibsp', 'parch']
        self.df = data[cols].dropna().copy()
        
        # Encoding kategori: male=0, female=1
        self.df['sex'] = self.df['sex'].map({'male': 0, 'female': 1})
        
        X = self.df.drop('survived', axis=1)
        y = self.df['survived']
        
        # Split data dengan stratify agar proporsi 'survived' seimbang di kedua set
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Dataset loaded. Training size: {len(self.X_train)}, Test size: {len(self.X_test)}")

    def train(self):
        """Melatih model Logistic Regression."""
        self.model.fit(self.X_train, self.y_train)
        return self.model.predict(self.X_test)

    def display_results(self, y_pred):
        """Menampilkan evaluasi mendalam dan interpretasi bisnis."""
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("\n" + "="*50)
        print(f"{'DETEKSI SURVIVAL TITANIC: EVALUASI':^50}")
        print("="*50)
        
        # Visualisasi sederhana Confusion Matrix
        print(f"{'':<15} | Pred: No (0) | Pred: Yes (1)")
        print(f"{'-'*45}")
        print(f"{'Actual: No (0)':<15} | {tn:^12} | {fp:^13} (TN | FP)")
        print(f"{'Actual: Yes (1)':<15} | {fn:^12} | {tp:^13} (FN | TP)")
        
        # Metrik Utama
        print("\n" + "-"*50)
        print(f"Accuracy  : {accuracy_score(self.y_test, y_pred):.3f}")
        print(f"Precision : {precision_score(self.y_test, y_pred):.3f} (Kemampuan deteksi selamat tanpa salah tuduh)")
        print(f"Recall    : {recall_score(self.y_test, y_pred):.3f} (Kemampuan menangkap semua orang yang selamat)")
        print(f"F1-Score  : {f1_score(self.y_test, y_pred):.3f} (Keseimbangan Precision & Recall)")
        
        # Interpretasi Bisnis/Konteks
        print("\n" + "="*50)
        print("INTERPRETASI MODEL")
        print("="*50)
        print(f"* Ada {fn} orang yang diprediksi tewas padahal selamat (Missed opportunities).")
        print(f"* Ada {fp} orang yang diprediksi selamat padahal tewas (False alarms).")
        print("\nSKLEARN REPORT:")
        print(classification_report(self.y_test, y_pred, target_names=['Died', 'Survived']))

# ── RUNTIME ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluator = TitanicEvaluator()
    evaluator.load_and_preprocess()
    predictions = evaluator.train()
    evaluator.display_results(predictions)

    #interpretasi bisnis — dalam konteks Titanic, mana yang lebih penting, precision atau recall? Jelaskan alasanmu.
    # dalam konteks titanic, lebih penting recall karena memaksimalkan TP dan meminimalkan FN, karena tujuan utamanya Menemukan dan menyelamatkan setiap orang yang memiliki peluang bertahan hidup.

    # Model kamu punya Precision = 0.90 dan Recall = 0.30. F1 Score-nya berapa? Dan apa artinya secara praktis — apakah model ini "bagus"?
    # f1 score = 0.45, 2 x(0.90 * 0.30) / (0.90 + 0.30) = 0.45, artinya nilai model ini cukup buruk karena nilainya tidak sampai 0,5 model hanya menebak kurang dari setengah dari data yang sebenarnya.
    #Di kasus prediksi apakah penumpang Titanic selamat (1) atau tidak (0) — kamu sebagai ML engineer, mana yang kamu pilih untuk dioptimasi: precision atau recall? Kenapa?
    # recall lebih penting karena tujuan utamanya Menemukan dan menyelamatkan setiap orang yang memiliki peluang bertahan hidup daripada model prediksi akan selamat, dan mencari berapa banyak yang benar-benar selamat, hal ini termasuk pembosoran karena bisa saja terjadi tim mencari orang yang dianggap selamat namun sebenarnya sudah tiada.