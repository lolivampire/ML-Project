import time
import warnings
import numpy as np
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Mengabaikan UserWarning dari sklearn terkait joblib/parallel processing
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

class AdvancedTuningLab:
    """
    Pipeline eksperimen untuk Hyperparameter Tuning lanjutan menggunakan 
    RandomizedSearchCV dan teknik Early Stopping.
    """
    
    def __init__(self):
        # 1. Load Data
        data = load_breast_cancer()
        self.X, self.y = data.data, data.target
        
        # 2. Split Data (Konsistensi dengan W04D03)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        print(f"✅ Data dimuat. Ukuran Train: {self.X_train.shape[0]}, Test: {self.X_test.shape[0]}")

    def tune_random_forest(self, n_iter: int = 30):
        """Tuning Random Forest menggunakan distribusi probabilitas parameter."""
        print(f"\n🚀 Memulai RandomizedSearch RF ({n_iter} iterasi)...")
        
        # Definisi Distribusi Parameter
        # scipy.stats uniform(loc, scale) -> range: [loc, loc + scale]
        param_dist = {
            'n_estimators': randint(50, 500),         # Acak antara 50 - 499
            'max_depth': randint(3, 20),              # Acak antara 3 - 19
            'min_samples_split': randint(2, 20),
            'min_samples_leaf': randint(1, 10),
            'max_features': uniform(0.3, 0.7),        # Float acak 0.3 - 1.0
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            estimator=rf,
            param_distributions=param_dist,
            n_iter=n_iter,           # Hanya mencoba n kombinasi acak
            cv=5,
            scoring='roc_auc',
            return_train_score=True, # Wajib untuk deteksi overfit
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        start = time.time()
        random_search.fit(self.X_train, self.y_train)
        
        self._print_search_results("Random Forest", random_search, time.time() - start)

    def demo_xgb_early_stopping(self):
        """Mendemonstrasikan Early Stopping XGBoost dengan Validation Set terpisah."""
        print("\n🚀 Memulai XGBoost dengan Early Stopping...")
        
        # Early Stopping butuh Validation Set murni (tidak bisa pakai CV bawaan)
        X_tr, X_val, y_tr, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )

        # Catatan: Di XGBoost versi baru (>=1.6), early_stopping_rounds dimasukkan ke constructor
        xgb_model = xgb.XGBClassifier(
            n_estimators=500,           # Set batas atas sangat tinggi
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            early_stopping_rounds=20,    # Berhenti jika 20 ronde stagnan
            random_state=42,
            n_jobs=-1
        )

        xgb_model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr), (X_val, y_val)],
            verbose=50  # Print log setiap 50 iterasi
        )
        n = xgb_model.n_estimators
        best_iter = xgb_model.best_iteration
        print(f"\n✅ Early Stopping Aktif!")
        print(f"📊 Iterasi Terbaik: {best_iter} (dari maksimal {n})")
        print(f"📊 AUC Terbaik: {xgb_model.best_score:.4f}")
        print(f"⏱ Menghemat waktu komputasi untuk {n - best_iter} pohon.")
        print(f"Model berhenti di iterasi ke-{best_iter} dari maksimum {n}. Artinya terdapat {n - best_iter} iterasi dihemat. Ini menunjukkan bahwa tanpa early stopping, model sudah mulai overfit sejak iterasi ke-{best_iter}.")

    def tune_xgboost(self, n_iter: int = 20):
        """Tuning parameter XGBoost selain n_estimators menggunakan RandomizedSearch."""
        print(f"\n🚀 Memulai RandomizedSearch XGBoost ({n_iter} iterasi)...")
        
        param_dist = {
            'learning_rate': uniform(0.01, 0.29),   # Range 0.01 - 0.30
            'max_depth': randint(3, 10),
            'subsample': uniform(0.6, 0.4),         # Range 0.6 - 1.0
            'colsample_bytree': uniform(0.5, 0.5),  # Range 0.5 - 1.0
            'min_child_weight': randint(1, 10),
        }

        xgb_cv = xgb.XGBClassifier(
            n_estimators=150,       # FIXED n_estimators karena di dalam CV sulit pakai early stopping
            eval_metric='auc',
            random_state=42,
            n_jobs=1                # Set 1 agar tidak bentrok dengan n_jobs=-1 di RandomizedSearch
        )

        random_search = RandomizedSearchCV(
            estimator=xgb_cv,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=5,
            scoring='roc_auc',
            return_train_score=True,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        start = time.time()
        random_search.fit(self.X_train, self.y_train)
        
        self._print_search_results("XGBoost", random_search, time.time() - start)

    def _print_search_results(self, model_name: str, search_obj: RandomizedSearchCV, time_taken: float):
        """Helper internal untuk mencetak hasil dari RandomizedSearchCV agar rapi."""
        best_idx = search_obj.best_index_
        train_auc = search_obj.cv_results_['mean_train_score'][best_idx]
        val_auc = search_obj.cv_results_['mean_test_score'][best_idx]
        gap = train_auc - val_auc

        print(f"\n⏱ Waktu {model_name} RandomizedSearch: {time_taken:.1f} detik")
        print(f"✅ Parameter Terbaik: {search_obj.best_params_}")
        print(f"📊 Train AUC: {train_auc:.4f} | Val AUC (CV): {val_auc:.4f} | Gap: {gap:.4f}")

    def benchmark_search_methods(self, n_iter_random):
        """
        Mengadu kecepatan dan efisiensi GridSearchCV vs RandomizedSearchCV 
        pada algoritma XGBoost.
        """
        print("\n🏁 Memulai Benchmark: GridSearch vs RandomizedSearch...")
        print("Silakan tunggu, ini mungkin memakan waktu beberapa saat...\n")
        
        # Model dasar yang sama untuk kedua metode
        xgb_for_grid   = xgb.XGBClassifier(eval_metric='auc', random_state=42, n_jobs=1)
        xgb_for_random = xgb.XGBClassifier(eval_metric='auc', random_state=42, n_jobs=1)
        
        # ── 1. GRID SEARCH (Brute Force) ──
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
        } # Total kombinasi: 3 x 3 x 3 = 27
        
        grid_search = GridSearchCV(
            estimator=xgb_for_grid, param_grid=param_grid, 
            cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
        )
        
        start_grid = time.time()
        grid_search.fit(self.X_train, self.y_train)
        time_grid = time.time() - start_grid
        
        # ── 2. RANDOMIZED SEARCH (Smart Sampling) ──
        param_dist = {
            'n_estimators': randint(50, 400),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.29),
        }
        
        random_search = RandomizedSearchCV(
            estimator=xgb_for_random, param_distributions=param_dist,
            n_iter=n_iter_random, # Hanya mencoba n kombinasi!
            cv=5, scoring='roc_auc', random_state=42, n_jobs=-1, verbose=0
        )
        
        start_random = time.time()
        random_search.fit(self.X_train, self.y_train)
        time_random = time.time() - start_random
        
        # ── 3. REPORTING ──
        print(f"📊 {' HASIL BENCHMARK ':=^70}")
        print(f"{'Metode':<20} | {'Waktu (s)':<10} | {'Kombinasi Dicoba':<18} | {'Best CV AUC':<10}")
        print("-" * 70)
        
        total_grid_comb = len(grid_search.cv_results_['params'])
        print(f"{'GridSearchCV':<20} | {time_grid:<10.1f} | {total_grid_comb:<18} | {grid_search.best_score_:.4f}")
        print(f"{'RandomizedSearchCV':<20} | {time_random:<10.1f} | {n_iter_random:<18} | {random_search.best_score_:.4f}")
        print("-" * 70)
        
        # Kesimpulan Analitis
        speedup = time_grid / time_random
        print(f"\n💡 KESIMPULAN:")
        print(f"RandomizedSearch dengan n_iter={n_iter_random} menemukan AUC {random_search.best_score_} dalam {time_random:.1f} detik. GridSearch menemukan AUC {grid_search.best_score_} dalam {time_grid:.1f} detik. Kesimpulan:")
        if time_random < time_grid:
            print(f"RandomizedSearch lebih cepat ({time_grid/time_random:.1f}x)")
        else:
            print(f"GridSearch lebih cepat di dataset kecil ini ({time_random/time_grid:.1f}x lebih lambat).")
            print(f"Catatan: RandomizedSearch unggul di dataset besar + grid parameter yang lebih luas.")
    

# ── EXECUTION SCRIPT ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    lab = AdvancedTuningLab()
    
    # 1. RandomizedSearch pada Random Forest
    lab.tune_random_forest(n_iter=20)
    
    # 2. Demonstrasi Early Stopping XGBoost murni
    lab.demo_xgb_early_stopping()
    
    # 3. RandomizedSearch pada parameter struktural XGBoost
    lab.tune_xgboost(n_iter=20)

    # Jalankan benchmark
    lab.benchmark_search_methods(n_iter_random=20)