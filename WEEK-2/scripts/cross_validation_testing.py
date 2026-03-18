from typing import Callable, Dict, List, Optional
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import numpy as np
from sklearn.datasets import load_breast_cancer

# ── CONSTANTS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

data = load_breast_cancer()
x, y = data.data, data.target

# ── FUNCTIONS ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def compare_cv_methods(
    x,
    y,
    random_states: Optional[List[int]] = None,
    test_size: float = 0.2,
    cv_splits: int = 5
) -> Dict[str, object]:
    """
    Bandingkan 3 metode CV: Holdout, KFold, StratifiedKFold.
    
    Args:
        x: Feature matrix.
        y: Target vector.
        random_states: List random_state untuk train_test_split. Default: [0, 10, 20, 30, 42].
        test_size: Ukuran test set.
        cv_splits: Jumlah splits untuk KFold/StratifiedKFold.
    
    Returns:
        Dictionary dengan hasil per random_state dan overall summary.
    """
    if random_states is None:
        random_states = [0, 10, 20, 30, 42]
    
    model = DecisionTreeClassifier(random_state=42)
    results = {}
    
    # Simpan means untuk overall summary
    holdout_means, kfold_means, skfold_means = [], [], []
    
    for rs in random_states:
        # 1. Holdout (Train-Test Split)
        X_train, X_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=rs
        )
        m = clone(model)  # fresh model tiap iterasi
        m.fit(X_train, y_train)
        holdout_score = m.score(X_test, y_test)
        holdout_means.append(holdout_score)
        
        # 2. KFold
        kf = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
        kf_scores = cross_val_score(clone(model), X_train, y_train, cv=kf, scoring='accuracy')
        kfold_means.append(kf_scores.mean())
        
        # 3. StratifiedKFold
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        skf_scores = cross_val_score(clone(model), X_train, y_train, cv=skf, scoring='accuracy')
        skfold_means.append(skf_scores.mean())
        
        # Simpan hasil per random_state
        results[rs] = {
            "holdout": {"scores": [holdout_score], "mean": holdout_score, "std": 0.0},
            "kfold": {"scores": kf_scores.tolist(), "mean": kf_scores.mean(), "std": kf_scores.std()},
            "stratified_kfold": {"scores": skf_scores.tolist(), "mean": skf_scores.mean(), "std": skf_scores.std()}
        }
    
    # Overall summary
    overall = {
        "holdout": {"mean": np.mean(holdout_means), "std": np.std(holdout_means)},
        "kfold": {"mean": np.mean(kfold_means), "std": np.std(kfold_means)},
        "stratified_kfold": {"mean": np.mean(skfold_means), "std": np.std(skfold_means)}
    }
    
    return {"by_random_state": results, "overall": overall}

def print_results(results: Dict[str, object]) -> None:
    """Print hasil dengan format rapi."""
    print("\n" + "="*70)
    print("HASIL PER RANDOM_STATE")
    print("="*70)
    
    for rs, data in results["by_random_state"].items():
        print(f"\nRandom State = {rs}")
        print("-" * 70)
        
        for method, scores_data in data.items():
            scores = scores_data["scores"]
            mean = scores_data["mean"]
            std = scores_data["std"]
            print(f"  {method.upper():20} | Scores: {[f'{s:.4f}' for s in scores]}")
            print(f"  {' '*20} | Mean: {mean:.4f}, Std: {std:.4f}")
    
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    for method, stats in results["overall"].items():
        mean = stats["mean"]
        std = stats["std"]
        print(f"{method.upper():20} | Mean: {mean:.4f}, Std: {std:.4f}")

# ── MAIN ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Main function."""
    
    results = compare_cv_methods(x, y)
    print_results(results)

    #output
# ======================================================================
# HASIL PER RANDOM_STATE
# ======================================================================

# Random State = 0
# ----------------------------------------------------------------------
#   HOLDOUT              | Scores: ['0.9123']
#                        | Mean: 0.9123, Std: 0.0000
#   KFOLD                | Scores: ['0.9560', '0.9011', '0.9121', '0.9231', '0.9341']
#                        | Mean: 0.9253, Std: 0.0189
#   STRATIFIED_KFOLD     | Scores: ['0.9121', '0.9560', '0.9341', '0.9121', '0.8791']
#                        | Mean: 0.9187, Std: 0.0256

# Random State = 10
# ----------------------------------------------------------------------
#   HOLDOUT              | Scores: ['0.8947']
#                        | Mean: 0.8947, Std: 0.0000
#   KFOLD                | Scores: ['0.9341', '0.9341', '0.9231', '0.9231', '0.9341']
#                        | Mean: 0.9297, Std: 0.0054
#   STRATIFIED_KFOLD     | Scores: ['0.9341', '0.9670', '0.9011', '0.9341', '0.9011']
#                        | Mean: 0.9275, Std: 0.0247

# Random State = 20
# ----------------------------------------------------------------------
#   HOLDOUT              | Scores: ['0.9386']
#                        | Mean: 0.9386, Std: 0.0000
#   KFOLD                | Scores: ['0.9231', '0.9341', '0.9011', '0.9341', '0.9121']
#                        | Mean: 0.9209, Std: 0.0128
#   STRATIFIED_KFOLD     | Scores: ['0.9011', '0.8901', '0.9451', '0.9231', '0.9341']
#                        | Mean: 0.9187, Std: 0.0204

# Random State = 30
# ----------------------------------------------------------------------
#   HOLDOUT              | Scores: ['0.9298']
#                        | Mean: 0.9298, Std: 0.0000
#   KFOLD                | Scores: ['0.9451', '0.8901', '0.9231', '0.9231', '0.9121']
#                        | Mean: 0.9187, Std: 0.0179
#   STRATIFIED_KFOLD     | Scores: ['0.9670', '0.9451', '0.8901', '0.9011', '0.9121']
#                        | Mean: 0.9231, Std: 0.0287

# Random State = 42
# ----------------------------------------------------------------------
#   HOLDOUT              | Scores: ['0.9474']
#                        | Mean: 0.9474, Std: 0.0000
#   KFOLD                | Scores: ['0.9011', '0.8571', '0.8901', '0.9121', '0.8681']
#                        | Mean: 0.8857, Std: 0.0204
#   STRATIFIED_KFOLD     | Scores: ['0.9780', '0.9451', '0.9011', '0.9451', '0.9011']
#                        | Mean: 0.9341, Std: 0.0295

# ======================================================================
# OVERALL SUMMARY
# ======================================================================
# HOLDOUT              | Mean: 0.9246, Std: 0.0189
# KFOLD                | Mean: 0.9160, Std: 0.0156
# STRATIFIED_KFOLD     | Mean: 0.9244, Std: 0.0058

    # ANALISIS:
# 1. Berapa range (max-min) dari 5 hasil train-test split?
# jawab : Holdout scores: [0.9123, 0.8947, 0.9386, 0.9298, 0.9474], dengan Range : 0.9474 - 0.8947 = 0.0526 (artinya hanya dengan mengganti cara membagi data, akurasi bisa berbeda hingga ~5%. Inilah yang membuat holdout tidak reliable.)
# 2. Std mana yang lebih kecil — KFold atau StratifiedKFold?
# jawab : stratified kfold std: 0.0058 < kfold std: 0.0156, jadi stratified lebih kecil std nya, berarti proporsi kelas di setiap fold dijaga konsisten, sehingga variasi hasil antar split lebih kecil.
# 3. Metode mana yang paling bisa dipercaya hasilnya, dan kenapa?
# jawab : StratifiedKFold karena StratifiedKFold menggunakan menggunakan semua data sebagai test secara bergantian sehingga tidak bergantung pada satu split keberuntungan. Kedua, menjaga proporsi kelas di setiap fold — hasilnya lebih stabil dan tidak terpengaruh distribusi kelas yang tidak merata.

if __name__ == "__main__":
    main()
