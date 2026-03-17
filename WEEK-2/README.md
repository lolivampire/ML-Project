# 📍 Session Context — ML Engineer Journey

## Status
- **Week**: 02  **Day**: 04  **Phase**: 1 — ML Fundamentals
- **Tanggal**: [isi tanggal besok]

## Progress
| Week | Tema | Status |
|------|------|--------|
| W01 | Python & Data Foundations | ✅ DONE |
| W02 | Supervised Learning | 3/6 ✅ |
| W03 | Feature Engineering & Evaluation | ○ |
| W04 | XGBoost, RandomForest & Tuning | ○ |

## W02 — Detail Harian
| Hari | Topik | Status |
|------|-------|--------|
| D01 | Linear Regression mendalam | ✅ |
| D02 | Logistic Regression & decision boundary | ✅ |
| D03 | Decision Trees: splitting criteria | ✅ |
| D04 | Cross-validation | ○ |
| D05 | Bias-variance tradeoff | ○ |
| D06 | Mini project: Classification | ○ |

## 🧠 Key Learnings — W02D03
- Decision Tree membagi ruang fitur dengan if-else questions, bukan garis lurus
- Gini Impurity mengukur kebersihan node — range [0, 0.5] untuk 2 kelas
- Entropy mengukur ketidakpastian — range [0, 1] untuk 2 kelas
- Weighted Gini harus dihitung dari data aktual — jangan hardcode n_left/n_total
- Information Gain = Entropy(parent) - weighted avg Entropy(children)
- Gini vs Entropy: hasil hampir identik — Gini sedikit lebih cepat (tanpa log2)
- Greedy search: tree hanya pilih split terbaik saat ini, tidak melihat ke depan
- Gap besar train vs test accuracy = tanda overfitting — bukan soal nilai absolutnya
- Sweet spot depth bergantung dataset — depth=1 bisa mengalahkan depth=8
- max_depth adalah parameter pertama yang harus dikontrol untuk cegah overfitting

## ⚠️ Blur / Perlu Review
- Weighted Gini: jangan hardcode — ambil n dari data aktual
- Global scope training: semua model harus di-train di dalam fungsi, bukan level modul

## 📁 Output Files
- `week-02/scripts/decision_tree.py` ✅
- `week-02/outputs/plots/decision_tree_analysis.png` ✅
- `week-02/outputs/plots/depth_analysis2.png` ✅
- `W02D03_Decision_Trees.pdf` ✅
- GitHub: https://github.com/lolivampire/ML-Project

## 📌 Next Session
- **W02D04** — Cross-Validation (K-Fold, Stratified)
- **Preview**: Train-test split biasa punya kelemahan serius —
  hasilnya bergantung pada bagaimana data kebetulan terbagi.
  Cross-validation menyelesaikan ini dengan cara yang lebih sistematis.