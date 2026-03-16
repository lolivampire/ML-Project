# 📍 Session Context — ML Engineer Journey
## Status
- **Week**: 02  **Day**: 03  **Phase**: 1 — ML Fundamentals
- **Tanggal**: [17/03/2026]
## Progress
| Week | Tema | Status |
|------|------|--------|
| W01 | Python & Data Foundations | ✅ DONE |
| W02 | Supervised Learning | 2/6 ✅ |
| W03 | Feature Engineering & Evaluation | ○ |
| W04 | XGBoost, RandomForest & Tuning | ○ |
## W02 — Detail Harian
| Hari | Topik | Status |
|------|-------|--------|
| D01 | Linear Regression mendalam | ✅ |
| D02 | Logistic Regression & decision boundary | ✅ |
| D03 | Decision Trees: splitting criteria | ○ |
| D04 | Cross-validation | ○ |
| D05 | Bias-variance tradeoff | ○ |
| D06 | Mini project: Classification | ○ |
## 🧠 Key Learnings — W02D02
- Linear Regression tidak cocok klasifikasi — output tidak terbatas
- Sigmoid mengkompresi z ke [0,1] — sigma(0) = 0.5, invers dari logit
- Log-odds: model linear sebenarnya memprediksi ln(p/1-p)
- Decision boundary muncul di z = 0, bukan digambar manual
- error = y_pred - y adalah gradient BCE loss terhadap z
- 100% accuracy di data sintetis bukan indikator model hebat
- Bias mendekati 0 ketika kelas simetris terhadap origin
## ⚠️ Blur / Perlu Review
-
## 📁 Output Files
- `week-02/scripts/logistic_regression.py` ✅
- `week-02/outputs/decision_boundary.png` ✅
- GitHub: https://github.com/lolivampire/ML-Project
## 📌 Next Session
- **W02D03** — Decision Trees: Splitting Criteria
- **Preview**: Gini impurity vs Entropy — bagaimana tree memilih
  fitur terbaik untuk split, dan kapan ia mulai overfitting.