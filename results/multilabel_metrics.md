# Multi-Label Toxicity Sub-Type Evaluation (Held-Out Test Set: 31,915 Comments)

- **Classifier:** `OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=500, random_state=42))`
- **Features:** v2 `FeatureUnion` (25,000 features: Word (1,2) + Char (3,5))
- **Aggregate Scores:** Micro-F1 = **68.98%** | Macro-F1 = **56.79%**

| Toxicity Category | Test Positives | Prevalence | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `toxic` | 3,046 | 9.54% | 64.77% | 88.41% | **74.76%** | 0.9778 | 0.8795 |
| `severe_toxic` | 305 | 0.96% | 28.51% | 87.87% | **43.05%** | 0.9854 | 0.4387 |
| `obscene` | 1,662 | 5.21% | 69.50% | 90.91% | **78.78%** | 0.9911 | 0.8997 |
| `threat` | 98 | 0.31% | 24.83% | 72.45% | **36.98%** | 0.9896 | 0.4961 |
| `insult` | 1,576 | 4.94% | 56.01% | 89.34% | **68.85%** | 0.9831 | 0.8034 |
| `identity_hate` | 259 | 0.81% | 24.97% | 82.24% | **38.31%** | 0.9831 | 0.5040 |
