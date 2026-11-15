# 5-Fold Stratified Cross-Validation Benchmark (v2 FeatureUnion)

- **Training Split:** 127,656 comments (80% of 159,571; stratified `random_state=42`).
- **Features:** `FeatureUnion` (`word (1,2) max=10,000` + `char_wb (3,5) max=15,000`), fitted strictly inside each CV fold.
- **Winning Configuration (by Mean CV Abusive F1):** **Calibrated LinearSVC (C=0.5)** (`F1 = 79.44%`, `PR-AUC = 0.8900`).

| Rank | Model Configuration | Accuracy | Abusive Precision | Abusive Recall | Abusive F1 | ROC-AUC | PR-AUC | Mean Fit Time |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Calibrated LinearSVC (C=0.5) | 96.26% +/- 0.19% | 90.18% +/- 0.95% | 71.00% +/- 1.52% | **79.44%** +/- 1.16% | 0.9758 | 0.8900 | 48.09s |
| 2 | Calibrated LinearSVC (C=1.0) | 96.22% +/- 0.19% | 90.83% +/- 0.86% | 69.84% +/- 1.54% | **78.96%** +/- 1.20% | 0.9738 | 0.8861 | 44.64s |
| 3 | LogisticRegression (C=2.0) | 94.63% +/- 0.13% | 68.70% +/- 0.66% | 86.80% +/- 0.84% | **76.69%** +/- 0.51% | 0.9768 | 0.8910 | 6.34s |
| 4 | SGDClassifier (alpha=1e-5) | 94.35% +/- 0.12% | 66.97% +/- 0.70% | 87.67% +/- 0.93% | **75.93%** +/- 0.39% | 0.9773 | 0.8912 | 3.53s |
| 5 | LogisticRegression (C=1.0) | 94.34% +/- 0.13% | 66.93% +/- 0.58% | 87.60% +/- 0.87% | **75.88%** +/- 0.55% | 0.9771 | 0.8903 | 5.15s |
| 6 | LogisticRegression (C=0.5) | 94.21% +/- 0.14% | 66.22% +/- 0.65% | 87.87% +/- 0.75% | **75.52%** +/- 0.55% | 0.9769 | 0.8884 | 5.33s |
| 7 | SGDClassifier (alpha=1e-4) | 93.19% +/- 0.14% | 61.70% +/- 0.54% | 87.17% +/- 0.96% | **72.26%** +/- 0.58% | 0.9717 | 0.8686 | 4.84s |
| 8 | ComplementNB (alpha=1.0) | 87.91% +/- 0.07% | 45.20% +/- 0.19% | 89.21% +/- 0.92% | **60.00%** +/- 0.35% | 0.9540 | 0.8313 | 0.30s |
| 9 | ComplementNB (alpha=0.5) | 87.76% +/- 0.08% | 44.86% +/- 0.21% | 88.98% +/- 0.97% | **59.64%** +/- 0.36% | 0.9527 | 0.8286 | 0.37s |
