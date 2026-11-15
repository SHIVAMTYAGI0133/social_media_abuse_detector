# Cross-Domain Twitter Evaluation (Davidson et al. Dataset)

## Dataset Shift Summary
- **Source Domain (Jigsaw Wikipedia):** 159,571 comments — **10.17% Abusive** / 89.83% Safe.
- **Target Domain (Davidson Twitter):** 24,783 tweets (`hate_speech`: 1,430, `offensive_language`: 19,190, `neither`: 4,163) — **83.20% Abusive** / 16.80% Safe.
- **Split:** 80/20 Stratified (`random_state=42`): 19,826 Davidson train / 4,957 Davidson held-out test.

## Comparison on Exact Held-Out 20% Davidson Test Set (n = 4,957 tweets)

| Experiment | Accuracy | Abusive Precision | Abusive Recall | Abusive F1 | Macro F1 | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot Jigsaw v2** (`t=0.391`, Balanced) | 82.09% | 97.64% | 80.41% | 88.19% | 75.55% | 0.9022 | 0.9798 |
| **Zero-Shot Jigsaw v2** (`t=0.500`, Default) | 80.61% | 98.62% | 77.79% | 86.97% | 74.55% | 0.9022 | 0.9798 |
| **Domain-Adapted (Jigsaw + 80% Davidson)** (`t=0.500`) | **94.21%** | **96.83%** | **96.19%** | **96.51%** | **89.78%** | **0.9750** | **0.9944** |

## Zero-Shot Full Davidson Benchmark (n = 24,783 tweets)

- **Balanced (`t=0.391`):** Acc = 81.70%, Prec = 97.54%, Rec = 80.02%, F1 = 87.92%, ROC-AUC = 0.9012, PR-AUC = 0.9795
- **Default (`t=0.500`):** Acc = 80.02%, Prec = 98.19%, Rec = 77.42%, F1 = 86.58%

## Davidson Sub-Class Breakdown (20% Test Split)

| Davidson Sub-Class | Zero-Shot v2 (Balanced) | Domain-Adapted (`t=0.500`) |
| :--- | :---: | :---: |
| Hate Speech (`class=0`) Recall | 77.62% | 87.76% |
| Offensive Language (`class=1`) Recall | 80.61% | 96.82% |
| Clean / Neither (`class=2`) Specificity | 90.40% | 84.39% |

## Domain-Adapted Jigsaw Retention Check (n = 31,915 Jigsaw Test)

- **Jigsaw Held-Out Test (`t=0.500`):** Accuracy = 96.15%, Precision = 90.65%, Recall = 69.28%, F1 = 78.53%, ROC-AUC = 0.9746

## Key Engineering Takeaway
Zero-shot transfer from Wikipedia talk pages to Twitter drops sharply in recall due to informal slang, AAVE colloquialisms, and inverted class priors (10% abusive in Jigsaw vs 83% in Davidson). Adding just 80% of Davidson training tweets alongside Jigsaw recovers Twitter performance substantially while keeping Jigsaw accuracy strong.
