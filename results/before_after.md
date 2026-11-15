# AbuseGuard AI: Verified Before vs After Comparison (v1 → v2)

All metrics below are measured directly on the frozen `v1` baseline (`models/v1/`) and upgraded `v2` pipeline (`models/v2/`) using identical stratification (`random_state=42`, 127,656 train / 31,915 held-out test) with zero data leakage.

---

## 1. Core Classification Metrics on Jigsaw Held-Out Test Set (n = 31,915)

| System / Operating Mode | Threshold | Accuracy | Abusive Precision | Abusive Recall | Abusive F1 | ROC-AUC | PR-AUC | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1 Baseline** (`LogisticRegression`, Word 10k) | `0.5000` | 92.55% | 59.09% | 86.81% | 70.32% | 0.9648 | 0.8512 | 1,950 | 428 |
| **v2 Balanced** (`Calibrated LinearSVC`, FeatureUnion 25k) | `0.3905` | **96.11%** | **84.52%** | 75.56% | **79.79%** | **0.9761** | **0.8877** | **449** | 793 |
| **v2 Strict** (High Recall mode) | `0.0846` | 91.46% | 54.73% | **92.94%** | 68.89% | **0.9761** | **0.8877** | 2,495 | **229** |
| **v2 Lenient** (High Precision mode) | `0.5798` | 96.04% | **91.22%** | 67.55% | 77.62% | **0.9761** | **0.8877** | **211** | 1,053 |

### Key Quantitative Improvements (`v1` → `v2 Balanced`)
- **Abusive F1-Score:** `70.32%` → **`79.79%`** (**+9.47 percentage points**)
- **Abusive Precision:** `59.09%` → **`84.52%`** (**+25.43 percentage points**)
- **Overall Accuracy:** `92.55%` → **`96.11%`** (**+3.56 percentage points**)
- **False Positives Cut:** `1,950` → **`449`** (**77.0% fewer false alarms** on safe comments)
- **PR-AUC (Threshold-Free Imbalance Metric):** `0.8512` → **`0.8877`** (**+0.0365**)
- **ROC-AUC:** `0.9648` → **`0.9761`** (**+0.0113**)
- **Maximum Safety Recall (`v2 Strict`):** `86.81%` → **`92.94%`** (**199 fewer missed abusive comments**, FN reduced from 428 to 229)

---

## 2. Adversarial Robustness (20 Tricky Edge Cases)

| Category | Cases | v1 Baseline Pass | v2 Balanced Pass |
| :--- | :---: | :---: | :---: |
| Leetspeak (`1d10t`, `k1ll`, `5tup1d`) | 5 | 0 / 5 (0%) | **5 / 5 (100%)** |
| Symbol Masking (`f*ck`, `sh!t`, `@$$hole`) | 5 | 4 / 5 (80%) | **5 / 5 (100%)** |
| Stretched Letters (`stuuuupid`, `kiiiillll`) | 5 | 0 / 5 (0%) | **5 / 5 (100%)** |
| Safe Negations (`not bad`, `do not hate`) | 5 | 4 / 5 (80%) | **5 / 5 (100%)** |
| **Total** | **20** | **8 / 20 (40.0%)** | **20 / 20 (100.0%)** |

---

## 3. Feature & Capability Upgrades

| Dimension | v1 | v2 |
| :--- | :--- | :--- |
| **Text Preprocessing** | Lowercase, URL strip, punctuation spacing | + De-obfuscation (leetspeak, symbol masking, >2 repeat compression) while preserving negations |
| **Feature Extraction** | Word TF-IDF `(1,2)`, 10,000 features (note: `src/` default was 5,000) | `FeatureUnion`: Word `(1,2)` 10,000 + `char_wb` `(3,5)` 15,000 = **25,000 features** |
| **Model Selection Rigor** | Single split: 2 models (LogReg vs MultinomialNB) | 5-Fold Stratified CV across **4 families / 9 configs** (`Calibrated LinearSVC`, `LogisticRegression`, `SGDClassifier`, `ComplementNB`) |
| **Threshold Control** | Hardcoded `0.50` | 3 validation-calibrated modes (`Strict`, `Balanced`, `Lenient`) via 85/15 validation PR curve |
| **Granularity** | Binary only (`Safe` / `Abusive`) | Binary + **6-label sub-types** (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`: Micro-F1 **68.98%**, Macro-F1 **56.79%**) |
| **Multilingual / Hinglish** | English-only | + Explicit toggleable **Roman-script Hinglish rule layer** (`data/hinglish_lexicon.txt`) |
| **Cross-Domain Testing** | Unmeasured | Tested on **24,783 Davidson Tweets** (Zero-shot F1 **88.19%** → Domain-adapted F1 **96.51%**) |
| **Explainability & Serving** | Score only; single-comment Streamlit | + **TF-IDF × Weight** attribution chart, Batch CSV upload, Side-by-Side v1/v2 UI, and **FastAPI `/predict`** |
| **Automated Tests** | 45 passing unit tests | **64 passing unit & API tests** |

