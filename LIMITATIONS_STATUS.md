# AbuseGuard AI: Limitations Status Audit (L1–L9)

This document honestly evaluates the nine known limitations of **AbuseGuard AI v1** after completing the **v2 upgrade**, using only verified experimental results.

| ID | Limitation in v1 | v2 Status | Honest Evidence & Rationale |
| :---: | :--- | :---: | :--- |
| **L1** | **Leetspeak, Symbol Masking & Stretched-Letter Evasion** (`1d10t`, `f*ck`, `stuuuupid`) | **FIXED** | `src/text_preprocessing.py` resolves inline leetspeak, symbol-masked roots, and >2 repeated letters while keeping pure numbers (`12345`) and negations intact. On the 20-case adversarial suite (`results/adversarial_examples.md`), pass rate rose from **40.0% (8/20)** in v1 to **100.0% (20/20)** in v2. |
| **L2** | **Word-Only Vocabulary Blindness to Typos & Subwords** | **FIXED** | `src/feature_extraction.py` now uses a 25,000-dimension `FeatureUnion` combining word `(1,2)` n-grams (10k) with `char_wb` `(3,5)` subword n-grams (15k), improving Jigsaw PR-AUC from `0.8512` to `0.8877` and test F1 from `70.32%` to `79.79%`. |
| **L3** | **Fixed 0.50 Threshold With High False-Alarm Rate (1,950 FPs)** | **FIXED** | Replaced single `0.50` cutoff with 3 validation-calibrated operating modes in `models/v2/thresholds.json`: **Strict** (`0.0846`, 92.94% recall), **Balanced** (`0.3905`, 79.79% F1, 449 FPs — a 77% FP drop), and **Lenient** (`0.5798`, 91.22% precision, 211 FPs). |
| **L4** | **Binary-Only Classification (No Severity or Sub-Type Breakdown)** | **FIXED** | Added `OneVsRestClassifier(LogisticRegression)` across all 6 Jigsaw categories (`models/v2/multilabel_model.joblib`), achieving **68.98% Micro-F1** and **0.978–0.991 ROC-AUC** across all 6 labels (`results/multilabel_metrics.md`). |
| **L5** | **Black-Box Verdict Without Token-Level Explainability** | **FIXED** | `src/predict.py` computes exact linear feature attributions (`TF-IDF value × model coefficient`) rendered in Streamlit's *"Why Was This Flagged?"* chart and returned by FastAPI `/predict`. |
| **L6** | **Single-Domain Overfitting / Cross-Platform Shift (Wikipedia → Twitter)** | **PARTLY FIXED** | Measured zero-shot shift on 24,783 Davidson tweets (`results/cross_domain_tweets.md`): zero-shot Jigsaw v2 reached 80.41% recall / 88.19% F1 on the 20% test split, while domain-adapting on Jigsaw + 80% Davidson boosted Twitter F1 to **96.51%** (96.19% recall). However, the deployed default model remains Jigsaw-only unless retrained with multi-domain data, and unseen platforms (e.g., Reddit/gaming chat) still experience domain drift. |
| **L7** | **Blindness to Roman-Script Hinglish / Code-Mixed Abuse** | **PARTLY FIXED** | Added an explicit, toggleable Roman-script Hinglish rule layer (`data/hinglish_lexicon.txt` + `src/hinglish_detector.py`) covering categorized insults, obscenities, violent threats, and hate terms with repeat-collapsing and word-boundary safety. Because it is lexicon-based rather than learned from large Hinglish corpora, unseen phonetic spellings or subtle Hinglish context can still slip through. |
| **L8** | **Sarcasm, Irony & Deep Multi-Clause Semantic Context** | **NOT FIXED** | While negation bigrams (`not bad`, `never attack`) improved (passing 5/5 safe-negation adversarial tests), `Calibrated LinearSVC` over bag-of-ngrams TF-IDF lacks contextual attention. Sarcastic praise (*"Oh brilliant, another genius edit"*) or implied threats without explicit toxic n-grams cannot be reliably resolved without transformer encoders (e.g., DistilBERT/RoBERTa). |
| **L9** | **Evolving Unseen Slang, Emojis & Multimodal Memes/Images** | **NOT FIXED** | Punctuation/non-ASCII stripping drops emojis, and the text-only TF-IDF pipeline cannot inspect images, memes, or brand-new post-training internet slang absent from the frozen vocabulary (though `char_wb` mitigates minor spelling variants, and Batch CSV + FastAPI `/predict` now make human-in-the-loop review and retraining integration practical). |

---

## Summary Scorecard
- **FIXED:** 5 (`L1`, `L2`, `L3`, `L4`, `L5`)
- **PARTLY FIXED:** 2 (`L6`, `L7`)
- **NOT FIXED:** 2 (`L8`, `L9`)

