# AbuseGuard AI v2 — Presentation & Viva Guide (17 November Evaluation)

## Part A: 3-Minute Live Demo Script

### Minute 0:00 – 0:45 | Problem Framing & v1 → v2 Upgrade Story
- **Open on Tab 3 (`Evaluation & Benchmarks`)**:
  - *"In v1, AbuseGuard AI used a single word-level TF-IDF Logistic Regression model. It caught 86.81% of abusive comments, but produced 1,950 false alarms on safe comments (59.09% precision, 70.32% F1) and missed leetspeak/masked abuse."*
  - *"For v2, we ran a 5-fold Stratified Cross-Validation benchmark across 4 model families (9 configurations) on 127,656 training comments with zero data leakage. **Calibrated LinearSVC (C=0.5)** with a 25,000-feature **Word + Character N-Gram FeatureUnion** won with **79.44% CV F1** and **79.79% test F1** — cutting false positives by **77%** (from 1,950 down to 449) and raising precision to **84.52%**."*
  - Briefly point to the **PR Curve Thresholds** (`Strict`, `Balanced`, `Lenient`) and **Confusion Matrices** side-by-side.

### Minute 0:45 – 1:45 | Live Side-by-Side v1 vs v2 Comparison (Tab 1)
- In the left sidebar, select **Engine Mode: `Side-by-Side Compare`**, **Sensitivity: `Balanced`**, and keep **Hinglish Rule Layer: `ON`**.
- **Demo 1 (Leetspeak Bypass):** Select `3. Leetspeak Evasion (Abusive)` (`"y0u ar3 an 1d10t and 5tup1d 10s3r"`), click **Analyze Comment**.
  - Point out: *"`v1 Original` predicts **SAFE (30.6%)** because `1d10t` and `5tup1d` are out-of-vocabulary. `v2 Improved` normalizes it to `you are an idiot and stupid loser`, flags **ABUSIVE (~100%)**, tags **toxic + insult**, and shows the top contributing word/character n-grams in 'Why Was This Flagged?'."*
- **Demo 2 (Safe Negation):** Select `2. Polite Negation Edge Case (Safe)` (`"This edit is not bad at all; let us never attack users and keep discussions polite."`), click **Analyze Comment**.
  - Point out: *"Negations (`not`, `never`) are preserved, so v2 verifies it as **SAFE** instead of blindly flagging the word `bad` or `attack`."*
- **Demo 3 (Hinglish Hybrid Rule Layer):** Select `7. Roman Hinglish Abuse (Rule Layer)`, click **Analyze Comment**, then toggle `Enable Hinglish Rule Layer` OFF and re-analyze to show how the rule layer transparently catches code-mixed Hinglish insults.

### Minute 1:45 – 2:30 | Sensitivity Modes & Batch CSV Moderation (Tab 2)
- Switch to `v2 Improved` and briefly explain the **Strict (`0.085`, 92.9% recall)** vs **Balanced (`0.391`, 79.8% F1)** vs **Lenient (`0.580`, 91.2% precision)** sidebar modes.
- Open **Tab 2 (`Batch CSV Upload`)**, click **`Load Sample 8-Comment Batch`**, click **`Run Batch Prediction`**, and show the KPI counts, annotated table, and **Download Annotated Results CSV** button.

### Minute 2:30 – 3:00 | API, Cross-Domain Honesty & Testing
- Mention the **FastAPI `/predict` endpoint (`api.py`)** and **64 passing `pytest` tests**.
- Conclude honestly using **Tab 3 / `LIMITATIONS_STATUS.md`**: *"We also measured domain shift on 24,783 Twitter posts (Davidson dataset): zero-shot recall drops to 80.4%, and domain adaptation recovers F1 to 96.5%. Complex sarcasm (L8) and unseen slang/images (L9) remain future transformer/multimodal work."*

---

## Part B: 12 Viva Questions & Plain-Language Answers

### 1. Why did you upgrade from word-only TF-IDF to a `FeatureUnion` of word + character n-grams?
- **Answer:** Word-only TF-IDF breaks completely if a user changes even one letter (like `id10t` or `stuuuupid`). Adding `char_wb` (3–5 character n-grams inside word boundaries) breaks words into overlapping pieces (`idi`, `dio`, `iot`), so even partial misspellings or variants share features with toxic words while keeping full-word bigrams (`not bad`) intact.

### 2. Why did `Calibrated LinearSVC` beat `LogisticRegression` and `ComplementNB` in your 5-fold CV?
- **Answer:** Linear Support Vector Machines find a maximum-margin boundary in high-dimensional sparse space (25,000 TF-IDF features), which prevents overfitting on noisy words. On 5-fold CV, `Calibrated LinearSVC (C=0.5)` achieved **79.44% F1** and **90.18% precision**, compared to **76.69% F1** for the best `LogisticRegression` and **60.00% F1** for `ComplementNB`.

### 3. Why wrap `LinearSVC` in `CalibratedClassifierCV`?
- **Answer:** Standard `LinearSVC` outputs raw signed distances (`decision_function`), not calibrated probabilities between `0` and `1`. Wrapping it with `CalibratedClassifierCV(cv=3)` fits Platt scaling (sigmoid calibration) so we get trustworthy probability estimates needed for threshold tuning (`Strict` / `Balanced` / `Lenient`) and UI confidence meters.

### 4. How did you prevent data leakage during all these upgrades?
- **Answer:** We locked the exact **80/20 stratified split (`random_state=42`: 127,656 train / 31,915 test)** at the very start. Both the 5-fold CV and the 85/15 validation split (`108,507` / `19,149`) for threshold calibration happened **strictly inside the 127,656 training set**, fitting `FeatureUnion` fresh inside every fold. The 31,915 test set was touched only once for final evaluation.

### 5. Why do you have Strict, Balanced, and Lenient sensitivity modes instead of one `0.50` threshold?
- **Answer:** Different moderation workflows have different error costs. In `v2`, **Strict (`0.085`)** catches **92.94%** of all abusive comments for human review queues; **Balanced (`0.391`)** maximizes F1 (`79.79%`); and **Lenient (`0.580`)** enforces **91.22% precision** (only 211 false alarms out of 28,670 safe comments) for automated actions.

### 6. Why did you keep `models/v1/` frozen instead of overwriting it?
- **Answer:** Freezing `models/v1/` enables exact, apples-to-apples benchmarking on the identical 31,915 test comments and lets evaluators test live **Side-by-Side Comparison** in the Streamlit app to see v1 fail on obfuscations that v2 catches.

### 7. Wait — v1 had 86.81% recall while v2 Balanced has 75.56% recall. Is v2 worse on recall?
- **Answer:** No—that is a threshold operating-point difference, not weaker ranking. Look at the threshold-free **PR-AUC** (`0.8512` → `0.8877`) and **ROC-AUC** (`0.9648` → `0.9761`): v2 ranks abusive comments strictly better at every level. If high recall is the priority, switching v2 to **Strict mode** achieves **92.94% recall** (beating v1's 86.81% by +6.13 points), while **Balanced mode** improves F1 by +9.47 points by eliminating 1,501 false positives.

### 8. How does your text de-obfuscation avoid breaking normal numbers like `12345` or `v2`?
- **Answer:** Purely numeric tokens (`12345`, `3.14`) have no letters, so digit decoding skips them completely. Symbol masking (`f*ck`, `sh!t`) runs via targeted toxic-root patterns before punctuation spacing, so normal punctuation like `stop!now` still splits cleanly into `stop now`. And repeat compression only shrinks 3+ identical consecutive letters, leaving normal doubles like `good` and `kill` untouched.

### 9. How does the Multi-Label classifier work, and why are F1 scores lower on `threat` and `severe_toxic`?
- **Answer:** We trained a `OneVsRestClassifier(LogisticRegression(class_weight='balanced'))` with 6 independent binary heads on the 6 Jigsaw columns. Rare classes like `threat` (only 98 positives in 31,915 test comments, 0.31%) and `identity_hate` (0.81%) still achieve **0.9896** and **0.9831 ROC-AUC** and **72%–82% recall**, but `class_weight='balanced'` over-predicts positives on rare classes, lowering precision and F1.

### 10. Why did you add a rule-based Hinglish layer instead of training an ML model for Hinglish?
- **Answer:** Jigsaw only contains English Wikipedia comments, and there is no large labeled Hinglish dataset in this repository. A transparent, toggleable word-boundary lexicon (`data/hinglish_lexicon.txt`) with repeat-collapsing (`chutiyyaaaa` → `chutiya`) catches Roman Hinglish abuse immediately without risking false positives on English words like `March` or `Randy`.

### 11. What did the Davidson Twitter cross-domain test teach you?
- **Answer:** It proved **domain shift is real**. Jigsaw Wikipedia comments are 10.17% abusive and relatively formal, whereas Davidson tweets are 83.20% abusive/offensive with heavy Twitter slang. Zero-shot Jigsaw v2 dropped to 80.41% recall on Davidson's 20% test split, while training on Jigsaw + 80% Davidson boosted Twitter F1 from **88.19% to 96.51%** without hurting Jigsaw accuracy (`96.15%`).

### 12. How does "Why Was This Flagged?" compute feature explanations, and what can v2 still NOT solve?
- **Answer:** Because `Calibrated LinearSVC` is linear, each feature's contribution to the raw decision score is simply **TF-IDF value × learned linear weight** (averaged across the 3 calibration folds). Positive products push toward Abusive; negative products push toward Safe. As documented in `LIMITATIONS_STATUS.md`, v2 still cannot reliably understand **complex sarcasm/irony** (`L8`) or **emojis, memes, and unseen slang** (`L9`), which would require fine-tuned transformers (e.g., DistilBERT) and multimodal models in future work.

