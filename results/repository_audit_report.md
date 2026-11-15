# AbuseGuard AI — Principal Research Scientist & Code Audit Report

**Repository:** `social_media_abuse_detector` (**AbuseGuard AI v2**)  
**Audit Mode:** Static Deep-Code Inspection (Read-Only; Zero Modifications to Code, Models, or Scripts)  
**Scope Inspected:** All modules in `src/`, `scripts/`, `tests/`, root entrypoints (`app.py`, `api.py`, `train.py`, `main.py`, `requirements.txt`), lexicon (`data/hinglish_lexicon.txt`), and persisted model/evaluation metadata (`models/`, `results/`).

---

## Executive Summary

**AbuseGuard AI v2** represents a methodologically sound classical-NLP upgrade over the frozen `v1` baseline (`models/v1/`). On the exact same stratified **127,656 train / 31,915 held-out Jigsaw test split** (`random_state=42`) with zero test-set leakage, upgrading from a single `Word (1,2)` TF-IDF Logistic Regression (`0.500` threshold) to `FeatureUnion` (25,000 word + subword dims) + `CalibratedClassifierCV(LinearSVC, C=0.5)` with validation-calibrated thresholds raised **Abusive F1 from `70.32%` to `79.79%` (+9.47 pts)**, **Abusive Precision from `59.09%` to `84.52%` (+25.43 pts)**, **PR-AUC from `0.8512` to `0.8877`**, and **cut false positives by 77.0% (`1,950 → 449`)**, while lifting adversarial evasion accuracy from **40.0% (`8/20`) to 100.0% (`20/20`)**.

However, deep code auditing surfaced several **concrete engineering risks, silent edge-case bugs, UI state-sync defects, and structural model limitations**:

| Severity | Finding | Primary Location(s) |
| :--- | :--- | :--- |
| **High** | **Training Entrypoint Drift:** Running `python train.py` or `python -m src.model_training` executes the *legacy v1* pipeline (`Word 10k` + LogReg/NB) and silently overwrites root `models/`, leaving `models/v2/` out of sync with root `models/`. | [train.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/train.py#L14-L19), [model_training.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/model_training.py#L474-L636), [benchmark_models.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/scripts/benchmark_models.py#L521-L548) |
| **High** | **Stale Sidebar State in Streamlit UI:** Changing `Engine Mode`, `Sensitivity Mode`, or `Hinglish Rule Layer` in the sidebar *after* clicking Analyze updates widget labels/headers (`v2 Improved (Lenient)`) while continuing to render the **stale** verdict, probabilities, and old `threshold_used` (`0.391`) from `st.session_state["last_outputs"]`. | [app.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L349-L368), [app.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L622-L633), [app.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L722-L741) |
| **High** | **Linear Bag-of-Ngrams Blind Spots (Spurious Pronouns & Negation Scope):** Single-word L2 TF-IDF normalization saturates `"you"` (`+8.31` v1 / `+2.25` v2 weight → `99.9%` v1, `97.2%` v2 Abusive), and `"you are not idiot"` scores `100.0%` Abusive because `"not idiot"` is OOV and `char_wb (3,5)` ignores cross-word negation. | [feature_extraction.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/feature_extraction.py#L42-L79), [text_preprocessing.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L20-L21) |
| **Medium** | **Hinglish Symbol Masking Mismatch & Missing Feature Attribution:** Inline comment in `_normalize_for_hinglish()` cites `bsd*k`, yet `_PUNCT_SPACE_TABLE` splits `*` into whitespace (`"bsd k"`), causing symbol-masked Hinglish (`bsd*k`, `ch*tiya`) and `1→l` (`10da`) to miss; pure Hinglish hits also produce an empty feature-attribution chart and bypass user threshold overrides. | [hinglish_detector.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/hinglish_detector.py#L35-L43), [predict.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L226-L255) |
| **Medium** | **Email/`@handle` Corruption & Stretch Collapse Artifacts:** `_MIXED_SYMBOL_TOKEN_PATTERN` rewrites `user@gmail.com` to `useragmail com` and `@john` to `ajohn`; `_REPEAT_3PLUS_PATTERN` collapses non-whitelisted natural double-letter words when stretched (`"haaaapppyyy"` → `"hapy"`). | [text_preprocessing.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L122-L173) |
| **Medium** | **Uncalibrated Multi-Label Cutoffs on Rare Classes:** `OneVsRestClassifier` uses fixed `0.50` thresholds, yielding `24.83%`–`28.51%` precision on rare classes (`threat`, `identity_hate`, `severe_toxic`) despite `0.983–0.991` ROC-AUC, causing spurious badges (e.g., `threat` on `"you"`). | [benchmark_models.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/scripts/benchmark_models.py#L552-L576), [predict.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L242-L248), [multilabel_metrics.md](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/results/multilabel_metrics.md#L7-L14) |
| **Low/Med** | **Test Suite Environment Coupling & Zero UI/Script Coverage:** `tests/test_api.py` (6 tests) fails without pre-built `models/v2/`; two integration tests in `tests/test_predict.py` silently pass when artifacts are absent; `app.py`, `main.py`, `train.py`, and `scripts/*.py` have `0%` automated test coverage. | [test_api.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_api.py#L15-L85), [test_predict.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_predict.py#L114-L200) |

---

## 1. Architecture & Pipeline Integrity

### 1.1 End-to-End Inference Data Flow (`src/predict.py`)

Live inference across both the Streamlit dashboard ([app.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py)) and the FastAPI microservice ([api.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/api.py)) converges on `predict_comment()` in [predict.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L146-L271):

```text
raw_text
  │
  ├──(empty / whitespace)─────────────────────────────► ValueError (422 in API; warning in UI)
  │
  ├──► [1] preprocess_text(raw_str, normalize_obfuscation=True/False)   (src/text_preprocessing.py)
  │        └─ lowercase → URL strip → masked roots (26 regexes) → @/$ inline sub
  │           → punctuation-to-space → digit leet + >2 repeat collapse → space trim
  │
  ├──► [2] detect_hinglish_abuse(raw_str)  [if enable_hinglish=True]    (src/hinglish_detector.py)
  │        └─ runs on RAW text via lru_cache lexicon (88 patterns, exact & all-repeat collapsed)
  │
  ├──(cleaned == "" e.g. URL/punctuation only)────────► Short-circuit Safe (0.0) or Hinglish Rule Score
  │
  ├──► [3] transform_tfidf(vectorizer, [cleaned])                       (src/feature_extraction.py)
  │        ├─ v1: Word TfidfVectorizer(1,2), max_features=10,000, L2-norm
  │        └─ v2: FeatureUnion = Word(1,2) [10,000] + char_wb(3,5) [15,000] = 25,000 sparse dims
  │
  ├──► [4] Primary Classifier + Thresholding + Hybrid Rule Merge
  │        ├─ v2: CalibratedClassifierCV(LinearSVC C=0.5, cv=3) → Platt ml_abusive_prob
  │        ├─ Threshold: Strict (0.0846) | Balanced (0.3905) | Lenient (0.5798) | Default (0.500)
  │        └─ If Hinglish flagged: final_prob = max(ml_abusive_prob, rule_score), is_abusive = True
  │
  ├──► [5] Multi-Label Sub-Type Tagger [if v2 multilabel_model present]
  │        └─ OneVsRestClassifier(LogisticRegression balanced) across 6 Jigsaw labels (0.50 cutoff)
  │           + unions matched Hinglish categories (insult / obscene / threat / hate)
  │
  └──► [6] Linear Attribution explain_prediction() [if explain=True]
           └─ Extracts coef_ (or mean of CalibratedClassifierCV folds / MNB log-prob delta)
              and ranks top_k by |tfidf_value * coefficient|
```

#### Component-Level Verification

1. **Preprocessing (`src/text_preprocessing.py`):**
   - `preprocess_text()` ([L197-L253](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L197-L253)) cleanly isolates `v1` (`normalize_obfuscation=False`) and `v2` (`normalize_obfuscation=True`) behavior.
   - `URL_PATTERN` strips `http://`, `https://`, and `www.*` links before punctuation handling ([L32](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L32), [L242](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L242)).
   - Punctuation is mapped to `" "` via `PUNCTUATION_SPACED_TABLE` ([L35-L37](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L35-L37)) so `hello.world` becomes `hello world` rather than `helloworld`, and negations (`not`, `never`, `no`) are never removed.
2. **Feature Engineering (`src/feature_extraction.py`):**
   - `create_feature_union_vectorizer()` ([L42-L79](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/feature_extraction.py#L42-L79)) binds two independent `TfidfVectorizer` transformers (`word` and `char`) inside a scikit-learn `FeatureUnion`. Each branch independently applies `sublinear_tf=True` (`1 + log(tf)`) and row-wise L2 normalization before sparse horizontal stacking (`hstack` across 10,000 + 15,000 = 25,000 columns).
   - `_is_vectorizer_fitted()` ([L82-L94](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/feature_extraction.py#L82-L94)) properly inspects both sub-transformers' `vocabulary_` and `idf_` attributes before permitting `transform_tfidf()` ([L151-L166](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/feature_extraction.py#L151-L166)).
3. **Training, CV & Leakage Audit (`src/model_training.py` & `scripts/`):**
   - **Data Leakage Discipline is Clean:** In both [verify_v1_baseline.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/scripts/verify_v1_baseline.py#L55-L75) and [benchmark_models.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/scripts/benchmark_models.py#L290-L505), raw Jigsaw data (`159,571` rows, 0 nulls/duplicates dropped by `prepare_dataset`) is split `80/20` stratified (`127,656` train / `31,915` test, `random_state=42`) **before** any IDF weights or model parameters are estimated. Inside 5-fold CV (`StratifiedKFold`, `benchmark_models.py` [L326-L355](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/scripts/benchmark_models.py#L326-L355)), `FeatureUnion` is instantiated and fit fresh inside each fold. Threshold calibration (`Strict=0.0846`, `Balanced=0.3905`, `Lenient=0.5798`) fits strictly on an `85/15` stratified sub-split of the training partition (`108,507` train_sub / `19,149` validation, [L409-L487](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/scripts/benchmark_models.py#L409-L487)). The `31,915` test split is touched once at evaluation.
   - *Minor methodological note:* `remove_duplicates(subset=["comment_text"])` and `preprocess_text_series()` execute prior to `split_data()`. While deterministic and leakage-free for IDF/weights (and `prepare_dataset` dropped 0 rows on Jigsaw), post-normalization near-duplicates (e.g., identical text differing only by capitalization or trailing punctuation) can still straddle train and test splits.

### 1.2 Architectural Integrity Issues & Subtle Bugs Found

1. **Critical Training Entrypoint Drift (`train.py` & `src/model_training.py` vs `scripts/benchmark_models.py`):**
   - [train.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/train.py#L11-L19) invokes `src.model_training.run_training_pipeline()` ([L474-L651](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/model_training.py#L474-L651)), which **still builds the legacy `v1` architecture**: a single `TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)` (`L554-L558`), comparing only `LogisticRegression` and `MultinomialNB` (`L573-L580`), and saving directly to root `models/` (`output_dir=None` at `L635`).
   - Meanwhile, `run_training_pipeline()` calls `preprocess_text_series(prepared_df["comment_text"])` (`L536`) whose default changed to `normalize_obfuscation=True`, meaning `python train.py` neither reproduces frozen `models/v1/` (`normalize_obfuscation=False`) nor builds `v2` (`FeatureUnion`, `Calibrated LinearSVC`, `thresholds.json`, `multilabel_model.joblib`), and **overwrites `models/best_model.joblib` and `models/evaluation_summary.json` with a stale v1-type artifact**.
   - Furthermore, [benchmark_models.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/scripts/benchmark_models.py#L521-L548) saves `thresholds.json` (`L476`) and `multilabel_model.joblib` (`L557`) **only** under `models/v2/`, while copying `best_model.joblib`, `tfidf_vectorizer.joblib`, `confusion_matrix.png`, and `evaluation_summary.json` to root `models/`. Notice that `app.py` ([L264-L266](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L264-L266)) falls back to root `models/` without `thresholds.json` or `multilabel_model.joblib` if `models/v2/` is missing.
   - Similarly, [feature_extraction.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/feature_extraction.py#L22) leaves `create_tfidf_vectorizer(max_features=5000)` while `model_training.py` overrides `max_features=10000` (already flagged in `results/v1_baseline.json`).

2. **Streamlit UI Stale Session-State Defect ([app.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py)):**
   - In Tab 1, `st.session_state["last_outputs"]` caches the result dictionary from the last **Analyze Comment** button click ([L604-L605](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L604-L605)).
   - The sidebar widgets (`engine_choice` [L349](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L349), `sensitivity_choice` [L356](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L356), `enable_hinglish` [L363](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L363)) define **no `on_change` callback** to invalidate `last_outputs` (unlike `demo_selector` at [L316-L322](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L316-L322)).
   - Yet rendering at [L622-L632](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py#L622-L632) passes the *newly changed* `{sensitivity_choice}` into `_render_result_panel(..., f"v2 Improved ({sensitivity_choice})")` while the card body reads `res["threshold_used"]`, `res["is_abusive"]`, and `res["hinglish_match"]` from the **stale** `outputs["v2"]`!
   - **Verified Live Effect (Observed in User Screenshot):** When switching from `Balanced` to `Lenient` without re-clicking Analyze, the sidebar and top caption display `Lenient (t=0.580)` and the card title displays `v2 Improved (Lenient): ABUSIVE`, while the card body still reads `threshold 0.391` from the prior `Balanced` run. Similarly, switching `engine_choice` from `v2 Improved` to `Side-by-Side Compare` drops silently into single-column `v2` (`L629`) because `outputs["v1"]` is absent, and changing sidebar settings does not invalidate `st.session_state["batch_results_df"]` in Tab 2 (`L722-L741`).

3. **Artifact Re-Sync Risk in `recalibrate_thresholds.py` & `api.py` Schema Omissions:**
   - [recalibrate_thresholds.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/scripts/recalibrate_thresholds.py#L125-L163) updates `thresholds.json`, `pr_curve_thresholds.png`, and `evaluation_summary.json`, but never updates `confusion_matrices_v1_vs_v2.png` if the Balanced operating threshold changes (safe here only because `t_bal=0.3905` remained identical).
   - In [api.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/api.py#L74-L87), `PredictResponse` omits the optional `"note"` key returned by `predict_comment()` when `cleaned_text == ""` (URL/punctuation-only comment, [predict.py L199](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L199)), silently discarding the explanation of why no ML features ran.

---

## 2. Edge Case & Failure Mode Analysis

### 2.1 Spurious Token Correlation Under L2 Normalization (`"you"` → 97.2% Abusive)

On the single-word input `"you"`, live evaluation produced **`99.9% Abusive` (v1)** and **`97.2% Abusive` (v2, all modes)**, with v2 multi-label `OneVsRestClassifier` even triggering false sub-type badges (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`). Code and statistical inspection explains exactly why:

1. **Corpus Label Bias:** Across prepared Jigsaw comments (`159,571` rows), the baseline abuse rate is **`10.17%`**. Comments containing `"you"` (`75,701` rows) are abusive **`13.46%`** of the time vs **`7.20%`** across comments without `"you"` (`83,870` rows) — nearly a **2× base-rate skew** arising from Wikipedia talk-page personal attacks (*"you are..."*, *"shut up you..."*).
2. **L2 Unit-Vector Amplification:** Both v1 (`TfidfVectorizer`) and v2 (`FeatureUnion`, where each branch applies `norm="l2"`) scale every document vector to unit L2 norm. For a single-word input `"you"`, `word__you` receives TF-IDF value **`1.0`**, and the `char_wb (3,5)` n-grams (`' yo '`, `'you '`, `' ou'`, etc.) also receive maximal unit-normalized weight. Because all other 24,990+ dimensions are `0.0`, the linear score $z = w_{\text{you}} \cdot 1.0 + \sum w_{\text{char}} \cdot x_{\text{char}} + b$ concentrates the full learned weight of second-person addressing (`w_v1("you") = +8.31`, mean SVM weight `word__you = +2.25`) without any dampening from benign context tokens, saturating both `CalibratedClassifierCV` and the `OvR` logistic heads.

### 2.2 Negation Blindness (`"you are not idiot"` → 100.0% Abusive)

Although v2 passed `5/5` curated safe-negation examples (`ADV-16` through `ADV-20`, e.g., `"i do not hate your work"` dropped from `51.1%` v1 to `15.1%` v2 because `"not hate"` exists in the top-10k word bigram vocabulary and `"hate"` carries a moderate coefficient), `"you are not idiot"` still scores **`100.0% Abusive`** in both v1 and v2 (even in `Lenient` mode):

1. **OOV Negation Bigrams Under Frequency Cap (`max_features=10,000`):** Exact negation-slur bigrams such as `"not idiot"` or `"not stupid"` occur rarely enough in Wikipedia talk pages to fall outside the top 10,000 word `(1,2)` features.
2. **Unweighted Unigram Summation & Subword Reinforcement:** Without syntactic dependency scope or a `"not <X>"` bigram feature, TF-IDF reduces `"you are not idiot"` to an additive bag of signals: `"idiot"` (`word`) plus 8 overlapping `char_wb` slices (`'diot'`, `'idiot'`, `'idio'`, etc.) carry overwhelmingly large positive weights that completely dwarf the mild negative coefficients on `"not"` and `"are not"`. Character n-grams (`char_wb` pads only single word boundaries) cannot represent negation across word spaces.

### 2.3 De-Obfuscation & OOV Slang Gaps (`src/text_preprocessing.py`)

While `normalize_obfuscation_text()` ([L177-L194](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L177-L194)) cleanly lifted adversarial evasion from `40.0%` to `100.0%`, line-level inspection reveals four edge cases:

1. **Email Address & Social Handle Mangling (`_decode_inline_symbols`, [L123-L135](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L123-L135)):**
   - `_MIXED_SYMBOL_TOKEN_PATTERN = re.compile(r"[a-z0-9@$]+")` executes *prior* to `PUNCTUATION_SPACED_TABLE`.
   - Any email address (`"user@gmail.com"`) matches `"user@gmail"`, contains `@` and letters, and mutates into **`"useragmail com"`**. Any social mention (`"@sarah"`) mutates into **`"asarah"`** — particularly impactful on Twitter/cross-platform text (Davidson dataset).
2. **Unnatural Collapse of Stretched Natural-Double Words ([L106-L120](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L106-L120), [L166-L173](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L166-L173)):**
   - Only 13 words are whitelisted in `_STRETCHED_DOUBLE_LETTER_PATTERNS` (`kill`, `hell`, `ass`, `piss`, `faggot`, `nigg*`, `bullshit`, `fool`, `agree`, `good`, `cool`, `free`, `noob`).
   - Any other stretched word with a natural double consonant collapses `3+` repeats down to **1** character (`"haaaapppyyy"` → `"hapy"`, `"sooooorrrry"` → `"sory"`, `"winnnner"` → `"winer"`), pushing clean tokens out of the `word (1,2)` vocabulary.
3. **Closed-List Masking & Multi-Char Masking (`_MASKED_ROOT_PATTERNS`, [L43-L80](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/text_preprocessing.py#L43-L80)):**
   - Regex patterns cover single-character masking on ~35 toxic roots and spaced single-letter patterns only for `f u c k`, `s h i t`, `b i t c h`, `k i l l`. Multi-asterisk censoring (`"f**k"`, `"s***"`, `"b****"`), omitted roots (`"r*pe"`, `"wh*te tr*sh"`), Unicode homoglyphs (Cyrillic `а`/`о`/`і`), and zero-width joiners (`\u200b`) slip past `normalize_obfuscation_text()`, leaving partial reliance on `char_wb`.
4. **Unicode, Emoji & Post-2018 OOV Slang (`LIMITATIONS_STATUS.md` `L9`):**
   - Non-ASCII emojis (`🖕`, `🔪`, `💀`) and diacritics lack Unicode NFKD folding or semantic mapping and are stripped by `TfidfVectorizer`'s default `\b\w\w+\b` word tokenizer, while zero-shot cross-domain recall on informal/AAVE Twitter slang remains vulnerable without domain adaptation (`80.41%` zero-shot recall vs `96.19%` domain-adapted on Davidson test split, [cross_domain_tweets.md](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/results/cross_domain_tweets.md#L12-L14)).

### 2.4 Roman-Script Hinglish Rule Engine Gaps (`src/hinglish_detector.py` & `src/predict.py`)

1. **Broken Symbol-Masking Promise (`bsd*k`):** [hinglish_detector.py L37-L39](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/hinglish_detector.py#L37-L39) states `# Basic inline leet normalization for Hinglish masking (e.g. chut1ya, bsd*k, k@mina)`. While `chut1ya` and `k@mina` work, `_PUNCT_SPACE_TABLE` replaces `*` with `" "`, turning **`"bsd*k"` → `"bsd k"`** and `"ch*tiya"` → `"ch tiya"`, which **fails to match** `\bbsdk\b` or `\bchutiya\b`!
2. **No `1 → l` Dual Decoding:** Line `L38` maps `1` strictly to `i`, missing common Hindi/Hinglish substitutions where `1` stands for `l` (e.g., `"10da"` / `"1und"` → `"ioda"` / `"iund"`).
3. **Absolute Threshold Bypass & No Context Window ([predict.py L226-L229](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L226-L229)):** A single lexicon match forces `is_abusive = True` with `final_abusive_prob = max(ml_abusive_prob, rule_score)` (`0.88–0.98`), ignoring user threshold settings (even a strict custom threshold `0.9999` would still flip to `Abusive` if Hinglish triggers — note `test_predict_explainability_and_threshold_override` only passes because its comment has no Hinglish!) and ignoring benign reporting/quoting context (*"Please block users calling others 'chutiya'"*).
4. **Missing Explanation Bar for Hinglish Hits ([predict.py L255](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L255)):** `explain_prediction()` inspects only English TF-IDF sparse features (`word__` and `char__`). Pure-Hinglish comments produce an empty attribution list (`[]`) in the API and render `"No active vocabulary features matched"` in Streamlit, creating a UX contradiction beneath an `ABUSIVE (Hinglish Rule Triggered)` banner.

### 2.5 Multi-Label Precision Collapse on Rare Classes (`OneVsRestClassifier`)

As measured in [multilabel_metrics.md](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/results/multilabel_metrics.md), all 6 binary Logistic Regression heads use `class_weight="balanced"` with an **uncalibrated `0.50` decision threshold** ([predict.py L246](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L246)). On imbalanced minority labels (`prevalence < 1%`), balanced class weighting shifts intercepts upward, producing high ROC-AUC (`0.983–0.991`) and high recall (`72.5–87.9%`) at the expense of severe false-positive inflation:
- `threat` (`0.31%` prevalence, 98 positives): **Precision `24.83%`** (Recall `72.45%`, F1 `36.98%`)
- `identity_hate` (`0.81%`, 259 positives): **Precision `24.97%`** (Recall `82.24%`, F1 `38.31%`)
- `severe_toxic` (`0.96%`, 305 positives): **Precision `28.51%`** (Recall `87.87%`, F1 `43.05%`)

---

## 3. Test Coverage & Robustness

### 3.1 Verified Test Suite Inventory (`64` Tests Across `7` Files)

| Test Module | Count | What Is Covered | Data / Dependency Isolation |
| :--- | :---: | :--- | :--- |
| [test_data_loader.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_data_loader.py) | 12 | CSV load, missing file/cols, empty/null content, strict `{0,1}` label validation, `abusive` target union, missing-text & dedup helpers | 100% synthetic DataFrames + `tmp_path` |
| [test_text_preprocessing.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_text_preprocessing.py) | 13 | Case folding, URL removal, whitespace, punctuation splitting, negation retention, None/NaN, series error, leet/masking/stretch, `normalize_obfuscation=False` | 100% self-contained unit tests |
| [test_feature_extraction.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_feature_extraction.py) | 9 | Vectorizer defaults, sparse output, frozen-vocab transform guard, `NotFittedError`, empty guards, `FeatureUnion` (`word__`/`char__` prefixes) | 100% synthetic toy corpora |
| [test_model_training.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_model_training.py) | 11 | Stratified split reproducibility & mismatch errors, zero-leakage test-vocab isolation, LR/MNB training & metrics, model selection, joblib save/load, synthetic `run_training_pipeline` | 100% synthetic + `tmp_path` |
| [test_predict.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_predict.py) | 9 | Safe/abusive inference, blank `ValueError`, URL-only input, missing dir error, linear explanation & threshold override, Hinglish toggle, + **2 conditional disk-artifact tests** | 7 synthetic + 2 conditional (`models/`, `models/v2/`) |
| [test_hinglish_detector.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_hinglish_detector.py) | 4 | Lexicon category loading, insult/obscene/threat/stretched detection, false-positive safety (`March` vs `mar`, `Randy` vs `randi`) | Reads live `data/hinglish_lexicon.txt` |
| [test_api.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_api.py) | 6 | `GET /health`, `POST /predict` (Safe, obfuscated+explain+multilabel, Hinglish toggle, threshold ordering, blank `422`) | **Coupled to live `models/v2/` via `TestClient(app)`** |
| **Total** | **64** | Core `src/` modules + FastAPI happy paths (**`64 passed, 1 warning`**) | Clean synthetic `src/` tests; `api` + integration coupled to disk |

*(Note on runtime warning: Starlette/FastAPI `TestClient` emits a minor deprecation warning unless pinned/updated alongside `httpx`; all 64 assertions pass cleanly.)*

### 3.2 Fragile Tests & Untested Code Paths

1. **Fragile Live-Artifact Coupling in [test_api.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_api.py#L13-L85):**
   - `client = TestClient(app)` runs against `get_v2_bundle()` reading `models/v2/` from disk with no synthetic fallback fixture or dependency override. In CI or on a fresh Git clone (where `.joblib` models are gitignored/untrained), 5 of 6 tests (`test_health_endpoint` and 4 `/predict` tests) will immediately fail with `503`.
2. **Silent No-Op Integration Tests in [test_predict.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_predict.py#L114-L200):**
   - `test_predict_using_real_saved_artifacts_if_present()` (`L114-L138`) and `test_v2_bundle_end_to_end_if_present()` (`L181-L200`) wrap all assertions inside `if ...is_file():` without `pytest.skip()`. If model files are absent or deleted, pytest falsely reports them as `PASSED`.
   - Furthermore, `test_predict_using_real_saved_artifacts_if_present()` tests `models/best_model.joblib` with `threshold=None` (`L126-L137`, evaluating Platt `0.50` rather than Balanced `0.3905`).
3. **Completely Untested Modules (0% Automated Coverage):**
   - [app.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py) (`921` lines): No `AppTest` smoke tests for sidebar mode switching, Tab 1 demo selection, Tab 2 CSV parsing/column selection, or Tab 3 missing/present `results/*.json` error tolerance.
   - `scripts/` (`benchmark_models.py`, `recalibrate_thresholds.py`, `eval_adversarial.py`, `eval_cross_domain.py`, `verify_v1_baseline.py` — `1,359` lines total): Threshold calibration edge cases (e.g., empty `strict_candidates`/`lenient_candidates` fallbacks) and probability helpers have zero unit tests.
   - [train.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/train.py) (`36` lines) and [main.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/main.py) (`211` lines).
4. **Untested `src/` & `api.py` Edge Branches:**
   - `_extract_linear_weights()` for `MultinomialNB` (`log_probs[1] - log_probs[0]`, [predict.py L85-L89](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L85-L89)) has zero unit tests.
   - Empty-cleaned-text + Hinglish hit branch ([predict.py L180-L200](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/src/predict.py#L180-L200)) and `/predict` on URL-only input (`"https://example.com"`) are untested.
   - [api.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/api.py) `/health` `503` handler (`L111-L112`), invalid `sensitivity` `422`, and `top_k` validation bounds (`1 <= top_k <= 25`) are untested.
   - Symbol-masked Hinglish (`bsd*k`, `ch*tiya`) is not tested in [test_hinglish_detector.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/tests/test_hinglish_detector.py) (which hides the `_PUNCT_SPACE_TABLE` bug).

---

## 4. Suggested Next-Stage Polish (Optional, Non-Breaking Enhancements)

All enhancements below preserve backward compatibility with existing `src/` signatures, `api.py` `/predict` contracts, and frozen `models/v1/` and `models/v2/` artifacts:

1. **Fix Sidebar State Sync + Add Short-Text / Borderline Confidence Calibration UI ([app.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/app.py)):**
   - Attach an `on_change` callback to `engine_choice`, `sensitivity_choice`, and `enable_hinglish` (or automatically re-run `_run_analysis()` when `st.session_state["analyzed"]` is `True` and sidebar settings differ from stored input settings). That eliminates the stale-threshold mismatch between the card header (`Lenient`) and body (`threshold 0.391`). Do the same invalidation/recompute check for Tab 2's `batch_results_df`.
   - Add a clear **Confidence & Uncertainty Band** (e.g., *High Confidence*, *Borderline — near threshold*, or *Short-Input Saturation Warning* when `len(cleaned.split()) <= 2` and `ml_abusive_probability >= 0.85` on common pronouns like `"you"`) alongside the existing calibration caption so moderators immediately recognize L2 single-token saturation.
2. **Exportable Moderation Audit Logs (Tab 1 History + Enhanced Tab 2 Export):**
   - Maintain a lightweight session audit log (`st.session_state["audit_log"]`) recording timestamp, engine mode, sensitivity & threshold, final verdict, `ml_abusive_probability`, active categories, Hinglish hits, and top attribution features, with one-click **CSV / JSON Audit Log Download**.
   - In Tab 2 (`Batch CSV Upload`), add an optional **Filter View** (`All` / `Abusive Only` / `Safe Only` / `Hinglish Hits` / `Threshold Borderline ±0.10`) above the results table before export.
3. **Hinglish Masking Parity & Virtual Rule Attribution (`src/hinglish_detector.py`, `src/predict.py`):**
   - Update `_normalize_for_hinglish()` to strip or decode inline obfuscation characters (`*`, `!`, `#`) *inside* alphanumeric tokens (so `bsd*k` → `bsdk` and `ch*tiya` → `chtiya`/`chutiya` match as documented) and check `1→l` variants.
   - When `hinglish_info["hinglish_flagged"]` is `True`, append synthetic attribution bars (e.g., `Hinglish: kameena [rule]`) to `top_features` (or render a dedicated callout banner inside `_plot_why_flagged`) so moderators never see *"No active vocabulary features matched"* on flagged Hinglish comments.
4. **Email / `@Handle` Preservation Before `@→a` Decoding (`src/text_preprocessing.py`):**
   - Protect email addresses (`\S+@\S+\.\S+`) and leading `@mentions` before `_decode_inline_symbols()` runs so `@username` and `user@gmail.com` are not silently rewritten to `ausername` / `useragmail`. *(Wait to deploy in training only when rebuilding a `v3` model to maintain train/serve preprocessing parity, or keep behind a parameter flag.)*
5. **Per-Label Validation Calibration for Multi-Label Sub-Types (`models/v2/`):**
   - Calibrate per-label F1/precision-targeted thresholds across the 6 `OneVsRestClassifier` heads on the existing `19,149`-sample validation split (`multilabel_thresholds.json`), cutting false-positive `threat` / `severe_toxic` / `identity_hate` badges (currently `24.8%–28.5%` precision at `0.50`) without altering any model weights.
6. **Unify `train.py` Safety & Harden `tests/` Without Retraining:**
   - Guard [train.py](file:///c:/Users/lucky/OneDrive/Desktop/social_media_abuse_detector/train.py) with an explicit `--pipeline {v1,v2}` CLI flag (defaulting to `v2` or saving to a non-clobbering path) and keep `thresholds.json` + `multilabel_model.joblib` synchronized wherever `best_model.joblib` is promoted.
   - Decouple `tests/test_api.py` from disk `models/v2/` using a small synthetic bundle override fixture for unit CI, convert `if is_file()` blocks in `test_predict.py` to explicit `@pytest.mark.skipif`, add unit tests for `MultinomialNB` weights and symbol-masked Hinglish, and add `streamlit.testing.v1.AppTest` UI smoke tests.
7. **Longer-Term Model Architecture Note (`v3` Roadmap — Beyond Bugfixes):**
   - Resolving cross-word negation scope (`"you are not idiot"`), compositional context, and sarcasm (`L8`) fundamentally requires contextual attention encoders (e.g., fine-tuned `DistilBERT` / `RoBERTa-base` or `DeBERTa-v3-small`) rather than bag-of-ngrams TF-IDF linear models.

