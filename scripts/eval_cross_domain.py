"""
eval_cross_domain.py - Phase 3: Cross-Domain Evaluation on Davidson Twitter Dataset.

1. Zero-shot evaluation of Jigsaw-trained v2 model on Davidson tweets (both full
   24,783 tweets and the 20% held-out 4,957 tweet test split).
2. Domain-adapted model trained on Jigsaw 80% train (127,656) + Davidson 80% train
   (19,826) and evaluated on the held-out 20% Davidson test split (4,957 tweets)
   as well as the held-out 20% Jigsaw test split (31,915 comments).
3. Saves results to results/cross_domain_tweets.md and results/cross_domain_tweets.json.

Usage:
    python scripts/eval_cross_domain.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_default_dataset_path, load_data
from src.feature_extraction import (
    create_feature_union_vectorizer,
    fit_tfidf_vectorizer,
    transform_tfidf,
)
from src.model_training import split_data
from src.preprocessing import prepare_dataset
from src.text_preprocessing import preprocess_text_series


def _get_probs(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def _metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "abusive_precision": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
        "abusive_recall": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "abusive_f1": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    t0 = time.time()
    v2_dir = PROJECT_ROOT / "models" / "v2"
    results_dir = PROJECT_ROOT / "results"
    dav_path = PROJECT_ROOT / "data" / "external" / "davidson_tweets.csv"

    if not dav_path.is_file():
        raise FileNotFoundError(f"Davidson dataset missing at {dav_path}")

    # Load v2 model, vectorizer, and balanced threshold
    v2_model = joblib.load(v2_dir / "best_model.joblib")
    v2_vec = joblib.load(v2_dir / "tfidf_vectorizer.joblib")
    v2_thresh = 0.5
    if (v2_dir / "thresholds.json").is_file():
        with open(v2_dir / "thresholds.json", "r", encoding="utf-8") as f:
            v2_thresh = float(json.load(f)["thresholds"]["balanced"]["threshold"])

    # 1. Load & prepare Davidson tweets
    print("[1/4] Loading and preprocessing Davidson Twitter dataset...")
    df_dav = pd.read_csv(dav_path)
    # Davidson classes: 0 = hate_speech, 1 = offensive_language, 2 = neither
    df_dav["abusive"] = df_dav["class"].isin([0, 1]).astype(int)
    dav_clean = preprocess_text_series(df_dav["tweet"], normalize_obfuscation=True)
    dav_y = df_dav["abusive"].to_numpy()
    dav_raw_class = df_dav["class"].to_numpy()

    dav_idx = np.arange(len(df_dav))
    dav_tr_idx, dav_te_idx = train_test_split(
        dav_idx, test_size=0.2, random_state=42, stratify=dav_raw_class
    )

    X_dav_all_vec = transform_tfidf(v2_vec, dav_clean)
    X_dav_te_vec = transform_tfidf(v2_vec, dav_clean.iloc[dav_te_idx])
    y_dav_te = dav_y[dav_te_idx]
    raw_class_te = dav_raw_class[dav_te_idx]

    # 2. Zero-shot evaluation
    print("[2/4] Evaluating Jigsaw-trained v2 zero-shot on Davidson tweets...")
    probs_all = _get_probs(v2_model, X_dav_all_vec)
    probs_te = _get_probs(v2_model, X_dav_te_vec)

    zs_all_bal = _metrics(dav_y, probs_all, threshold=v2_thresh)
    zs_all_050 = _metrics(dav_y, probs_all, threshold=0.50)
    zs_te_bal = _metrics(y_dav_te, probs_te, threshold=v2_thresh)
    zs_te_050 = _metrics(y_dav_te, probs_te, threshold=0.50)

    # Sub-class recall breakdown on 20% test split (Balanced threshold)
    preds_te_bal = (probs_te >= v2_thresh).astype(int)
    hate_recall_zs = float((preds_te_bal[raw_class_te == 0] == 1).mean())
    off_recall_zs = float((preds_te_bal[raw_class_te == 1] == 1).mean())
    neither_spec_zs = float((preds_te_bal[raw_class_te == 2] == 0).mean())

    # 3. Domain-Adapted Model (Jigsaw 80% train + Davidson 80% train)
    print("[3/4] Training domain-adapted model (Jigsaw train + 80% Davidson train)...")
    raw_jig = load_data(get_default_dataset_path())
    prep_jig = prepare_dataset(raw_jig)
    jig_clean = preprocess_text_series(prep_jig["comment_text"], normalize_obfuscation=True)
    X_jig_tr, X_jig_te, y_jig_tr, y_jig_te = split_data(
        jig_clean, prep_jig["abusive"], test_size=0.2, random_state=42, stratify=True
    )

    X_comb_tr = pd.concat([X_jig_tr, dav_clean.iloc[dav_tr_idx]], ignore_index=True)
    y_comb_tr = np.concatenate([y_jig_tr.to_numpy(), dav_y[dav_tr_idx]])

    da_vec = fit_tfidf_vectorizer(create_feature_union_vectorizer(), X_comb_tr)
    X_comb_tr_vec = transform_tfidf(da_vec, X_comb_tr)
    X_dav_te_da_vec = transform_tfidf(da_vec, dav_clean.iloc[dav_te_idx])
    X_jig_te_da_vec = transform_tfidf(da_vec, X_jig_te)

    da_model = clone(v2_model)
    da_model.fit(X_comb_tr_vec, y_comb_tr)

    print("[4/4] Evaluating domain-adapted model on held-out 20% Davidson & Jigsaw test sets...")
    da_dav_probs = _get_probs(da_model, X_dav_te_da_vec)
    da_jig_probs = _get_probs(da_model, X_jig_te_da_vec)

    da_dav_050 = _metrics(y_dav_te, da_dav_probs, threshold=0.50)
    da_jig_050 = _metrics(y_jig_te.to_numpy(), da_jig_probs, threshold=0.50)

    preds_te_da = (da_dav_probs >= 0.50).astype(int)
    hate_recall_da = float((preds_te_da[raw_class_te == 0] == 1).mean())
    off_recall_da = float((preds_te_da[raw_class_te == 1] == 1).mean())
    neither_spec_da = float((preds_te_da[raw_class_te == 2] == 0).mean())

    payload = {
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "davidson_stats": {
            "total_tweets": int(len(df_dav)),
            "hate_speech_0": int((df_dav["class"] == 0).sum()),
            "offensive_language_1": int((df_dav["class"] == 1).sum()),
            "neither_2": int((df_dav["class"] == 2).sum()),
            "abusive_pct": float(dav_y.mean() * 100),
            "train_split_80": int(len(dav_tr_idx)),
            "test_split_20": int(len(dav_te_idx)),
        },
        "zero_shot_full_davidson": {"balanced": zs_all_bal, "default_050": zs_all_050},
        "zero_shot_test_20": {
            "balanced": zs_te_bal,
            "default_050": zs_te_050,
            "subclass": {
                "hate_speech_recall": hate_recall_zs,
                "offensive_recall": off_recall_zs,
                "neither_specificity": neither_spec_zs,
            },
        },
        "domain_adapted_test_20": {
            "davidson_test_050": da_dav_050,
            "jigsaw_test_050": da_jig_050,
            "subclass": {
                "hate_speech_recall": hate_recall_da,
                "offensive_recall": off_recall_da,
                "neither_specificity": neither_spec_da,
            },
        },
    }
    with open(results_dir / "cross_domain_tweets.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md_lines = [
        "# Cross-Domain Twitter Evaluation (Davidson et al. Dataset)",
        "",
        "## Dataset Shift Summary",
        f"- **Source Domain (Jigsaw Wikipedia):** 159,571 comments — **10.17% Abusive** / 89.83% Safe.",
        f"- **Target Domain (Davidson Twitter):** {len(df_dav):,} tweets (`hate_speech`: {(df_dav['class']==0).sum():,}, `offensive_language`: {(df_dav['class']==1).sum():,}, `neither`: {(df_dav['class']==2).sum():,}) — **{dav_y.mean()*100:.2f}% Abusive** / {(1-dav_y.mean())*100:.2f}% Safe.",
        f"- **Split:** 80/20 Stratified (`random_state=42`): {len(dav_tr_idx):,} Davidson train / {len(dav_te_idx):,} Davidson held-out test.",
        "",
        "## Comparison on Exact Held-Out 20% Davidson Test Set (n = 4,957 tweets)",
        "",
        "| Experiment | Accuracy | Abusive Precision | Abusive Recall | Abusive F1 | Macro F1 | ROC-AUC | PR-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Zero-Shot Jigsaw v2** (`t={v2_thresh:.3f}`, Balanced) | {zs_te_bal['accuracy']*100:.2f}% | {zs_te_bal['abusive_precision']*100:.2f}% | {zs_te_bal['abusive_recall']*100:.2f}% | {zs_te_bal['abusive_f1']*100:.2f}% | {zs_te_bal['macro_f1']*100:.2f}% | {zs_te_bal['roc_auc']:.4f} | {zs_te_bal['pr_auc']:.4f} |",
        f"| **Zero-Shot Jigsaw v2** (`t=0.500`, Default) | {zs_te_050['accuracy']*100:.2f}% | {zs_te_050['abusive_precision']*100:.2f}% | {zs_te_050['abusive_recall']*100:.2f}% | {zs_te_050['abusive_f1']*100:.2f}% | {zs_te_050['macro_f1']*100:.2f}% | {zs_te_050['roc_auc']:.4f} | {zs_te_050['pr_auc']:.4f} |",
        f"| **Domain-Adapted (Jigsaw + 80% Davidson)** (`t=0.500`) | **{da_dav_050['accuracy']*100:.2f}%** | **{da_dav_050['abusive_precision']*100:.2f}%** | **{da_dav_050['abusive_recall']*100:.2f}%** | **{da_dav_050['abusive_f1']*100:.2f}%** | **{da_dav_050['macro_f1']*100:.2f}%** | **{da_dav_050['roc_auc']:.4f}** | **{da_dav_050['pr_auc']:.4f}** |",
        "",
        "## Zero-Shot Full Davidson Benchmark (n = 24,783 tweets)",
        "",
        f"- **Balanced (`t={v2_thresh:.3f}`):** Acc = {zs_all_bal['accuracy']*100:.2f}%, Prec = {zs_all_bal['abusive_precision']*100:.2f}%, Rec = {zs_all_bal['abusive_recall']*100:.2f}%, F1 = {zs_all_bal['abusive_f1']*100:.2f}%, ROC-AUC = {zs_all_bal['roc_auc']:.4f}, PR-AUC = {zs_all_bal['pr_auc']:.4f}",
        f"- **Default (`t=0.500`):** Acc = {zs_all_050['accuracy']*100:.2f}%, Prec = {zs_all_050['abusive_precision']*100:.2f}%, Rec = {zs_all_050['abusive_recall']*100:.2f}%, F1 = {zs_all_050['abusive_f1']*100:.2f}%",
        "",
        "## Davidson Sub-Class Breakdown (20% Test Split)",
        "",
        "| Davidson Sub-Class | Zero-Shot v2 (Balanced) | Domain-Adapted (`t=0.500`) |",
        "| :--- | :---: | :---: |",
        f"| Hate Speech (`class=0`) Recall | {hate_recall_zs*100:.2f}% | {hate_recall_da*100:.2f}% |",
        f"| Offensive Language (`class=1`) Recall | {off_recall_zs*100:.2f}% | {off_recall_da*100:.2f}% |",
        f"| Clean / Neither (`class=2`) Specificity | {neither_spec_zs*100:.2f}% | {neither_spec_da*100:.2f}% |",
        "",
        "## Domain-Adapted Jigsaw Retention Check (n = 31,915 Jigsaw Test)",
        "",
        f"- **Jigsaw Held-Out Test (`t=0.500`):** Accuracy = {da_jig_050['accuracy']*100:.2f}%, Precision = {da_jig_050['abusive_precision']*100:.2f}%, Recall = {da_jig_050['abusive_recall']*100:.2f}%, F1 = {da_jig_050['abusive_f1']*100:.2f}%, ROC-AUC = {da_jig_050['roc_auc']:.4f}",
        "",
        "## Key Engineering Takeaway",
        "Zero-shot transfer from Wikipedia talk pages to Twitter drops sharply in recall due to informal slang, AAVE colloquialisms, and inverted class priors (10% abusive in Jigsaw vs 83% in Davidson). Adding just 80% of Davidson training tweets alongside Jigsaw recovers Twitter performance substantially while keeping Jigsaw accuracy strong.",
        "",
    ]
    (results_dir / "cross_domain_tweets.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[DONE] Cross-domain evaluation finished in {time.time()-t0:.1f}s -> {results_dir / 'cross_domain_tweets.md'}")
    print(f"Zero-Shot (20% test @bal): F1={zs_te_bal['abusive_f1']*100:.2f}%, Rec={zs_te_bal['abusive_recall']*100:.2f}% | Domain-Adapted: F1={da_dav_050['abusive_f1']*100:.2f}%, Rec={da_dav_050['abusive_recall']*100:.2f}%")


if __name__ == "__main__":
    main()

