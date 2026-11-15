"""
recalibrate_thresholds.py - Updates Lenient threshold target to >= 92% validation
precision so Strict / Balanced / Lenient represent three distinct operating points
without re-running the 5-fold CV benchmark.

Zero data leakage: calibrates exclusively on the 85/15 training/validation split
(108,507 train_sub / 19,149 val) and re-evaluates the frozen v2 model on the
31,915 test set at the updated operating thresholds.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_models import (
    _evaluate_at_threshold,
    _get_abusive_probs,
    _plot_pr_thresholds,
    _plot_v1_vs_v2_confusion_matrices,
)
from src.data_loader import get_default_dataset_path, load_data
from src.feature_extraction import (
    create_feature_union_vectorizer,
    fit_tfidf_vectorizer,
    transform_tfidf,
)
from src.model_training import plot_confusion_matrix, split_data
from src.preprocessing import prepare_dataset
from src.text_preprocessing import preprocess_text_series


def main() -> None:
    results_dir = PROJECT_ROOT / "results"
    v2_dir = PROJECT_ROOT / "models" / "v2"
    root_models_dir = PROJECT_ROOT / "models"

    v2_model = joblib.load(v2_dir / "best_model.joblib")
    v2_vec = joblib.load(v2_dir / "tfidf_vectorizer.joblib")

    raw_df = load_data(get_default_dataset_path())
    prepared_df = prepare_dataset(raw_df)
    clean_texts = preprocess_text_series(prepared_df["comment_text"], normalize_obfuscation=True)

    X_train_texts, X_test_texts, y_train, y_test = split_data(
        clean_texts, prepared_df["abusive"], test_size=0.2, random_state=42, stratify=True
    )

    # 85/15 stratified validation split inside training set
    tr_sub_idx, val_sub_idx = train_test_split(
        np.arange(len(X_train_texts)), test_size=0.15, random_state=42, stratify=y_train.to_numpy()
    )
    fu_val = fit_tfidf_vectorizer(create_feature_union_vectorizer(), X_train_texts.iloc[tr_sub_idx])
    X_tr_sub = transform_tfidf(fu_val, X_train_texts.iloc[tr_sub_idx])
    X_val_sub = transform_tfidf(fu_val, X_train_texts.iloc[val_sub_idx])
    y_tr_sub = y_train.iloc[tr_sub_idx].to_numpy()
    y_val_sub = y_train.iloc[val_sub_idx].to_numpy()

    val_clf = clone(v2_model)
    val_clf.fit(X_tr_sub, y_tr_sub)
    val_probs = _get_abusive_probs(val_clf, X_val_sub)
    val_pr_auc = float(average_precision_score(y_val_sub, val_probs))

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_val_sub, val_probs)
    p_valid = prec_arr[:-1]
    r_valid = rec_arr[:-1]
    f1_valid = np.where((p_valid + r_valid) > 0, 2.0 * p_valid * r_valid / (p_valid + r_valid), 0.0)

    idx_bal = int(np.argmax(f1_valid))
    t_bal = float(thresh_arr[idx_bal])

    strict_candidates = np.where((r_valid >= 0.92) & (thresh_arr < t_bal))[0]
    idx_strict = int(strict_candidates[np.argmax(f1_valid[strict_candidates])])
    t_strict = float(thresh_arr[idx_strict])

    # Lenient: High Precision target >= 92% on validation
    lenient_candidates = np.where((p_valid >= 0.92) & (thresh_arr > t_bal))[0]
    if len(lenient_candidates) == 0:
        lenient_candidates = np.where((p_valid >= 0.90) & (thresh_arr > t_bal))[0]
    idx_lenient = int(lenient_candidates[np.argmax(r_valid[lenient_candidates])])
    t_lenient = float(thresh_arr[idx_lenient])

    thresholds_payload = {
        "calibrated_on_val_samples": int(len(y_val_sub)),
        "train_sub_samples": int(len(y_tr_sub)),
        "winner_model": "Calibrated LinearSVC (C=0.5)",
        "val_pr_auc": val_pr_auc,
        "thresholds": {
            "strict": {
                "threshold": round(t_strict, 4),
                "target": "High Recall (>= 92% validation recall)",
                "val_precision": float(p_valid[idx_strict]),
                "val_recall": float(r_valid[idx_strict]),
                "val_f1": float(f1_valid[idx_strict]),
            },
            "balanced": {
                "threshold": round(t_bal, 4),
                "target": "Max Validation F1-Score",
                "val_precision": float(p_valid[idx_bal]),
                "val_recall": float(r_valid[idx_bal]),
                "val_f1": float(f1_valid[idx_bal]),
            },
            "lenient": {
                "threshold": round(t_lenient, 4),
                "target": "High Precision (>= 92% validation precision)",
                "val_precision": float(p_valid[idx_lenient]),
                "val_recall": float(r_valid[idx_lenient]),
                "val_f1": float(f1_valid[idx_lenient]),
            },
        },
    }
    with open(v2_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_payload, f, indent=2)

    _plot_pr_thresholds(
        prec_arr,
        rec_arr,
        thresholds_payload["thresholds"],
        val_pr_auc,
        results_dir / "pr_curve_thresholds.png",
    )

    # Evaluate frozen v2 model on 31,915 test set with updated thresholds
    X_test_vec = transform_tfidf(v2_vec, X_test_texts)
    y_test_arr = y_test.to_numpy()
    test_probs = _get_abusive_probs(v2_model, X_test_vec)
    test_roc_auc = float(roc_auc_score(y_test_arr, test_probs))
    test_pr_auc = float(average_precision_score(y_test_arr, test_probs))

    eval_default = _evaluate_at_threshold(y_test_arr, test_probs, 0.5)
    eval_strict = _evaluate_at_threshold(y_test_arr, test_probs, thresholds_payload["thresholds"]["strict"]["threshold"])
    eval_balanced = _evaluate_at_threshold(y_test_arr, test_probs, thresholds_payload["thresholds"]["balanced"]["threshold"])
    eval_lenient = _evaluate_at_threshold(y_test_arr, test_probs, thresholds_payload["thresholds"]["lenient"]["threshold"])

    v2_summary = {
        "selected_best_model": "Calibrated LinearSVC (C=0.5)",
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_dimensions": int(X_test_vec.shape[1]),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test_arr)),
        "roc_auc": test_roc_auc,
        "pr_auc": test_pr_auc,
        "test_metrics_by_mode": {
            "default_0.50": eval_default,
            "strict": eval_strict,
            "balanced": eval_balanced,
            "lenient": eval_lenient,
        },
    }
    with open(v2_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, indent=2)
    with open(root_models_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, indent=2)

    print(
        f"[OK] Updated Thresholds -> Strict={t_strict:.4f} (Test Rec={eval_strict['recall']*100:.2f}%, Prec={eval_strict['precision']*100:.2f}%) | "
        f"Balanced={t_bal:.4f} (Test F1={eval_balanced['f1']*100:.2f}%) | "
        f"Lenient={t_lenient:.4f} (Test Prec={eval_lenient['precision']*100:.2f}%, Rec={eval_lenient['recall']*100:.2f}%)"
    )


if __name__ == "__main__":
    main()

