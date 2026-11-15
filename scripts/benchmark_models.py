"""
benchmark_models.py - Phase 2: 4-Model 5-Fold Stratified CV, Validation Threshold
Calibration (Strict / Balanced / Lenient), Final v2 Training, and Multi-Label Classifier.

Zero Data Leakage Discipline:
- Exact 80/20 stratified train/test split (random_state=42): 127,656 train / 31,915 test.
- 5-fold Stratified CV runs strictly inside the 127,656 training set, fitting the
  FeatureUnion vectorizer fresh inside every fold.
- Threshold calibration (Strict / Balanced / Lenient) runs strictly on an 85/15
  stratified split of the 127,656 training set (~108,507 train_sub / ~19,149 val).
- The 31,915 held-out test set is evaluated ONCE at the end with frozen artifacts.

Usage:
    python scripts/benchmark_models.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import TOXICITY_COLUMNS, get_default_dataset_path, load_data
from src.feature_extraction import (
    create_feature_union_vectorizer,
    fit_tfidf_vectorizer,
    transform_tfidf,
)
from src.model_training import evaluate_model, plot_confusion_matrix, split_data
from src.preprocessing import prepare_dataset
from src.text_preprocessing import preprocess_text_series


def _build_candidate_factories() -> List[Tuple[str, str, Callable[[], Any]]]:
    """Returns (display_name, family_name, factory_fn) for all 9 benchmark configs."""
    return [
        (
            "LogisticRegression (C=0.5)",
            "LogisticRegression",
            lambda: LogisticRegression(class_weight="balanced", C=0.5, max_iter=1000, random_state=42),
        ),
        (
            "LogisticRegression (C=1.0)",
            "LogisticRegression",
            lambda: LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42),
        ),
        (
            "LogisticRegression (C=2.0)",
            "LogisticRegression",
            lambda: LogisticRegression(class_weight="balanced", C=2.0, max_iter=1000, random_state=42),
        ),
        (
            "Calibrated LinearSVC (C=0.5)",
            "LinearSVC (Calibrated)",
            lambda: CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", C=0.5, max_iter=2000, random_state=42),
                cv=3,
                n_jobs=-1,
            ),
        ),
        (
            "Calibrated LinearSVC (C=1.0)",
            "LinearSVC (Calibrated)",
            lambda: CalibratedClassifierCV(
                estimator=LinearSVC(class_weight="balanced", C=1.0, max_iter=2000, random_state=42),
                cv=3,
                n_jobs=-1,
            ),
        ),
        (
            "SGDClassifier (alpha=1e-4)",
            "SGDClassifier",
            lambda: SGDClassifier(
                loss="log_loss", class_weight="balanced", alpha=1e-4, max_iter=1000, random_state=42
            ),
        ),
        (
            "SGDClassifier (alpha=1e-5)",
            "SGDClassifier",
            lambda: SGDClassifier(
                loss="log_loss", class_weight="balanced", alpha=1e-5, max_iter=1000, random_state=42
            ),
        ),
        (
            "ComplementNB (alpha=0.5)",
            "ComplementNB",
            lambda: ComplementNB(alpha=0.5),
        ),
        (
            "ComplementNB (alpha=1.0)",
            "ComplementNB",
            lambda: ComplementNB(alpha=1.0),
        ),
    ]


def _get_abusive_probs(model: Any, X: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


def _evaluate_at_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, preds, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, preds, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def _plot_cv_comparison(cv_rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Plots grouped comparison of Precision, Recall, F1, and PR-AUC across CV candidates."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = [r["model"] for r in cv_rows]
    f1s = [r["f1_mean"] * 100 for r in cv_rows]
    recs = [r["recall_mean"] * 100 for r in cv_rows]
    precs = [r["precision_mean"] * 100 for r in cv_rows]
    pr_aucs = [r["pr_auc_mean"] * 100 for r in cv_rows]

    y_pos = np.arange(len(names))
    height = 0.2

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(y_pos + 1.5 * height, f1s, height, label="Abusive F1 (%)", color="#4f46e5")
    ax.barh(y_pos + 0.5 * height, recs, height, label="Abusive Recall (%)", color="#16a34a")
    ax.barh(y_pos - 0.5 * height, precs, height, label="Abusive Precision (%)", color="#f59e0b")
    ax.barh(y_pos - 1.5 * height, pr_aucs, height, label="PR-AUC (%)", color="#0ea5e9")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Score (%)", fontsize=11, fontweight="bold")
    ax.set_title("5-Fold Stratified CV Model Benchmark (127,656 Training Comments)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(40, 102)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", frameon=True)

    for idx, val in enumerate(f1s):
        ax.text(val + 0.5, y_pos[idx] + 1.5 * height, f"{val:.2f}%", va="center", fontsize=8.5, fontweight="bold", color="#312e81")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_pr_thresholds(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds_meta: Dict[str, Dict[str, Any]],
    pr_auc_val: float,
    output_path: Path,
) -> None:
    """Plots Validation Precision-Recall curve with Strict / Balanced / Lenient operating points."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(recalls, precisions, color="#4f46e5", lw=2.2, label=f"Validation PR Curve (PR-AUC = {pr_auc_val:.4f})")

    markers = [
        ("strict", "Strict (High Recall)", "#dc2626", "s"),
        ("balanced", "Balanced (Max F1)", "#16a34a", "o"),
        ("lenient", "Lenient (High Precision)", "#d97706", "^"),
    ]
    for key, label, color, marker in markers:
        m = thresholds_meta[key]
        ax.scatter(
            m["val_recall"],
            m["val_precision"],
            color=color,
            marker=marker,
            s=110,
            zorder=5,
            edgecolors="black",
            linewidths=0.8,
            label=f"{label} @ t={m['threshold']:.3f} (P={m['val_precision']*100:.1f}%, R={m['val_recall']*100:.1f}%, F1={m['val_f1']*100:.1f}%)",
        )

    ax.set_title("Validation Precision-Recall Curve & Operating Thresholds", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlabel("Abusive Recall", fontsize=11, fontweight="bold")
    ax.set_ylabel("Abusive Precision", fontsize=11, fontweight="bold")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower left", fontsize=8.5, frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_v1_vs_v2_confusion_matrices(
    cm_v1: np.ndarray,
    cm_v2: np.ndarray,
    v1_label: str,
    v2_label: str,
    output_path: Path,
) -> None:
    """Plots side-by-side 2x2 confusion matrix heatmaps for v1 vs v2 on the 31,915 test set."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    classes = ["Safe (0)", "Abusive (1)"]

    for ax, cm, title, cmap in [
        (axes[0], cm_v1, v1_label, "Oranges"),
        (axes[1], cm_v2, v2_label, "Blues"),
    ]:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = np.arange(2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(classes, fontsize=9.5, fontweight="bold")
        ax.set_yticklabels(classes, fontsize=9.5, fontweight="bold")

        labels_matrix = [
            [f"TN\n{cm[0, 0]:,}", f"FP\n{cm[0, 1]:,}"],
            [f"FN\n{cm[1, 0]:,}", f"TP\n{cm[1, 1]:,}"],
        ]
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    labels_matrix[i][j],
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=10.5,
                    fontweight="bold",
                )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted", fontsize=10, fontweight="bold")
        ax.set_ylabel("True", fontsize=10, fontweight="bold")

    fig.suptitle("Held-Out Test Set Confusion Matrices (n = 31,915): v1 vs v2", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    t_start = time.time()
    results_dir = PROJECT_ROOT / "results"
    v1_dir = PROJECT_ROOT / "models" / "v1"
    v2_dir = PROJECT_ROOT / "models" / "v2"
    root_models_dir = PROJECT_ROOT / "models"
    results_dir.mkdir(parents=True, exist_ok=True)
    v2_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & prepare Jigsaw dataset
    print("[1/6] Loading and preparing Jigsaw dataset...")
    raw_df = load_data(get_default_dataset_path())
    prepared_df = prepare_dataset(raw_df)

    print("[2/6] Applying v2 text preprocessing (with de-obfuscation & negation preservation)...")
    clean_texts = preprocess_text_series(prepared_df["comment_text"], normalize_obfuscation=True)
    y_binary = prepared_df["abusive"].to_numpy()
    Y_multilabel = prepared_df[TOXICITY_COLUMNS].to_numpy()

    # Exact 80/20 stratified split (random_state=42) matching split_data()
    all_indices = np.arange(len(prepared_df))
    train_idx, test_idx = train_test_split(
        all_indices, test_size=0.2, random_state=42, stratify=y_binary
    )

    # Verify exact parity with src.model_training.split_data
    X_tr_check, X_te_check, y_tr_check, y_te_check = split_data(
        clean_texts, prepared_df["abusive"], test_size=0.2, random_state=42, stratify=True
    )
    assert len(train_idx) == len(X_tr_check) == 127656
    assert len(test_idx) == len(X_te_check) == 31915
    assert np.array_equal(clean_texts.iloc[train_idx].to_numpy(), X_tr_check.to_numpy())
    assert np.array_equal(y_binary[train_idx], y_tr_check.to_numpy())

    X_train_texts = clean_texts.iloc[train_idx].to_numpy()
    X_test_texts = clean_texts.iloc[test_idx].to_numpy()
    y_train = y_binary[train_idx]
    y_test = y_binary[test_idx]
    Y_train_multi = Y_multilabel[train_idx]
    Y_test_multi = Y_multilabel[test_idx]

    print(f"[OK] Exact split verified: Train={len(X_train_texts):,} | Test={len(X_test_texts):,}")

    # 2. 5-Fold Stratified CV inside 127,656 training set
    print("[3/6] Running 5-fold Stratified CV across 4 model families (9 configurations)...")
    candidates = _build_candidate_factories()
    fold_metrics: Dict[str, List[Dict[str, float]]] = {name: [] for name, _, _ in candidates}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold_num, (f_tr_idx, f_va_idx) in enumerate(skf.split(X_train_texts, y_train), 1):
        t_fold = time.time()
        fu_fold = create_feature_union_vectorizer()
        fu_fold = fit_tfidf_vectorizer(fu_fold, X_train_texts[f_tr_idx])
        X_f_tr = transform_tfidf(fu_fold, X_train_texts[f_tr_idx])
        X_f_va = transform_tfidf(fu_fold, X_train_texts[f_va_idx])
        y_f_tr = y_train[f_tr_idx]
        y_f_va = y_train[f_va_idx]

        for name, _, factory in candidates:
            t_m = time.time()
            clf = factory()
            clf.fit(X_f_tr, y_f_tr)
            fit_sec = time.time() - t_m

            probs = _get_abusive_probs(clf, X_f_va)
            preds = (probs >= 0.5).astype(int)

            fold_metrics[name].append({
                "accuracy": float(accuracy_score(y_f_va, preds)),
                "precision": float(precision_score(y_f_va, preds, pos_label=1, zero_division=0)),
                "recall": float(recall_score(y_f_va, preds, pos_label=1, zero_division=0)),
                "f1": float(f1_score(y_f_va, preds, pos_label=1, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_f_va, probs)),
                "pr_auc": float(average_precision_score(y_f_va, probs)),
                "fit_sec": float(fit_sec),
            })

        print(f"  - Fold {fold_num}/5 completed in {time.time() - t_fold:.1f}s")

    cv_summary_rows = []
    for name, family, _ in candidates:
        rows = fold_metrics[name]
        entry = {
            "model": name,
            "family": family,
            "accuracy_mean": float(np.mean([r["accuracy"] for r in rows])),
            "accuracy_std": float(np.std([r["accuracy"] for r in rows])),
            "precision_mean": float(np.mean([r["precision"] for r in rows])),
            "precision_std": float(np.std([r["precision"] for r in rows])),
            "recall_mean": float(np.mean([r["recall"] for r in rows])),
            "recall_std": float(np.std([r["recall"] for r in rows])),
            "f1_mean": float(np.mean([r["f1"] for r in rows])),
            "f1_std": float(np.std([r["f1"] for r in rows])),
            "roc_auc_mean": float(np.mean([r["roc_auc"] for r in rows])),
            "pr_auc_mean": float(np.mean([r["pr_auc"] for r in rows])),
            "fit_sec_mean": float(np.mean([r["fit_sec"] for r in rows])),
        }
        cv_summary_rows.append(entry)

    # Sort descending by mean CV F1 (tie-break by mean CV Recall)
    cv_summary_rows.sort(key=lambda r: (r["f1_mean"], r["recall_mean"]), reverse=True)
    winner_name = cv_summary_rows[0]["model"]
    winner_factory = dict((name, factory) for name, _, factory in candidates)[winner_name]

    # Save CV markdown + json + chart
    cv_md_lines = [
        "# 5-Fold Stratified Cross-Validation Benchmark (v2 FeatureUnion)",
        "",
        "- **Training Split:** 127,656 comments (80% of 159,571; stratified `random_state=42`).",
        "- **Features:** `FeatureUnion` (`word (1,2) max=10,000` + `char_wb (3,5) max=15,000`), fitted strictly inside each CV fold.",
        f"- **Winning Configuration (by Mean CV Abusive F1):** **{winner_name}** (`F1 = {cv_summary_rows[0]['f1_mean']*100:.2f}%`, `PR-AUC = {cv_summary_rows[0]['pr_auc_mean']:.4f}`).",
        "",
        "| Rank | Model Configuration | Accuracy | Abusive Precision | Abusive Recall | Abusive F1 | ROC-AUC | PR-AUC | Mean Fit Time |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for rank, r in enumerate(cv_summary_rows, 1):
        cv_md_lines.append(
            f"| {rank} | {r['model']} | {r['accuracy_mean']*100:.2f}% +/- {r['accuracy_std']*100:.2f}% | "
            f"{r['precision_mean']*100:.2f}% +/- {r['precision_std']*100:.2f}% | "
            f"{r['recall_mean']*100:.2f}% +/- {r['recall_std']*100:.2f}% | "
            f"**{r['f1_mean']*100:.2f}%** +/- {r['f1_std']*100:.2f}% | "
            f"{r['roc_auc_mean']:.4f} | {r['pr_auc_mean']:.4f} | {r['fit_sec_mean']:.2f}s |"
        )
    (results_dir / "model_comparison_cv.md").write_text("\n".join(cv_md_lines) + "\n", encoding="utf-8")
    with open(results_dir / "model_comparison_cv.json", "w", encoding="utf-8") as f:
        json.dump({"winner": winner_name, "cv_results": cv_summary_rows}, f, indent=2)
    _plot_cv_comparison(cv_summary_rows, results_dir / "model_comparison.png")
    print(f"[OK] CV Winner: {winner_name} (F1={cv_summary_rows[0]['f1_mean']*100:.2f}%)")

    # 3. Threshold Calibration via 85/15 Stratified Validation Split of Training Set
    print("[4/6] Calibrating Strict / Balanced / Lenient thresholds on 85/15 training/validation split...")
    tr_sub_idx, val_sub_idx = train_test_split(
        np.arange(len(X_train_texts)), test_size=0.15, random_state=42, stratify=y_train
    )
    fu_val = fit_tfidf_vectorizer(create_feature_union_vectorizer(), X_train_texts[tr_sub_idx])
    X_tr_sub = transform_tfidf(fu_val, X_train_texts[tr_sub_idx])
    X_val_sub = transform_tfidf(fu_val, X_train_texts[val_sub_idx])
    y_tr_sub = y_train[tr_sub_idx]
    y_val_sub = y_train[val_sub_idx]

    val_clf = winner_factory()
    val_clf.fit(X_tr_sub, y_tr_sub)
    val_probs = _get_abusive_probs(val_clf, X_val_sub)
    val_pr_auc = float(average_precision_score(y_val_sub, val_probs))

    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_val_sub, val_probs)
    p_valid = prec_arr[:-1]
    r_valid = rec_arr[:-1]
    f1_valid = np.where((p_valid + r_valid) > 0, 2.0 * p_valid * r_valid / (p_valid + r_valid), 0.0)

    # Balanced: Maximize F1 on validation split
    idx_bal = int(np.argmax(f1_valid))
    t_bal = float(thresh_arr[idx_bal])

    # Strict: High Recall target (>= 0.92 with highest F1 / threshold)
    strict_candidates = np.where((r_valid >= 0.92) & (thresh_arr < t_bal))[0]
    if len(strict_candidates) == 0:
        strict_candidates = np.where((r_valid >= 0.90) & (thresh_arr < t_bal))[0]
    idx_strict = int(strict_candidates[np.argmax(f1_valid[strict_candidates])])
    t_strict = float(thresh_arr[idx_strict])

    # Lenient: High Precision target (>= 0.85 with highest recall / lowest threshold)
    lenient_candidates = np.where((p_valid >= 0.85) & (thresh_arr > t_bal))[0]
    if len(lenient_candidates) == 0:
        lenient_candidates = np.where((p_valid >= 0.80) & (thresh_arr > t_bal))[0]
    idx_lenient = int(lenient_candidates[np.argmax(r_valid[lenient_candidates])])
    t_lenient = float(thresh_arr[idx_lenient])

    thresholds_payload = {
        "calibrated_on_val_samples": int(len(y_val_sub)),
        "train_sub_samples": int(len(y_tr_sub)),
        "winner_model": winner_name,
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
                "target": "High Precision (>= 85% validation precision)",
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
    print(
        f"[OK] Calibrated thresholds -> Strict={t_strict:.4f}, Balanced={t_bal:.4f}, Lenient={t_lenient:.4f}"
    )

    # 4. Fit Winning Model on Full 127,656 Training Set & Evaluate ONCE on 31,915 Test Set
    print("[5/6] Fitting winning model + FeatureUnion on full 127,656 training split & evaluating on 31,915 test split...")
    v2_vectorizer = fit_tfidf_vectorizer(create_feature_union_vectorizer(), X_train_texts)
    X_train_vec = transform_tfidf(v2_vectorizer, X_train_texts)
    X_test_vec = transform_tfidf(v2_vectorizer, X_test_texts)

    v2_model = winner_factory()
    v2_model.fit(X_train_vec, y_train)
    test_probs = _get_abusive_probs(v2_model, X_test_vec)
    test_roc_auc = float(roc_auc_score(y_test, test_probs))
    test_pr_auc = float(average_precision_score(y_test, test_probs))

    eval_default = _evaluate_at_threshold(y_test, test_probs, 0.5)
    eval_strict = _evaluate_at_threshold(y_test, test_probs, thresholds_payload["thresholds"]["strict"]["threshold"])
    eval_balanced = _evaluate_at_threshold(y_test, test_probs, thresholds_payload["thresholds"]["balanced"]["threshold"])
    eval_lenient = _evaluate_at_threshold(y_test, test_probs, thresholds_payload["thresholds"]["lenient"]["threshold"])

    # Load v1 baseline for side-by-side confusion matrix plot
    with open(results_dir / "v1_baseline.json", "r", encoding="utf-8") as f:
        v1_data = json.load(f)
    cm_v1 = np.array(v1_data["v1_test_metrics"]["confusion_matrix"])
    cm_v2_bal = np.array(eval_balanced["confusion_matrix"])

    _plot_v1_vs_v2_confusion_matrices(
        cm_v1,
        cm_v2_bal,
        f"v1 Baseline (F1={v1_data['v1_test_metrics']['abusive_f1']*100:.2f}%)",
        f"v2 {winner_name} Balanced (F1={eval_balanced['f1']*100:.2f}%)",
        results_dir / "confusion_matrices_v1_vs_v2.png",
    )

    # Persist v2 artifacts
    joblib.dump(v2_model, v2_dir / "best_model.joblib")
    joblib.dump(v2_vectorizer, v2_dir / "tfidf_vectorizer.joblib")
    plot_confusion_matrix(cm_v2_bal, f"v2 {winner_name} (Balanced)", v2_dir / "confusion_matrix.png")

    v2_summary = {
        "selected_best_model": winner_name,
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feature_dimensions": int(X_train_vec.shape[1]),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
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

    # Also update root models/ to serve v2 by default (keeping models/v1/ untouched!)
    joblib.dump(v2_model, root_models_dir / "best_model.joblib")
    joblib.dump(v2_vectorizer, root_models_dir / "tfidf_vectorizer.joblib")
    plot_confusion_matrix(cm_v2_bal, f"v2 {winner_name} (Balanced)", root_models_dir / "confusion_matrix.png")
    with open(root_models_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(v2_summary, f, indent=2)

    # 5. Train Multi-Label Sub-Type Classifier (6 Jigsaw categories)
    print("[6/6] Training OneVsRest Multi-Label Classifier across 6 Jigsaw sub-types...")
    multilabel_clf = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=500, random_state=42),
        n_jobs=-1,
    )
    multilabel_clf.fit(X_train_vec, Y_train_multi)
    joblib.dump(multilabel_clf, v2_dir / "multilabel_model.joblib")

    Y_test_pred = multilabel_clf.predict(X_test_vec)
    Y_test_prob = multilabel_clf.predict_proba(X_test_vec)

    per_label_rows = []
    for idx, col in enumerate(TOXICITY_COLUMNS):
        yt = Y_test_multi[:, idx]
        yp = Y_test_pred[:, idx]
        yprob = Y_test_prob[:, idx]
        per_label_rows.append({
            "label": col,
            "test_positives": int(yt.sum()),
            "prevalence_pct": float(yt.mean() * 100),
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "roc_auc": float(roc_auc_score(yt, yprob)),
            "pr_auc": float(average_precision_score(yt, yprob)),
        })

    macro_f1 = float(f1_score(Y_test_multi, Y_test_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(Y_test_multi, Y_test_pred, average="micro", zero_division=0))

    ml_lines = [
        "# Multi-Label Toxicity Sub-Type Evaluation (Held-Out Test Set: 31,915 Comments)",
        "",
        "- **Classifier:** `OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=500, random_state=42))`",
        f"- **Features:** v2 `FeatureUnion` ({X_train_vec.shape[1]:,} features: Word (1,2) + Char (3,5))",
        f"- **Aggregate Scores:** Micro-F1 = **{micro_f1*100:.2f}%** | Macro-F1 = **{macro_f1*100:.2f}%**",
        "",
        "| Toxicity Category | Test Positives | Prevalence | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in per_label_rows:
        ml_lines.append(
            f"| `{r['label']}` | {r['test_positives']:,} | {r['prevalence_pct']:.2f}% | "
            f"{r['precision']*100:.2f}% | {r['recall']*100:.2f}% | **{r['f1']*100:.2f}%** | "
            f"{r['roc_auc']:.4f} | {r['pr_auc']:.4f} |"
        )
    (results_dir / "multilabel_metrics.md").write_text("\n".join(ml_lines) + "\n", encoding="utf-8")
    with open(results_dir / "multilabel_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"micro_f1": micro_f1, "macro_f1": macro_f1, "per_label": per_label_rows},
            f,
            indent=2,
        )

    elapsed = time.time() - t_start
    print(f"[DONE] Phase 2 complete in {elapsed:.1f}s!")
    print(
        f"Test v2 Balanced ({winner_name} @{eval_balanced['threshold']:.4f}): "
        f"Acc={eval_balanced['accuracy']*100:.2f}% | Prec={eval_balanced['precision']*100:.2f}% | "
        f"Rec={eval_balanced['recall']*100:.2f}% | F1={eval_balanced['f1']*100:.2f}% | "
        f"ROC-AUC={test_roc_auc:.4f} | PR-AUC={test_pr_auc:.4f}"
    )
    print(f"Multi-Label: Micro-F1={micro_f1*100:.2f}% | Macro-F1={macro_f1*100:.2f}%")


if __name__ == "__main__":
    main()

