"""
verify_v1_baseline.py - Verifies frozen v1 model metrics on the exact 31,915 held-out test set
and inspects parameter discrepancies between src/ defaults and train.py/model_training.py.

Usage:
    python scripts/verify_v1_baseline.py
"""

import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_default_dataset_path, load_data
from src.feature_extraction import create_tfidf_vectorizer, transform_tfidf
from src.model_training import evaluate_model, run_training_pipeline, split_data
from src.preprocessing import prepare_dataset
from src.text_preprocessing import preprocess_text


def _safe_json_params(params: dict) -> dict:
    clean = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


def main() -> None:
    v1_dir = PROJECT_ROOT / "models" / "v1"
    results_path = PROJECT_ROOT / "results" / "v1_baseline.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    model_path = v1_dir / "best_model.joblib"
    vec_path = v1_dir / "tfidf_vectorizer.joblib"
    orig_summary_path = v1_dir / "evaluation_summary.json"

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)

    with open(orig_summary_path, "r", encoding="utf-8") as f:
        orig_summary = json.load(f)

    # Load dataset and reproduce exact 80/20 stratified split (random_state=42)
    raw_df = load_data(get_default_dataset_path())
    prepared_df = prepare_dataset(raw_df)

    # Check if preprocess_text accepts normalize_obfuscation (after Phase 1.2)
    sig_preprocess = inspect.signature(preprocess_text)
    if "normalize_obfuscation" in sig_preprocess.parameters:
        clean_texts = prepared_df["comment_text"].apply(
            lambda t: preprocess_text(t, normalize_obfuscation=False)
        )
    else:
        clean_texts = prepared_df["comment_text"].apply(preprocess_text)

    labels = prepared_df["abusive"]

    X_train_raw, X_test_raw, y_train, y_test = split_data(
        clean_texts, labels, test_size=0.2, random_state=42, stratify=True
    )

    X_test_vec = transform_tfidf(vectorizer, X_test_raw)
    eval_metrics = evaluate_model(model, X_test_vec, y_test, model_name="v1 Logistic Regression")

    y_probs = model.predict_proba(X_test_vec)[:, 1]
    roc_auc = float(roc_auc_score(y_test, y_probs))
    pr_auc = float(average_precision_score(y_test, y_probs))

    # Inspect src/ defaults vs actual v1 saved artifacts & train.py/run_training_pipeline
    sig_create_vec = inspect.signature(create_tfidf_vectorizer)
    sig_run_pipe = inspect.signature(run_training_pipeline)

    src_vec_default_max_features = sig_create_vec.parameters["max_features"].default
    train_pipe_default_max_features = sig_run_pipe.parameters["max_features"].default
    v1_vec_params = _safe_json_params(vectorizer.get_params())
    v1_model_params = _safe_json_params(model.get_params())

    parameter_mismatches = [
        {
            "parameter": "max_features",
            "src_feature_extraction_default": src_vec_default_max_features,
            "src_model_training_and_train_py_effective": train_pipe_default_max_features,
            "saved_v1_vectorizer_actual": v1_vec_params.get("max_features"),
            "saved_v1_vocab_size": len(vectorizer.vocabulary_),
            "note": (
                "src/feature_extraction.py create_tfidf_vectorizer() defaults to max_features=5000, "
                "whereas train.py -> run_training_pipeline() overrides max_features=10000."
            ),
        }
    ]

    orig_lr = orig_summary.get("results", {}).get("Logistic Regression", {})
    matches_saved_summary = (
        abs(eval_metrics["accuracy"] - orig_lr.get("accuracy", -1)) < 1e-9
        and abs(eval_metrics["abusive_f1"] - orig_lr.get("abusive_f1", -1)) < 1e-9
        and eval_metrics["confusion_matrix"] == orig_lr.get("confusion_matrix")
    )

    output_payload = {
        "verified_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_total_rows_raw": int(len(raw_df)),
        "dataset_total_rows_prepared": int(len(prepared_df)),
        "split": {
            "train_size": int(len(X_train_raw)),
            "test_size": int(len(X_test_raw)),
            "test_abusive_count": int((y_test == 1).sum()),
            "test_safe_count": int((y_test == 0).sum()),
            "random_state": 42,
            "stratified": True,
        },
        "v1_test_metrics": {
            "model_name": "Logistic Regression (v1)",
            "accuracy": eval_metrics["accuracy"],
            "abusive_precision": eval_metrics["abusive_precision"],
            "abusive_recall": eval_metrics["abusive_recall"],
            "abusive_f1": eval_metrics["abusive_f1"],
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "confusion_matrix": eval_metrics["confusion_matrix"],
            "true_negatives": eval_metrics["true_negatives"],
            "false_positives": eval_metrics["false_positives"],
            "false_negatives": eval_metrics["false_negatives"],
            "true_positives": eval_metrics["true_positives"],
            "matches_original_evaluation_summary_json": matches_saved_summary,
        },
        "v1_original_comparison_models": {
            name: {
                "accuracy": res.get("accuracy"),
                "abusive_precision": res.get("abusive_precision"),
                "abusive_recall": res.get("abusive_recall"),
                "abusive_f1": res.get("abusive_f1"),
                "confusion_matrix": res.get("confusion_matrix"),
            }
            for name, res in orig_summary.get("results", {}).items()
        },
        "v1_inspected_params": {
            "model_type": type(model).__name__,
            "model_params": v1_model_params,
            "vectorizer_type": type(vectorizer).__name__,
            "vectorizer_params": v1_vec_params,
            "learned_vocab_size": int(len(vectorizer.vocabulary_)),
        },
        "parameter_mismatches": parameter_mismatches,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"[OK] v1 verification complete -> {results_path}")
    print(
        f"Train={len(X_train_raw):,} | Test={len(X_test_raw):,} | "
        f"Acc={eval_metrics['accuracy']*100:.2f}% | Prec={eval_metrics['abusive_precision']*100:.2f}% | "
        f"Rec={eval_metrics['abusive_recall']*100:.2f}% | F1={eval_metrics['abusive_f1']*100:.2f}% | "
        f"ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | Matches JSON={matches_saved_summary}"
    )


if __name__ == "__main__":
    main()

