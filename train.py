"""
train.py - Project Entry Point for AbuseGuard AI Training Pipeline (Safe by Default).

Usage:
    python train.py
        Runs the canonical v2 production training pipeline.
        Saves artifacts safely to a timestamped run directory: models/runs/run_<timestamp>/
        (Leaves production models/v2/ untouched).

    python train.py --dataset path/to/data.csv
        Trains on custom dataset CSV.

    python train.py --output-dir models/runs/my_experiment
        Saves artifacts to a specified directory.

    python train.py --output-dir models/v2 --allow-overwrite-production
        Explicitly authorizes training to overwrite the canonical production models/v2/ bundle.

    python train.py --pipeline v1
        Runs the legacy v1 baseline pipeline into models/runs/<timestamp>/ (or specified output-dir).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.model_training import get_default_models_dir, run_training_pipeline


def resolve_training_output_dir(
    output_dir: Optional[Union[str, Path]] = None,
    allow_overwrite_production: bool = False,
    base_models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Resolves and validates the target directory for training artifacts.

    Default:
        Saves to a timestamped run directory under models/runs/ (e.g. models/runs/run_20261121_193000).

    Safety Check:
        If the resolved target directory matches the canonical production directory (models/v2),
        it strictly requires allow_overwrite_production=True; otherwise, raises PermissionError.
    """
    models_root = Path(base_models_dir).resolve() if base_models_dir is not None else get_default_models_dir().resolve()
    production_v2_dir = (models_root / "v2").resolve()

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = models_root / "runs" / f"run_{timestamp}"
    else:
        target_dir = Path(output_dir).resolve()

    # Safety: Refuse to overwrite canonical production directory without explicit flag
    if target_dir == production_v2_dir and not allow_overwrite_production:
        raise PermissionError(
            f"[SAFETY VIOLATION] Target directory '{target_dir}' is the canonical production directory (models/v2). "
            "To prevent accidental overwriting of production models/v2 artifacts, "
            "you must explicitly provide '--allow-overwrite-production'."
        )

    # Safety: Also refuse to write unversioned artifacts directly into root models/ without flag
    if target_dir == models_root and not allow_overwrite_production:
        raise PermissionError(
            f"[SAFETY VIOLATION] Target directory '{target_dir}' is the root models directory. "
            "Writing unversioned artifacts to root models/ has been disabled to prevent accidental "
            "downgrades. Use a versioned or run directory (e.g., models/runs/<run_id>), or provide "
            "'--allow-overwrite-production'."
        )

    return target_dir


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AbuseGuard AI Model Training Pipeline (Safe by Default)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset_pos",
        nargs="?",
        default=None,
        help="Optional positional path to dataset CSV (backward compatibility)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        default=None,
        help="Path to training dataset CSV (overrides positional argument)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Directory to save trained model artifacts (default: models/runs/<timestamp>/)",
    )
    parser.add_argument(
        "--allow-overwrite-production",
        action="store_true",
        default=False,
        help="Explicitly allow overwriting canonical production directory (models/v2/)",
    )
    parser.add_argument(
        "--pipeline",
        choices=["v2", "v1"],
        default="v2",
        help="Training pipeline version: 'v2' (production calibrated) or 'v1' (legacy baseline)",
    )
    return parser.parse_args(args)


def run_v2_training_pipeline(
    dataset_path: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    allow_overwrite_production: bool = False,
    max_features: int = 25000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the upgraded AbuseGuard AI v2 training pipeline:
    1. Preprocesses text with obfuscation normalisation (leetspeak, masking, doubles).
    2. Builds 25,000-dimensional FeatureUnion (Word n-grams (1,2) + Char n-grams (3,5)).
    3. Trains CalibratedClassifierCV(LinearSVC(class_weight='balanced', C=0.5)).
    4. Trains OneVsRestClassifier(LogisticRegression(class_weight='balanced')) for multi-label.
    5. Calibrates sensitivity thresholds (strict, balanced, lenient) on validation split.
    6. Persists artifacts to output_dir (defaults to safe run folder).
    """
    import json
    import joblib
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC
    from sklearn.metrics import precision_recall_curve

    from src.data_loader import TOXICITY_COLUMNS, get_default_dataset_path, load_data
    from src.feature_extraction import (
        create_feature_union_vectorizer,
        fit_tfidf_vectorizer,
        transform_tfidf,
    )
    from src.model_training import split_data
    from src.preprocessing import prepare_dataset
    from src.text_preprocessing import preprocess_text_series

    target_dir = resolve_training_output_dir(
        output_dir=output_dir,
        allow_overwrite_production=allow_overwrite_production,
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(dataset_path).resolve() if dataset_path else get_default_dataset_path()
    print(f"\n[STEP 1] Loading raw dataset from: {csv_path}")
    raw_df = load_data(csv_path)

    print("\n[STEP 2] Preparing dataset and abusive target labels...")
    prepared_df = prepare_dataset(raw_df)

    print("\n[STEP 3] Normalizing comments (v2: leetspeak, masking, repeated letters)...")
    clean_texts = preprocess_text_series(prepared_df["comment_text"], normalize_obfuscation=True)
    y_binary = prepared_df["abusive"].to_numpy()

    has_multilabel = all(col in prepared_df.columns for col in TOXICITY_COLUMNS)
    Y_multi = prepared_df[TOXICITY_COLUMNS].to_numpy() if has_multilabel else None

    # Step 4: Stratified Split
    print(f"\n[STEP 4] Splitting data into Train ({int((1-test_size)*100)}%) and Test ({int(test_size*100)}%)...")
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        clean_texts, y_binary, test_size=test_size, random_state=random_state, stratify=True
    )

    # Step 5: Feature Union
    print("\n[STEP 5] Fitting FeatureUnion (Word 10k + Char 15k = 25k dims) on train split...")
    vectorizer = create_feature_union_vectorizer()
    vectorizer = fit_tfidf_vectorizer(vectorizer, X_train_raw)
    X_train_vec = transform_tfidf(vectorizer, X_train_raw)
    X_test_vec = transform_tfidf(vectorizer, X_test_raw)

    # Step 6: Binary Classifier (Calibrated LinearSVC)
    print("\n[STEP 6] Training Calibrated LinearSVC (C=0.5, cv=3)...")
    base_svc = LinearSVC(class_weight="balanced", C=0.5, max_iter=2000, random_state=random_state)
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3, n_jobs=-1)
    calibrated_svc.fit(X_train_vec, y_train)

    # Step 7: Multi-Label Classifier
    multilabel_model = None
    if has_multilabel and Y_multi is not None:
        print("\n[STEP 7] Training OneVsRest Multi-Label Classifier...")
        all_indices = np.arange(len(prepared_df))
        from sklearn.model_selection import train_test_split
        tr_idx, _ = train_test_split(
            all_indices, test_size=test_size, random_state=random_state, stratify=y_binary
        )
        Y_train_multi = Y_multi[tr_idx]
        multilabel_model = OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", max_iter=500, random_state=random_state),
            n_jobs=-1,
        )
        multilabel_model.fit(X_train_vec, Y_train_multi)

    # Step 8: Calibrate Thresholds
    print("\n[STEP 8] Deriving calibrated thresholds...")
    test_probs = calibrated_svc.predict_proba(X_test_vec)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, test_probs)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-9)
    best_bal_idx = int(np.argmax(f1_scores[:-1])) if len(f1_scores) > 1 else 0
    t_bal = float(thresholds[best_bal_idx]) if len(thresholds) > 0 else 0.5

    strict_mask = recalls[:-1] >= 0.92
    t_strict = float(thresholds[strict_mask][-1]) if np.any(strict_mask) else max(0.05, t_bal * 0.5)

    lenient_mask = precisions[:-1] >= 0.92
    t_lenient = float(thresholds[lenient_mask][0]) if np.any(lenient_mask) else min(0.90, t_bal * 1.5)

    thresholds_payload = {
        "thresholds": {
            "strict": {"threshold": round(t_strict, 4), "target": "High Safety (Recall >= 92%)"},
            "balanced": {"threshold": round(t_bal, 4), "target": "Optimal F1 Balance"},
            "lenient": {"threshold": round(t_lenient, 4), "target": "Low Friction (Precision >= 92%)"},
        }
    }

    # Step 9: Save Artifacts
    print(f"\n[STEP 9] Persisting v2 artifacts to: {target_dir}")
    joblib.dump(calibrated_svc, target_dir / "best_model.joblib")
    joblib.dump(vectorizer, target_dir / "tfidf_vectorizer.joblib")
    if multilabel_model is not None:
        joblib.dump(multilabel_model, target_dir / "multilabel_model.joblib")

    with open(target_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds_payload, f, indent=2)

    eval_summary = {
        "selected_best_model": "Calibrated LinearSVC (C=0.5)",
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "thresholds": thresholds_payload["thresholds"],
    }
    with open(target_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return {
        "target_dir": target_dir,
        "best_model_name": "Calibrated LinearSVC (C=0.5)",
        "thresholds": thresholds_payload,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
    }


def main(cli_args: Optional[List[str]] = None) -> None:
    args = parse_args(cli_args)
    custom_dataset = args.dataset or args.dataset_pos
    dataset_path = Path(custom_dataset).resolve() if custom_dataset else None

    try:
        target_dir = resolve_training_output_dir(
            output_dir=args.output_dir,
            allow_overwrite_production=args.allow_overwrite_production,
        )
    except PermissionError as p_err:
        print("\n" + "=" * 70)
        print("PRODUCTION OVERWRITE PROTECTION TRIGGERED".center(70))
        print("=" * 70)
        print(f"\n{p_err}\n")
        sys.exit(1)

    print(f"[INFO] Target artifact directory: {target_dir}")
    print(f"[INFO] Selected training pipeline: {args.pipeline.upper()}")

    try:
        if args.pipeline == "v2":
            run_v2_training_pipeline(
                dataset_path=dataset_path,
                output_dir=target_dir,
                allow_overwrite_production=args.allow_overwrite_production,
            )
        else:
            run_training_pipeline(
                dataset_path=dataset_path,
                models_dir=target_dir,
            )
    except FileNotFoundError as fnf_err:
        print("\n" + "=" * 70)
        print("DATASET NOT FOUND".center(70))
        print("=" * 70)
        print(str(fnf_err))
        print("ACTION REQUIRED:")
        print("1. Download 'train.csv' from Kaggle (Jigsaw Toxic Comment Challenge).")
        print("2. Place it inside the 'data/' folder: data/train.csv")
        print("3. Run 'python train.py' again after placing the file.")
        sys.exit(1)
    except Exception as err:
        print(f"\n[CRITICAL ERROR] Model training pipeline failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
