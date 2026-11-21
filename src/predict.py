"""
predict.py - Reusable Prediction & Explainability Module for Live Inference (v1 & v2).

Preserves 100% backward compatibility for:
- `get_prediction_pipeline(models_dir=None)`
- `predict_comment(raw_text, model, vectorizer)`

And adds v2 support for:
- Toggleable text de-obfuscation (`normalize_obfuscation`)
- Calibrated decision thresholds (Strict / Balanced / Lenient)
- Multi-label toxicity sub-type prediction (6 Jigsaw labels)
- Explicit toggleable Roman-script Hinglish rule layer (`enable_hinglish`)
- Token/subword contribution explanation (`TF-IDF value * model coefficient`)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np

from .data_loader import TOXICITY_COLUMNS
from .english_negation import apply_english_negation_scoping
from .feature_extraction import get_feature_names, transform_tfidf
from .hinglish_detector import detect_hinglish_abuse
from .model_training import get_default_models_dir, load_model_artifacts
from .text_preprocessing import preprocess_text


def get_prediction_pipeline(
    models_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Any]:
    """
    Loads and returns the trained model and fitted vectorizer.
    Defaults to the canonical production directory (models/v2/) to ensure consistency
    with api.py and app.py, with no silent fallback to legacy V1.
    """
    target_dir = Path(models_dir) if models_dir is not None else (get_default_models_dir() / "v2")
    return load_model_artifacts(models_dir=target_dir)


def load_v2_bundle(
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Loads the complete v2 inference bundle from models/v2/ (or custom dir):
    - best_model.joblib
    - tfidf_vectorizer.joblib
    - multilabel_model.joblib (if present)
    - thresholds.json (if present)
    """
    target_dir = Path(models_dir) if models_dir else (get_default_models_dir() / "v2")
    model, vectorizer = load_model_artifacts(models_dir=target_dir)

    multilabel_path = target_dir / "multilabel_model.joblib"
    multilabel_model = joblib.load(multilabel_path) if multilabel_path.is_file() else None

    thresholds_path = target_dir / "thresholds.json"
    thresholds: Dict[str, Any] = {}
    if thresholds_path.is_file():
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = json.load(f)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "multilabel_model": multilabel_model,
        "thresholds": thresholds,
    }


def _extract_linear_weights(model: Any) -> Optional[np.ndarray]:
    """Extracts 1D linear feature weights from LR, SGD, LinearSVC, or CalibratedClassifierCV."""
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        return coef[0].ravel() if coef.ndim > 1 else coef.ravel()

    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        coefs = []
        for cal in model.calibrated_classifiers_:
            est = getattr(cal, "estimator", None)
            if est is not None and hasattr(est, "coef_"):
                c = np.asarray(est.coef_)
                coefs.append(c[0].ravel() if c.ndim > 1 else c.ravel())
        if coefs:
            return np.mean(coefs, axis=0)

    if hasattr(model, "feature_log_prob_"):
        log_probs = np.asarray(model.feature_log_prob_)
        if log_probs.shape[0] >= 2:
            # MultinomialNB: log P(w|1) - log P(w|0)
            return (log_probs[1] - log_probs[0]).ravel()

    return None


def _format_feature_label(raw_name: str) -> str:
    """Makes FeatureUnion names human-readable (e.g. 'word__idiot' -> 'idiot [word]')."""
    if raw_name.startswith("word__"):
        return f"{raw_name[6:]} [word]"
    if raw_name.startswith("char__"):
        return f"'{raw_name[6:]}' [char]"
    return raw_name


def explain_prediction(
    X_sparse: Any,
    model: Any,
    vectorizer: Any,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Computes top contributing words/subwords for a single vectorized comment via
    `TF-IDF value * model coefficient`.
    Positive contributions push toward Abusive (1); negative push toward Safe (0).
    """
    weights = _extract_linear_weights(model)
    if weights is None:
        return []

    feature_names = get_feature_names(vectorizer)
    row = X_sparse.getrow(0)
    indices = row.indices
    values = row.data

    if len(indices) == 0:
        return []

    contribs = []
    for idx, tfidf_val in zip(indices, values):
        w = float(weights[idx])
        c = float(tfidf_val) * w
        if abs(c) < 1e-9:
            continue
        raw_feat = str(feature_names[idx])
        contribs.append({
            "feature": _format_feature_label(raw_feat),
            "raw_feature": raw_feat,
            "tfidf_value": float(tfidf_val),
            "coefficient": w,
            "contribution": c,
            "direction": "Abusive (+)" if c > 0 else "Safe (-)",
        })

    contribs.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return contribs[:top_k]


def predict_comment(
    raw_text: str,
    model: Any,
    vectorizer: Any,
    normalize_obfuscation: bool = True,
    threshold: Optional[float] = None,
    multilabel_model: Optional[Any] = None,
    enable_hinglish: bool = False,
    explain: bool = False,
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Executes live inference on a single social media comment.

    Preserves full backward compatibility when called with `(raw_text, model, vectorizer)`.
    """
    if raw_text is None or not str(raw_text).strip():
        raise ValueError("[ERROR] Cannot analyze an empty or whitespace-only comment.")

    raw_str = str(raw_text)
    cleaned = preprocess_text(raw_str, normalize_obfuscation=normalize_obfuscation)

    hinglish_info = (
        detect_hinglish_abuse(raw_str)
        if enable_hinglish
        else {
            "hinglish_flagged": False,
            "matched_terms": [],
            "matched_categories": [],
            "rule_score": 0.0,
        }
    )

    # Empty after cleaning (e.g., URLs/punctuation only)
    if not cleaned.strip():
        ml_abusive_prob = 0.0
        final_abusive_prob = max(ml_abusive_prob, hinglish_info["rule_score"]) if hinglish_info["hinglish_flagged"] else 0.0
        eff_thresh = 0.5 if threshold is None else float(threshold)
        is_abusive = bool(final_abusive_prob >= eff_thresh) if hinglish_info["hinglish_flagged"] else False
        return {
            "is_abusive": is_abusive,
            "prediction_label": 1 if is_abusive else 0,
            "prediction_text": "Abusive" if is_abusive else "Safe",
            "safe_probability": 1.0 - final_abusive_prob,
            "abusive_probability": final_abusive_prob,
            "ml_abusive_probability": ml_abusive_prob,
            "threshold_used": eff_thresh,
            "cleaned_text": "",
            "raw_text": raw_str,
            "active_categories": list(hinglish_info["matched_categories"]),
            "category_probabilities": {},
            "hinglish_match": hinglish_info,
            "hinglish_suppressed_matches": hinglish_info.get("suppressed_matches", []),
            "needs_human_review": bool(hinglish_info.get("needs_human_review", False)),
            "top_features": [],
            "note": "Text contained only URLs or punctuation with no extractable words.",
        }

    # Vectorize
    X_sparse = transform_tfidf(vectorizer, [cleaned])

    # Model probability
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_sparse)[0]
        ml_safe_prob = float(probs[0])
        ml_abusive_prob = float(probs[1])
    else:
        pred_raw = int(model.predict(X_sparse)[0])
        ml_abusive_prob = 1.0 if pred_raw == 1 else 0.0
        ml_safe_prob = 1.0 - ml_abusive_prob

    # Apply threshold (exact model.predict() when threshold is None and no Hinglish hit)
    if threshold is None:
        eff_thresh = 0.5
    else:
        eff_thresh = float(threshold)

    # Apply English negation scoping layer (Fix 2.2)
    english_negation_info = apply_english_negation_scoping(
        text=raw_str,
        ml_abusive_prob=ml_abusive_prob,
        model=model,
        vectorizer=vectorizer,
        threshold=eff_thresh,
    )

    effective_ml_prob = (
        english_negation_info["adjusted_score"]
        if english_negation_info["adjustment_applied"]
        else ml_abusive_prob
    )

    # Combine with Hinglish rule layer if enabled
    if hinglish_info["hinglish_flagged"]:
        rule_score = float(hinglish_info.get("rule_score", 0.0))
        final_abusive_prob = max(effective_ml_prob, rule_score)
        final_safe_prob = max(0.0, 1.0 - final_abusive_prob)
        is_abusive = final_abusive_prob >= eff_thresh
    else:
        final_abusive_prob = effective_ml_prob
        final_safe_prob = max(0.0, 1.0 - final_abusive_prob)
        is_abusive = final_abusive_prob >= eff_thresh

    pred_label = 1 if is_abusive else 0
    label_text = "Abusive" if is_abusive else "Safe"

    # Multi-label sub-types
    category_probs: Dict[str, float] = {}
    active_categories: List[str] = []
    if multilabel_model is not None:
        ml_probs_vec = multilabel_model.predict_proba(X_sparse)[0]
        ml_preds_vec = multilabel_model.predict(X_sparse)[0]
        for col, prob_val, pred_val in zip(TOXICITY_COLUMNS, ml_probs_vec, ml_preds_vec):
            category_probs[col] = float(prob_val)
            if is_abusive and (int(pred_val) == 1 or float(prob_val) >= 0.5):
                active_categories.append(col)

    # Include Hinglish matched categories cleanly if flagged (active abuse only)
    if hinglish_info["hinglish_flagged"] and is_abusive:
        for cat in hinglish_info["matched_categories"]:
            if cat not in active_categories:
                active_categories.append(cat)

    top_features = explain_prediction(X_sparse, model, vectorizer, top_k=top_k) if explain else []

    return {
        "is_abusive": is_abusive,
        "prediction_label": pred_label,
        "prediction_text": label_text,
        "safe_probability": final_safe_prob,
        "abusive_probability": final_abusive_prob,
        "ml_abusive_probability": ml_abusive_prob,
        "threshold_used": eff_thresh,
        "cleaned_text": cleaned,
        "raw_text": raw_str,
        "active_categories": active_categories,
        "category_probabilities": category_probs,
        "hinglish_match": hinglish_info,
        "hinglish_suppressed_matches": hinglish_info.get("suppressed_matches", []),
        "english_negation_info": english_negation_info,
        "needs_human_review": bool(hinglish_info.get("needs_human_review", False)),
        "top_features": top_features,
    }
