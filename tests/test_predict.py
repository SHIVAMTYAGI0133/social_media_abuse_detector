"""
test_predict.py - Unit tests for the live prediction module (Part 4).

NOTE ON TEST DATA:
=============================================================================
These tests verify the inference logic using synthetic model fixtures as well
as the trained artifacts saved in the models/ directory.
=============================================================================
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
)
from src.model_training import train_logistic_regression
from src.predict import get_prediction_pipeline, predict_comment


@pytest.fixture
def synthetic_pipeline():
    """Creates a small self-contained synthetic model and vectorizer for isolated unit tests."""
    train_texts = [
        "thank you very much for this wonderful contribution",
        "great job on editing this page nicely",
        "you are a disgusting idiot and a total loser",
        "i will hurt you and attack you violently",
    ]
    y_train = pd.Series([0, 0, 1, 1])

    vec = create_tfidf_vectorizer(min_df=1)
    fitted_vec = fit_tfidf_vectorizer(vec, train_texts)
    X = fitted_vec.transform(train_texts)
    model = train_logistic_regression(X, y_train, random_state=42)

    return model, fitted_vec


def test_predict_safe_comment(synthetic_pipeline):
    """Verifies that a polite, constructive comment is classified as Safe."""
    model, vectorizer = synthetic_pipeline
    result = predict_comment(
        "Thank you so much for this wonderful contribution to the article!",
        model=model,
        vectorizer=vectorizer,
    )

    assert result["is_abusive"] is False
    assert result["prediction_label"] == 0
    assert result["prediction_text"] == "Safe"
    assert result["safe_probability"] >= 0.50
    assert result["abusive_probability"] <= 0.50
    assert "cleaned_text" in result


def test_predict_abusive_comment(synthetic_pipeline):
    """Verifies that an aggressive, hateful comment is classified as Abusive."""
    model, vectorizer = synthetic_pipeline
    result = predict_comment(
        "You are a disgusting idiot and I will hurt you!",
        model=model,
        vectorizer=vectorizer,
    )

    assert result["is_abusive"] is True
    assert result["prediction_label"] == 1
    assert result["prediction_text"] == "Abusive"
    assert result["abusive_probability"] >= 0.50
    assert result["safe_probability"] <= 0.50


def test_predict_empty_or_whitespace_raises_value_error(synthetic_pipeline):
    """Verifies that empty strings or whitespace-only inputs raise ValueError."""
    model, vectorizer = synthetic_pipeline

    with pytest.raises(ValueError) as exc_info_empty:
        predict_comment("", model=model, vectorizer=vectorizer)
    assert "Cannot analyze an empty" in str(exc_info_empty.value)

    with pytest.raises(ValueError) as exc_info_spaces:
        predict_comment("   \n\t  ", model=model, vectorizer=vectorizer)
    assert "Cannot analyze an empty" in str(exc_info_spaces.value)


def test_predict_symbols_or_url_only(synthetic_pipeline):
    """Verifies that comments containing only punctuation or URLs are handled gracefully."""
    model, vectorizer = synthetic_pipeline
    result = predict_comment("https://example.com/test !!!", model=model, vectorizer=vectorizer)

    assert result["prediction_text"] == "Safe"
    assert result["prediction_label"] == 0
    assert result["is_abusive"] is False


def test_get_prediction_pipeline_missing_dir(tmp_path: Path):
    """Verifies that get_prediction_pipeline raises FileNotFoundError if model files are missing."""
    empty_models_dir = tmp_path / "missing_models"
    empty_models_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_info:
        get_prediction_pipeline(models_dir=empty_models_dir)
    assert "Trained model file not found" in str(exc_info.value)


def test_predict_using_real_saved_artifacts_if_present():
    """
    Integration test verifying that the actual saved model artifacts in models/
    can be loaded and produce valid predictions.
    """
    models_dir = PROJECT_ROOT / "models"
    model_file = models_dir / "best_model.joblib"
    vec_file = models_dir / "tfidf_vectorizer.joblib"

    if model_file.is_file() and vec_file.is_file():
        model, vectorizer = get_prediction_pipeline(models_dir=models_dir)

        safe_res = predict_comment(
            "This is a constructive, well-written and very helpful comment.",
            model=model,
            vectorizer=vectorizer,
        )
        assert safe_res["prediction_text"] == "Safe"

        abusive_res = predict_comment(
            "You are a brainless idiot and I hate you.",
            model=model,
            vectorizer=vectorizer,
        )
        assert abusive_res["prediction_text"] == "Abusive"


def test_predict_explainability_and_threshold_override(synthetic_pipeline):
    """Verifies v2 TF-IDF * coefficient explanation and custom threshold override."""
    model, vectorizer = synthetic_pipeline
    res = predict_comment(
        "You are a disgusting idiot and I will hurt you!",
        model=model,
        vectorizer=vectorizer,
        threshold=0.5,
        explain=True,
        top_k=5,
    )
    assert len(res["top_features"]) > 0
    assert "contribution" in res["top_features"][0]
    assert "direction" in res["top_features"][0]

    # Very high threshold should flip verdict to Safe even with identical probabilities
    res_lenient = predict_comment(
        "You are a disgusting idiot and I will hurt you!",
        model=model,
        vectorizer=vectorizer,
        threshold=0.9999,
    )
    assert res_lenient["is_abusive"] is False
    assert res_lenient["threshold_used"] == 0.9999


def test_predict_hinglish_toggle_integration(synthetic_pipeline):
    """Verifies enable_hinglish toggle flags Roman Hinglish when ON and skips rule when OFF."""
    model, vectorizer = synthetic_pipeline
    text = "Tu bahut bada kameena aur harami hai"

    res_on = predict_comment(text, model=model, vectorizer=vectorizer, enable_hinglish=True)
    assert res_on["is_abusive"] is True
    assert res_on["hinglish_match"]["hinglish_flagged"] is True
    assert "insult" in res_on["active_categories"]

    res_off = predict_comment(text, model=model, vectorizer=vectorizer, enable_hinglish=False)
    assert res_off["hinglish_match"]["hinglish_flagged"] is False


def test_v2_bundle_end_to_end_if_present():
    """Integration test verifying models/v2/ bundle (FeatureUnion + Calibrated LinearSVC + OvR)."""
    from src.predict import load_v2_bundle

    v2_dir = PROJECT_ROOT / "models" / "v2"
    if (v2_dir / "best_model.joblib").is_file() and (v2_dir / "tfidf_vectorizer.joblib").is_file():
        bundle = load_v2_bundle(v2_dir)
        bal_t = bundle["thresholds"]["thresholds"]["balanced"]["threshold"]
        res = predict_comment(
            "You are a 5tup1d 1d10t and I will k!ll you!",
            model=bundle["model"],
            vectorizer=bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=bundle.get("multilabel_model"),
            enable_hinglish=True,
            explain=True,
        )
        assert res["is_abusive"] is True
        assert len(res["top_features"]) > 0
        assert "threat" in res["category_probabilities"]
