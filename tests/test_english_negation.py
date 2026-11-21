"""
test_english_negation.py - Unit tests for Clause-Scoped English Negation Layer (Fix 2.2).

Verifies that:
1. Explicitly negated insults and speech denials are classified as Safe.
2. Direct insults, double negations, and intensifiers remain Abusive.
3. Violent threats containing negation words remain Abusive.
4. Multi-clause sentences with independent unnegated abuse remain Abusive.
5. The original ML score is strictly preserved in `ml_abusive_probability`.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_v2_bundle, predict_comment


@pytest.fixture(scope="module")
def v2_bundle():
    """Loads the canonical v2 production bundle."""
    return load_v2_bundle()


def test_mandatory_regression_cases(v2_bundle):
    """Verifies all 10 mandatory regression test cases specified in Fix 2.2."""
    model = v2_bundle["model"]
    vectorizer = v2_bundle["vectorizer"]
    threshold = 0.3905

    cases = [
        ("I am calling you an idiot.", "Abusive"),
        ("I am not calling you an idiot.", "Safe"),
        ("You are an idiot.", "Abusive"),
        ("You are not an idiot.", "Safe"),
        ("I cannot believe you are such an idiot.", "Abusive"),
        ("Don't call him an idiot.", "Safe"),
        ("I am not joking, I will hurt you.", "Abusive"),
        ("I won't hesitate to kill you.", "Abusive"),
        ("I will not stop until you are destroyed, you idiot.", "Abusive"),
        ("You are not not an idiot.", "Abusive"),
    ]

    for text, expected in cases:
        res = predict_comment(
            raw_text=text,
            model=model,
            vectorizer=vectorizer,
            threshold=threshold,
            enable_hinglish=True,
        )
        assert res["prediction_text"] == expected, (
            f"Failed on '{text}': expected {expected}, got {res['prediction_text']} "
            f"(ML prob: {res['ml_abusive_probability']:.4f}, final: {res['abusive_probability']:.4f})"
        )


def test_epistemic_and_speech_denials(v2_bundle):
    """Verifies that speech denials and epistemic disagreements are classified as Safe."""
    model = v2_bundle["model"]
    vectorizer = v2_bundle["vectorizer"]
    threshold = 0.3905

    safe_denials = [
        "I never said you were stupid.",
        "I don't think you are stupid.",
        "It is not true that you are an idiot.",
        "No one called you an idiot, calm down.",
        "You are not stupid, you are just inexperienced.",
        "Not an idiot, just learning.",
        "Why would you say that? You are definitely not an idiot.",
        "There is no reason to call anyone an idiot here.",
    ]

    for text in safe_denials:
        res = predict_comment(
            raw_text=text,
            model=model,
            vectorizer=vectorizer,
            threshold=threshold,
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, (
            f"Expected Safe for denial: '{text}', got Abusive (Prob: {res['abusive_probability']:.4f})"
        )


def test_quoted_and_reported_speech(v2_bundle):
    """Verifies that quotes, reported inquiries, and admonitions are classified as Safe."""
    model = v2_bundle["model"]
    vectorizer = v2_bundle["vectorizer"]
    threshold = 0.3905

    quoted_cases = [
        'He told me "you are an idiot" which was very mean.',
        'Why did you tweet "you are an idiot"?',
        'Stop shouting "idiot" at people.',
    ]

    for text in quoted_cases:
        res = predict_comment(
            raw_text=text,
            model=model,
            vectorizer=vectorizer,
            threshold=threshold,
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, (
            f"Expected Safe for quoted mention: '{text}', got Abusive (Prob: {res['abusive_probability']:.4f})"
        )


def test_multi_clause_with_unnegated_abuse(v2_bundle):
    """Verifies that if one clause is negated but another clause has direct abuse, it remains Abusive."""
    model = v2_bundle["model"]
    vectorizer = v2_bundle["vectorizer"]
    threshold = 0.3905

    mixed_cases = [
        "I am not calling you an idiot, but you are an absolute moron.",
        "You are not stupid, but you are a disgusting loser.",
        "Don't call him an idiot, you ugly clown.",
    ]

    for text in mixed_cases:
        res = predict_comment(
            raw_text=text,
            model=model,
            vectorizer=vectorizer,
            threshold=threshold,
            enable_hinglish=True,
        )
        assert res["is_abusive"] is True, (
            f"Expected Abusive for mixed comment: '{text}', got Safe (Prob: {res['abusive_probability']:.4f})"
        )


def test_score_preservation_and_telemetry(v2_bundle):
    """Verifies that the original ML probability is preserved and telemetry is returned."""
    model = v2_bundle["model"]
    vectorizer = v2_bundle["vectorizer"]

    text = "I am not calling you an idiot."
    res = predict_comment(
        raw_text=text,
        model=model,
        vectorizer=vectorizer,
        threshold=0.3905,
    )

    # Raw ML score should reflect the unadjusted model output (~99.9%)
    assert res["ml_abusive_probability"] > 0.95
    # Adjusted final score should be in the safe zone (< 0.3905)
    assert res["abusive_probability"] < 0.3905
    assert res["is_abusive"] is False

    # English negation telemetry must be present
    assert "english_negation_info" in res
    neg_info = res["english_negation_info"]
    assert neg_info["adjustment_applied"] is True
    assert "idiot" in neg_info["negated_terms"]
    assert neg_info["original_ml_score"] == res["ml_abusive_probability"]
    assert neg_info["adjusted_score"] == res["abusive_probability"]

