"""
test_hinglish_detector.py - Unit tests for Phase 3 Roman-script Hinglish hybrid rule layer.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.hinglish_detector import detect_hinglish_abuse, load_hinglish_lexicon


def test_load_hinglish_lexicon_categories():
    """Verifies lexicon loads categorized terms across insult, obscene, threat, and hate."""
    entries = load_hinglish_lexicon()
    assert len(entries) >= 40
    categories = {cat for cat, _ in entries}
    assert {"obscene", "insult", "threat", "hate"}.issubset(categories)


def test_hinglish_insult_and_obscene_detection():
    """Verifies Hinglish profanity and insults are flagged with category tags."""
    res_insult = detect_hinglish_abuse("Tu bahut bada kameena aur harami hai")
    assert res_insult["hinglish_flagged"] is True
    assert "insult" in res_insult["matched_categories"]
    assert "kameena" in res_insult["matched_terms"]
    assert "harami" in res_insult["matched_terms"]
    assert res_insult["rule_score"] >= 0.85

    res_obs = detect_hinglish_abuse("Chup kar chutiya bsdk")
    assert res_obs["hinglish_flagged"] is True
    assert "obscene" in res_obs["matched_categories"]


def test_hinglish_threat_phrase_and_stretched_letters():
    """Verifies multi-word Hinglish threats and stretched spellings are detected."""
    res_threat = detect_hinglish_abuse("Tujhe jaan se maar dunga agar phir bola")
    assert res_threat["hinglish_flagged"] is True
    assert "threat" in res_threat["matched_categories"]
    assert res_threat["rule_score"] >= 0.95

    res_stretch = detect_hinglish_abuse("Tu ek number ka chutiyyaaaa aur haraaaami hai")
    assert res_stretch["hinglish_flagged"] is True
    assert "chutiya" in res_stretch["matched_terms"]
    assert "harami" in res_stretch["matched_terms"]


def test_hinglish_clean_english_and_hinglish_no_false_positives():
    """Verifies clean English and polite Hinglish do not trigger false substring matches."""
    assert detect_hinglish_abuse("")["hinglish_flagged"] is False
    assert detect_hinglish_abuse("Thank you for your helpful contribution!")["hinglish_flagged"] is False
    assert detect_hinglish_abuse("Aapka kaam bahut accha hai, shukriya bhai!")["hinglish_flagged"] is False
    # Substring safety check (e.g., 'march' must not match 'mar', 'randy' must not match 'randi')
    assert detect_hinglish_abuse("Randy is coming in March for the seminar")["hinglish_flagged"] is False

