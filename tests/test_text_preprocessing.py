"""
test_text_preprocessing.py - Unit tests for text normalization & cleaning.

NOTE ON TEST DATA:
=============================================================================
All data fixtures used in this test file are small SYNTHETIC TOY EXAMPLES created
strictly for automated unit testing purposes. They do NOT contain or represent
real comments from the Kaggle Jigsaw Toxic Comment Classification dataset.
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

from src.text_preprocessing import preprocess_text, preprocess_text_series


def test_uppercase_to_lowercase():
    """Verifies that all uppercase characters are converted to lowercase."""
    raw = "YOU ARE BEING VERY RUDE"
    expected = "you are being very rude"
    assert preprocess_text(raw) == expected


def test_urls_are_removed():
    """Verifies that HTTP, HTTPS, and WWW URLs are removed from text."""
    raw_http = "Visit http://spam.com/phishing for more info"
    raw_https = "Check this link: https://malicious-site.org/index.html right now"
    raw_www = "Go to www.example.com to see details"

    assert "http" not in preprocess_text(raw_http)
    assert preprocess_text(raw_http) == "visit for more info"

    assert "https" not in preprocess_text(raw_https)
    assert preprocess_text(raw_https) == "check this link right now"

    assert "www" not in preprocess_text(raw_www)
    assert preprocess_text(raw_www) == "go to to see details"


def test_repeated_whitespace_normalized():
    """Verifies that multiple spaces, newlines, and tabs are collapsed into single spaces."""
    raw = "This    has \t too \n\n many \r spaces!   "
    expected = "this has too many spaces"
    assert preprocess_text(raw) == expected


def test_punctuation_does_not_merge_words():
    """
    CRITICAL CHECK: Verifies that punctuation removal does not merge words together.
    For example: 'hello.world' -> 'hello world', NOT 'helloworld'.
    """
    raw_dot = "hello.world"
    raw_slash = "stop/go"
    raw_hyphen = "bad-behavior"
    raw_exclamation = "stop!now"

    assert preprocess_text(raw_dot) == "hello world"
    assert preprocess_text(raw_slash) == "stop go"
    assert preprocess_text(raw_hyphen) == "bad behavior"
    assert preprocess_text(raw_exclamation) == "stop now"


def test_negations_are_preserved():
    """
    CRITICAL CHECK: Verifies that crucial negation words like 'not', 'never', 'no'
    are preserved intact and not stripped as generic stopwords.
    """
    raw = "This is NOT good, NEVER do that, NO excuse!"
    cleaned = preprocess_text(raw)

    assert "not" in cleaned.split()
    assert "never" in cleaned.split()
    assert "no" in cleaned.split()
    assert cleaned == "this is not good never do that no excuse"


def test_empty_and_whitespace_inputs():
    """Verifies that empty strings and whitespace-only strings return empty strings."""
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""
    assert preprocess_text("\t\n  \r") == ""


def test_missing_and_non_string_inputs():
    """Verifies that None, NaN, and non-string inputs (numbers, objects) are handled safely."""
    assert preprocess_text(None) == ""
    assert preprocess_text(float("nan")) == ""
    assert preprocess_text(12345) == "12345"
    assert preprocess_text(3.14) == "3 14"  # Dot is replaced with space


def test_preprocess_text_series():
    """Verifies that preprocess_text_series cleans a series of synthetic comments."""
    raw_comments = pd.Series([
        "YOU are RUDE!",
        "Visit https://test.com now",
        "hello.world",
        "",
        None,
    ])

    cleaned = preprocess_text_series(raw_comments)
    assert isinstance(cleaned, pd.Series)
    assert len(cleaned) == 5
    assert cleaned.iloc[0] == "you are rude"
    assert cleaned.iloc[1] == "visit now"
    assert cleaned.iloc[2] == "hello world"
    assert cleaned.iloc[3] == ""
    assert cleaned.iloc[4] == ""


def test_preprocess_text_series_empty_error():
    """Verifies that attempting to preprocess an empty series raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        preprocess_text_series([])
    assert "Cannot preprocess an empty collection" in str(exc_info.value)


def test_leetspeak_normalization():
    """Verifies that digits/symbols used as leetspeak inside words are decoded while pure numbers stay intact."""
    assert preprocess_text("y0u ar3 an 1d10t") == "you are an idiot"
    assert preprocess_text("k1ll y0urs3lf") == "kill yourself"
    assert preprocess_text("5tup1d 10s3r") == "stupid loser"
    assert preprocess_text("@sshole $hit a55 d13") == "asshole shit ass die"
    # Pure numbers and benign numeric suffixes should remain intact
    assert preprocess_text("Version v2 costs 500 or 3.14") == "version v2 costs 500 or 3 14"


def test_masked_profanity_and_threats():
    """Verifies symbol-masked toxic and threat roots are recovered while normal punctuation still splits."""
    assert preprocess_text("f*ck y0u and your sh!t edits") == "fuck you and your shit edits"
    assert preprocess_text("you are a b!tch and a c*nt") == "you are a bitch and a cunt"
    assert preprocess_text("i will k!ll you") == "i will kill you"
    assert preprocess_text("f.u.c.k off") == "fuck off"
    assert preprocess_text("stop!now") == "stop now"


def test_repeated_letter_compression_preserves_doubles_and_negations():
    """Verifies >2 repeated letters collapse while normal doubles ('good', 'kill') and negations stay intact."""
    assert preprocess_text("youuu areee sooooo stuuuupid") == "you are so stupid"
    assert preprocess_text("kiiiillll youuu looooser") == "kill you loser"
    assert preprocess_text("This is NOT good and NEVER kill anyone") == "this is not good and never kill anyone"


def test_legacy_v1_mode_disables_obfuscation_normalization():
    """Verifies normalize_obfuscation=False reproduces legacy v1 preprocessing behavior."""
    raw = "y0u ar3 an 1d10t f*ck stuuuupid"
    legacy = preprocess_text(raw, normalize_obfuscation=False)
    assert legacy == "y0u ar3 an 1d10t f ck stuuuupid"
