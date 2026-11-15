"""
hinglish_detector.py - Explicit, Toggleable Roman-Script Hinglish Rule Layer (Phase 3).

Provides deterministic, boundary-safe lexicon matching for Roman-script Hinglish
insults, obscenities, threats, and hate terms that an English-only Jigsaw TF-IDF
model cannot learn from Wikipedia talk-page training data.
"""

import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_PUNCT_SPACE_TABLE = str.maketrans({ch: " " for ch in string.punctuation})
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_ANY_RE = re.compile(r"([a-z])\1+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Category weights used to derive an interpretable rule severity score (0.0 - 1.0)
_CATEGORY_SCORES = {
    "threat": 0.98,
    "obscene": 0.94,
    "hate": 0.92,
    "insult": 0.88,
}


def get_default_hinglish_lexicon_path() -> Path:
    """Returns the default path to data/hinglish_lexicon.txt."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "hinglish_lexicon.txt"


def _normalize_for_hinglish(text: str, collapse_repeats: bool = False) -> str:
    cleaned = _URL_RE.sub(" ", text.lower())
    # Basic inline leet normalization for Hinglish masking (e.g. chut1ya, bsd*k, k@mina)
    cleaned = cleaned.replace("@", "a").replace("$", "s").replace("1", "i").replace("0", "o").replace("3", "e").replace("4", "a").replace("5", "s")
    cleaned = cleaned.translate(_PUNCT_SPACE_TABLE)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if collapse_repeats:
        cleaned = _REPEAT_ANY_RE.sub(r"\1", cleaned)
    return cleaned


@lru_cache(maxsize=4)
def _load_compiled_lexicon(lexicon_path_str: str) -> Tuple[Tuple[str, str, re.Pattern, re.Pattern], ...]:
    """Loads and compiles word-boundary regexes for all terms in the lexicon file."""
    path = Path(lexicon_path_str)
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] Hinglish lexicon file not found at: {path}")

    entries: List[Tuple[str, str, re.Pattern, re.Pattern]] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            category, term = line.split("|", 1)
            category = category.strip().lower()
            term = term.strip().lower()
            if not category or not term or (category, term) in seen:
                continue
            seen.add((category, term))

            exact_pat = re.compile(rf"\b{re.escape(term)}\b")
            collapsed_term = _REPEAT_ANY_RE.sub(r"\1", term)
            collapsed_pat = re.compile(rf"\b{re.escape(collapsed_term)}\b")
            entries.append((category, term, exact_pat, collapsed_pat))

    return tuple(entries)


def load_hinglish_lexicon(
    lexicon_path: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, str]]:
    """
    Loads the categorized Hinglish lexicon as a list of (category, term) tuples.
    """
    path = Path(lexicon_path).resolve() if lexicon_path else get_default_hinglish_lexicon_path()
    compiled = _load_compiled_lexicon(str(path))
    return [(cat, term) for cat, term, _, _ in compiled]


def detect_hinglish_abuse(
    text: Any,
    lexicon_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Checks a raw comment against the Roman-script Hinglish lexicon using word-boundary
    matching on both standard and repeat-collapsed text.

    Returns:
        dict with keys:
            - hinglish_flagged (bool)
            - matched_terms (list[str])
            - matched_categories (list[str])
            - rule_score (float in [0.0, 1.0])
    """
    if text is None or not str(text).strip():
        return {
            "hinglish_flagged": False,
            "matched_terms": [],
            "matched_categories": [],
            "rule_score": 0.0,
        }

    path = Path(lexicon_path).resolve() if lexicon_path else get_default_hinglish_lexicon_path()
    compiled_entries = _load_compiled_lexicon(str(path))

    raw_str = str(text)
    norm_exact = _normalize_for_hinglish(raw_str, collapse_repeats=False)
    norm_collapsed = _normalize_for_hinglish(raw_str, collapse_repeats=True)

    matched_terms: List[str] = []
    matched_categories: List[str] = []
    max_score = 0.0

    for category, term, exact_pat, collapsed_pat in compiled_entries:
        if exact_pat.search(norm_exact) or collapsed_pat.search(norm_collapsed):
            if term not in matched_terms:
                matched_terms.append(term)
            if category not in matched_categories:
                matched_categories.append(category)
            max_score = max(max_score, _CATEGORY_SCORES.get(category, 0.90))

    return {
        "hinglish_flagged": len(matched_terms) > 0,
        "matched_terms": matched_terms,
        "matched_categories": matched_categories,
        "rule_score": float(max_score),
    }

