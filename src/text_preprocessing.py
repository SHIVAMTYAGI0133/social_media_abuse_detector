"""
text_preprocessing.py - Text normalization, de-obfuscation, and cleaning for NLP.

This module normalizes raw social media comments to prepare them for TF-IDF
feature extraction and machine learning modeling.

Design Choices & Preprocessing Rules (v2 upgraded):
1. Lowercasing: Ensures words like "Rude", "RUDE", and "rude" map to the same token.
2. URL Removal: Web links (http/https/www) do not carry linguistic abusive meaning and add noise.
3. Obfuscation & Leetspeak De-obfuscation (v2, toggleable via `normalize_obfuscation`):
   - Normalizes symbol-masked toxic/threat roots (`f*ck`, `sh!t`, `b!tch`, `k!ll`,
     `a$$`, `@ss`, `id*ot`, `d!e`, spaced single-letter evasion like `f u c k` / `f.u.c.k`).
   - Converts leetspeak digits/symbols inside mixed word tokens (`1`->`i`/`l`, `0`->`o`,
     `3`->`e`, `4`->`a`, `5`->`s`, `7`->`t`, `@`->`a`, `$`->`s`) while preserving pure
     numbers (`12345`, `3.14`) and normal alphanumeric identifiers (`v2`, `mp3`).
   - Compresses stretched repeated letters (>2 repeats down to 1-2 letters, e.g.,
     `stuuuupid` -> `stupid`, `kiiiillll` -> `kill`) while keeping natural doubles (`good`, `kill`).
4. Careful Punctuation Handling: Remaining punctuation is replaced with a space instead of
   being silently deleted, preventing adjacent words from merging (`hello.world` -> `hello world`).
5. Negation Preservation: Crucial words like "not", "never", "no", "none" are preserved.
   We explicitly DO NOT perform aggressive stopword removal.
6. Robust Input Contract: Missing (None), non-string, or whitespace-only inputs safely
   return "".
"""

import re
import string
from typing import Any, List, Tuple, Union
import pandas as pd

# Regex pattern to identify URLs (http, https, and www prefixes)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Translation table to map punctuation characters to spaces
PUNCTUATION_SPACED_TABLE = str.maketrans(
    {char: " " for char in string.punctuation}
)

# Regex to collapse multiple whitespace characters (spaces, tabs, newlines) into a single space
WHITESPACE_PATTERN = re.compile(r"\s+")

# Curated symbol-masked toxic/threat root patterns (run before punctuation spacing)
_MASKED_ROOT_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Spaced or dotted single-letter evasion (e.g. f.u.c.k, s.h.i.t, k.i.l.l)
    (re.compile(r"(?<![a-z])f[\s._*-]+u[\s._*-]+c[\s._*-]+k(ing|er|ed|s)?(?![a-z])"), r"fuck\1"),
    (re.compile(r"(?<![a-z])s[\s._*-]+h[\s._*-]+[i1!][\s._*-]+t(ty|head|s)?(?![a-z])"), r"shit\1"),
    (re.compile(r"(?<![a-z])b[\s._*-]+[i1!][\s._*-]+t[\s._*-]+c[\s._*-]+h(es|ing|y)?(?![a-z])"), r"bitch\1"),
    (re.compile(r"(?<![a-z])k[\s._*-]+[i1!][\s._*-]+l[\s._*-]+l(s|ed|ing|er)?(?![a-z])"), r"kill\1"),
    # Symbol-masked roots (*, !, @, $, #, ., _, -) inside words
    (re.compile(r"(?<![a-z])f+(?:[\*!@#$%._-]+c?k|u[\*!@#$%._-]+c?k|[\*!@#$%._-]+u[\*!@#$%._-]+c[\*!@#$%._-]+k)(ing|ed|er|ers|s)?(?![a-z])"), r"fuck\1"),
    (re.compile(r"(?<![a-z])(?:ph|f)[\*!@#$%._-]+c+k+(ing|ed|er|s)?(?![a-z])"), r"fuck\1"),
    (re.compile(r"(?<![a-z])[s$5]+h[\*!@#$%1]+t+(s|ty|ting|head)?(?![a-z])"), r"shit\1"),
    (re.compile(r"(?<![a-z])[s$5]+[\*!@#$%._-]+i+t+(s|ty)?(?![a-z])"), r"shit\1"),
    (re.compile(r"(?<![a-z])b[\*!@#$%1]+t+c+h+(es|ing|y|s)?(?![a-z])"), r"bitch\1"),
    (re.compile(r"(?<![a-z])b+i+[\*!@#$%._-]+c+h+(es|ing|y|s)?(?![a-z])"), r"bitch\1"),
    (re.compile(r"(?<![a-z])k[\*!@#$%1]+l+l+(s|ed|ing|er|ers)?(?![a-z])"), r"kill\1"),
    (re.compile(r"(?<![a-z])k+i+[\*!@#$%1]+l+(s|ed|ing|er|ers)?(?![a-z])"), r"kill\1"),
    (re.compile(r"(?<![a-z])d[\*!1][\*!e3](s|d)?(?![a-z])"), r"die\1"),
    (re.compile(r"(?<![a-z])[a@4]+[\*@$5]+[\*@$5]+(holes?|hats?|wipe)?(?![a-z])"), r"ass\1"),
    (re.compile(r"(?<![a-z])[a@4]+s+s+h+[o0*]+l+[e3*]+(s)?(?![a-z])"), r"asshole\1"),
    (re.compile(r"(?<![a-z])c[\*!u@]+n+t+(s)?(?![a-z])"), r"cunt\1"),
    (re.compile(r"(?<![a-z])d[\*!1]+c+k+(s|head|heads)?(?![a-z])"), r"dick\1"),
    (re.compile(r"(?<![a-z])c+[o0*]+c+k+(s|sucker|suckers)?(?![a-z])"), r"cock\1"),
    (re.compile(r"(?<![a-z])s+l+[\*!u@]+t+(s|ty)?(?![a-z])"), r"slut\1"),
    (re.compile(r"(?<![a-z])w+h+[\*0o@]+r+[e3*]+(s)?(?![a-z])"), r"whore\1"),
    (re.compile(r"(?<![a-z])[i1!]+d+[\*!i1]+[\*0o@]+t+(s|ic)?(?![a-z])"), r"idiot\1"),
    (re.compile(r"(?<![a-z])[s$5]+t+[\*u]+p+[\*!i1]+d+(s|ity)?(?![a-z])"), r"stupid\1"),
    (re.compile(r"(?<![a-z])p+[a@4]+t+h+[e3*]+t+[\*!i1]+c+(?![a-z])"), r"pathetic"),
    (re.compile(r"(?<![a-z])l+[\*0o]+s+[\*3e]+r+(s)?(?![a-z])"), r"loser\1"),
    (re.compile(r"(?<![a-z])m+[\*0o]+r+[\*0o]+n+(s|ic)?(?![a-z])"), r"moron\1"),
    (re.compile(r"(?<![a-z])r+[\*3e]+t+[\*@4a]+r+d+(s|ed)?(?![a-z])"), r"retard\1"),
    (re.compile(r"(?<![a-z])f+[\*@4a]+g+(g+[o0*]+t+s?|s)?(?![a-z])"), r"fag\1"),
    (re.compile(r"(?<![a-z])n+[\*!i1]+g+g+([\*@4a]+|[\*3e]+r)(s)?(?![a-z])"), r"nigg\1\2"),
    (re.compile(r"(?<![a-z])m+[\*u]+r+d+[\*3e]+r+(s|ed|ing|er|ers)?(?![a-z])"), r"murder\1"),
    (re.compile(r"(?<![a-z])p+[\*!i1]+s+s+(ed|ing)?(?![a-z])"), r"piss\1"),
    (re.compile(r"(?<![a-z])h+[\*3e]+l+l+(?![a-z])"), r"hell"),
    (re.compile(r"(?<![a-z])c+r+[\*@4a]+p+(py|s)?(?![a-z])"), r"crap\1"),
    (re.compile(r"(?<![a-z])t+r+[\*@4a]+s+h+(y)?(?![a-z])"), r"trash\1"),
    (re.compile(r"(?<![a-z])[s$5]+[\*u]+c+k+(s|ed|ing|er|ers)?(?![a-z])"), r"suck\1"),
]

# Token-level inline symbol translation (@ -> a, $ -> s) when inside mixed word tokens
_INLINE_SYMBOL_TABLE = str.maketrans({"@": "a", "$": "s"})

# Leetspeak digit-to-letter mapping
_LEET_DIGIT_TO_I = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})
_LEET_DIGIT_TO_L = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})

# Words where '1' inside leetspeak stands for 'l' rather than 'i'
_L_VARIANT_WORDS = {
    "loser", "losers", "hell", "kill", "kills", "killed", "killing", "killer", "killers",
    "slut", "sluts", "asshole", "assholes", "bullshit", "ugly", "fool", "fools",
    "liar", "liars", "loser", "clown", "clowns",
}

# Known targets for tokens with trailing digits (e.g. a55 -> ass, d13 -> die, sh17 -> shit)
_TRAILING_LEET_TARGETS = {
    "ass", "asses", "asshole", "die", "dies", "died", "hoe", "hoes", "fag", "fags",
    "shit", "idiot", "hate", "are", "you", "kill", "suck", "hell", "crap", "dumb",
    "loser", "moron", "slut", "whore", "cunt", "bitch", "dick", "fuck", "trash",
    "threat", "nigger", "nigga", "retard", "pathetic", "disgusting",
    "go", "no", "so", "to", "do",
}

# Canonical forms for stretched words where natural double letters exist (e.g. kiiiillll -> kill)
_STRETCHED_DOUBLE_LETTER_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^k+i+l+l+(s|ed|ing|er|ers)?$"), r"kill\1"),
    (re.compile(r"^h+e+l+l+$"), "hell"),
    (re.compile(r"^a+s+s+(h+o+l+e+s?)?$"), r"ass\1"),
    (re.compile(r"^p+i+s+s+(ed|ing)?$"), r"piss\1"),
    (re.compile(r"^f+a+g+g+o+t+(s)?$"), r"faggot\1"),
    (re.compile(r"^n+i+g+g+(a+s?|e+r+s?)$"), r"nigg\1"),
    (re.compile(r"^b+u+l+l+s+h+i+t+$"), "bullshit"),
    (re.compile(r"^f+o+o+l+(s|ish)?$"), r"fool\1"),
    (re.compile(r"^a+g+r+e+e+(d|s|ing)?$"), r"agree\1"),
    (re.compile(r"^g+o+o+d+$"), "good"),
    (re.compile(r"^c+o+o+l+$"), "cool"),
    (re.compile(r"^f+r+e+e+$"), "free"),
    (re.compile(r"^n+o+o+b+(s)?$"), r"noob\1"),
]

_REPEAT_3PLUS_PATTERN = re.compile(r"([a-z])\1{2,}")
_MIXED_SYMBOL_TOKEN_PATTERN = re.compile(r"[a-z0-9@$]+")


def _decode_inline_symbols(text: str) -> str:
    """Replaces @->a and $->s only inside tokens that contain at least one ASCII letter."""
    def _replace_tok(match: re.Match) -> str:
        tok = match.group(0)
        if ("@" in tok or "$" in tok) and any("a" <= ch <= "z" for ch in tok):
            return tok.translate(_INLINE_SYMBOL_TABLE)
        return tok

    return _MIXED_SYMBOL_TOKEN_PATTERN.sub(_replace_tok, text)


def _decode_leet_and_compress_token(token: str) -> str:
    """
    Decodes leetspeak digits inside mixed alphanumeric tokens and compresses >2
    repeated consecutive letters down to 1-2 letters.
    Leaves pure numeric tokens ('12345', '3', '14') and benign suffix tokens ('v2', 'mp3') untouched.
    """
    if not any("a" <= ch <= "z" for ch in token):
        return token

    # Handle explicit mixed i/l pattern like k111 / ki11 / 10s3r
    if re.fullmatch(r"k[1i]{2,}(?:s|ed|ing|ers?)?", token):
        suffix = re.sub(r"^k[1i]+", "", token)
        token = "kill" + suffix
    elif any(d in token for d in "013457"):
        # Check if digit appears before/between letters or after an inline leet trigger
        has_leading_or_internal_digit = bool(re.search(r"[013457][a-z]", token))
        cand_i = token.translate(_LEET_DIGIT_TO_I)
        cand_l = token.translate(_LEET_DIGIT_TO_L)

        if cand_l in _L_VARIANT_WORDS:
            token = cand_l
        elif has_leading_or_internal_digit:
            token = cand_i
        elif cand_i in _TRAILING_LEET_TARGETS:
            token = cand_i
        elif cand_l in _TRAILING_LEET_TARGETS:
            token = cand_l

    # Compress >2 consecutive identical letters
    if _REPEAT_3PLUS_PATTERN.search(token):
        # First check stretched words with natural double letters (e.g., kiiiillll -> kill)
        for pattern, replacement in _STRETCHED_DOUBLE_LETTER_PATTERNS:
            if pattern.match(token):
                return pattern.sub(replacement, token)
        # Otherwise collapse 3+ repeats down to 1 canonical letter (e.g. stuuuupid -> stupid)
        token = _REPEAT_3PLUS_PATTERN.sub(r"\1", token)

    return token


def normalize_obfuscation_text(text: str) -> str:
    """
    Applies v2 de-obfuscation normalization to a lowercased, URL-stripped string:
    1. Resolves symbol-masked toxic/threat roots (`f*ck`, `sh!t`, `k!ll`, `a$$`, `f.u.c.k`).
    2. Converts inline `@`->`a` and `$`->`s` inside mixed word tokens.
    3. Replaces remaining punctuation with spaces (via standard spacing table).
    4. Decodes leetspeak digits inside words and compresses stretched letters (>2 repeats).
    """
    normalized = text
    for pattern, replacement in _MASKED_ROOT_PATTERNS:
        normalized = pattern.sub(replacement, normalized)

    normalized = _decode_inline_symbols(normalized)
    normalized = normalized.translate(PUNCTUATION_SPACED_TABLE)

    tokens = normalized.split()
    cleaned_tokens = [_decode_leet_and_compress_token(tok) for tok in tokens]
    return " ".join(cleaned_tokens)


def preprocess_text(text: Any, normalize_obfuscation: bool = True) -> str:
    """
    Normalizes a single text comment for NLP feature extraction.

    Transformation Steps:
    1. Input Validation: If text is None or not a string, safely convert or return "".
    2. Lowercasing: Converts all characters to lowercase.
    3. URL Removal: Replaces web links with spaces.
    4. Obfuscation Normalization (when normalize_obfuscation=True):
       Resolves masked toxic roots, leetspeak inside words, and >2 stretched repeats.
    5. Punctuation Handling: Replaces punctuation characters with spaces to preserve
       word boundaries and prevent words from accidentally merging.
    6. Negation Preservation: Negations ("not", "never", "no") and alphanumeric words
       are kept intact without aggressive stopword removal.
    7. Whitespace Normalization: Collapses multiple spaces/newlines into a single space
       and strips leading/trailing spaces.

    Parameters:
        text (Any): Input comment (typically str, but handles None, float, int gracefully).
        normalize_obfuscation (bool): Enables v2 leetspeak/masking/stretch normalization
            (default: True). Set False for exact v1 legacy baseline behavior.

    Returns:
        str: Normalized, lowercase text string. Returns "" if input is empty or invalid.
    """
    # 1. Validate and handle empty, None, or non-string input
    if text is None:
        return ""

    try:
        if pd.isna(text):
            return ""
    except (ValueError, TypeError):
        pass

    if not isinstance(text, str):
        text = str(text)

    if not text.strip():
        return ""

    # 2. Convert to lowercase
    normalized = text.lower()

    # 3. Remove URLs
    normalized = URL_PATTERN.sub(" ", normalized)

    # 4 & 5. Apply de-obfuscation + punctuation spacing
    if normalize_obfuscation:
        normalized = normalize_obfuscation_text(normalized)
    else:
        normalized = normalized.translate(PUNCTUATION_SPACED_TABLE)

    # 6. Normalize whitespace
    normalized = WHITESPACE_PATTERN.sub(" ", normalized).strip()

    return normalized


def preprocess_text_series(
    texts: Union[pd.Series, List[Any]],
    show_progress: bool = False,
    normalize_obfuscation: bool = True,
) -> pd.Series:
    """
    Applies text preprocessing to a collection or Pandas Series of comments.

    Parameters:
        texts (pd.Series or list): Collection of raw text comments.
        show_progress (bool): If True, prints status updates during processing.
        normalize_obfuscation (bool): Enables v2 leetspeak/masking/stretch normalization.

    Returns:
        pd.Series: Pandas Series of preprocessed, normalized text strings.

    Raises:
        ValueError: If texts collection is empty.
    """
    if len(texts) == 0:
        raise ValueError("[ERROR] Cannot preprocess an empty collection of texts.")

    if not isinstance(texts, pd.Series):
        series = pd.Series(texts, dtype=object)
    else:
        series = texts.copy()

    if show_progress:
        print(f"[INFO] Preprocessing {len(series):,} text comments...")

    cleaned_series = series.apply(
        lambda t: preprocess_text(t, normalize_obfuscation=normalize_obfuscation)
    )

    if show_progress:
        print("[INFO] Text preprocessing complete.")

    return cleaned_series
