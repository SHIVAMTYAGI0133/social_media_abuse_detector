"""
hinglish_detector.py - Context-Aware Roman-Script Hinglish Rule Layer (Fix 2.1 Hardening).

Provides deterministic, clause-scoped lexicon matching for Roman-script Hinglish
insults, obscenities, threats, and hate terms with intelligent context classification:
1. De-obfuscation of spaced/punctuated letter runs (e.g. k.u.t.t.a -> kutta, c h u t i y a -> chutiya)
   and inline leetspeak normalization before clause splitting.
2. Clause-scoped segmentation (punctuation & contrastive conjunctions).
3. Prohibition & pre-negation (speech-prohibition 'stop calling/saying', removing bare 'stop').
4. Targeted distant belief/speech negation frames (did not think/say/call/intend).
5. Tight Hindi/Hinglish post-negation detection (<term> nahi / mat bol).
6. Scare-quote bypass protection while preserving educational/reporting quotations.
7. First-person self-reference suppression scoped strictly to the 'insult' category.
8. Reported speech recognition with human-review escalation flag.
"""

import re
import string
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

_PUNCT_SPACE_TABLE = str.maketrans({ch: " " for ch in string.punctuation})
_WHITESPACE_RE = re.compile(r"\s+")
_REPEAT_ANY_RE = re.compile(r"([a-z])\1+", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Category weights used to derive an interpretable rule severity score (0.0 - 1.0)
_CATEGORY_SCORES = {
    "threat": 0.98,
    "obscene": 0.94,
    "hate": 0.92,
    "insult": 0.88,
}

# Clause boundaries: split on strong punctuation, comma pauses, or contrastive conjunctions
_CLAUSE_SPLIT_RE = re.compile(
    r"[\.\!\?\,\;\:\n\r—\-\|]+|\b(?:but|lekin|par|magar|kintu|however|although|yet|balki)\b",
    re.IGNORECASE,
)

# Quotation markers
_QUOTE_CHARS = set("'\"“”‘’`«»")

# Educational / Meta-linguistic framing patterns
_EDU_FRAMING_PHRASES = [
    r"\b(?:word|words|words\s+like|term|terms|slur|slurs)\b",
    r"\b(?:in\s+hindi|hindi\s+for|hindi\s+word|in\s+hindi\s+is)\b",
    r"\b(?:means|meaning|translates\s+to|translation|translate)\b",
    r"\b(?:quoted|quote|quoted\s+in|dictionary|dictionaries|definition|etymology|etymological)\b",
]
_EDU_FRAMING_COMBINED_RE = re.compile("|".join(_EDU_FRAMING_PHRASES), re.IGNORECASE)

# Reported speech patterns
_REPORTED_SPEECH_RES = [
    re.compile(
        r"\b(?:he|she|they|someone|somebody|the\s+\w+|usne|woh|wo)\s+(?:called|calling|shouted|shouting|yelled|yelling|screamed|screaming|said|saying|told|bola|kaha|bol\s+raha\s+tha|chillaya)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?:called|calling)\s+(?:me|him|her|them|us|someone)\b", re.IGNORECASE),
    re.compile(r"\b(?:shouted|shouting|yelled|yelling|screamed|screaming)\s+(?:at\b)?", re.IGNORECASE),
    re.compile(r"\b(?:said|saying|told|telling)\s+that\b", re.IGNORECASE),
]

# Pre-negation & speech-prohibition patterns (scoped before the term in the same clause)
# Bare 'stop', 'avoid', 'refrain from' and bare 'never' are REMOVED to prevent false suppression
# of "stop acting like a paranoid chutiya" and "You never listen you chutiya"
_PRE_NEGATION_RES = [
    re.compile(r"\b(?:do\s+not|don\'?t|dont|never|stop|avoid|refrain\s+from|should\s+not|shouldn\'?t)\s+(?:call|calling|say|saying|use|using|refer\s+to|label|labeling)\b", re.IGNORECASE),
    re.compile(r"\b(?:do\s+not|don\'?t|dont|should\s+not|shouldn\'?t)\b", re.IGNORECASE),
    re.compile(r"\b(?:not|no)\s+(?:a|an|any)?\b", re.IGNORECASE),
    re.compile(r"\bnever\s+(?:a|an|any)\b", re.IGNORECASE),
    re.compile(r"\b(?:is|are|am|was|were)\s+(?:not|never)\b", re.IGNORECASE),
    re.compile(r"\b(?:isn\'?t|aren\'?t|wasn\'?t|weren\'?t)\b", re.IGNORECASE),
    re.compile(r"\b(?:mat\s+bol|mat\s+bolo|mat\s+kaho|mat\s+kehna|mat\s+bano|kabhi\s+mat|na\s+bol|na\s+bolo|na\s+kaho)\b", re.IGNORECASE),
    re.compile(r"\b(?:kisi\s+ko|kisi\s+bhi)\b", re.IGNORECASE),
]

# Targeted distant belief/speech negation frames (clause-wide search before term)
# Does NOT match generic negation like "I don't care what you think you kamina" or "You never listen you chutiya"
_DISTANT_BELIEF_SPEECH_NEGATION_RES = [
    re.compile(
        r"\b(?:did\s+not|didn\'?t|do\s+not|don\'?t|does\s+not|doesn\'?t|was\s+not|wasn\'?t|were\s+not|weren\'?t|cannot|can\'?t|could\s+not|couldn\'?t|has\s+not|hasn\'?t|have\s+not|haven\'?t)\s+(?!care\b)(?:[\w\']+\s+){0,6}\b(?:think|say|saying|call|calling|claim|claiming|mean|meaning|portray|portraying|accuse|accused|accusing|intend|intending|imagine|imagining|permit|permitting|support|supporting|label|labeling|refer)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bnever\s+(?:[\w\']+\s+){0,6}\b(?:called|calling|call|said|say|thought|think|used|use|permit|permitting|label|labeling|accused|accuse|justification)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:would\s+not|wouldn\'?t|should\s+not|shouldn\'?t)\s+(?:[\w\']+\s+){0,6}\b(?:call|calling|say|saying|use|using|label|labeling)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|nobody|no\s+one)\s+(?:[\w\']+\s+){0,6}\b(?:reason|justification|accused|permit|support|called|calling|call)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:do\s+not|don\'?t|never|should\s+not|shouldn\'?t|please\s+do\s+not)\s+call\s+(?:[\w\']+\s+){0,6}\b",
        re.IGNORECASE,
    ),
]

# Post-negation patterns (tightly bound within 1-2 tokens following the term)
_POST_NEGATION_RES = [
    re.compile(r"^\s*(?:bhi\s+|toh\s+|to\s+)?(?:nahi|nahin|nhi|na)\b(?:\s+(?:hai|tha|hoga|kuch|tha\s+ye))?", re.IGNORECASE),
    re.compile(r"^\s*(?:bhi\s+)?mat\s+(?:bol|bolo|kaho|kehna|bano|samjho|samajh|karna|kar)\b", re.IGNORECASE),
]

# De-obfuscation helpers & patterns
_LEET_MAP = str.maketrans({"@": "a", "$": "s", "1": "i", "0": "o", "3": "e", "4": "a", "5": "s"})
_PUNCT_RUN_RE = re.compile(r"\b([a-zA-Z0-9@$](?:[\.\-_][a-zA-Z0-9@$]){2,})\b")
_SPACED_RUN_RE = re.compile(r"\b([a-zA-Z0-9@$](?:\s+[a-zA-Z0-9@$]){3,})\b")
_LEET_TOKEN_RE = re.compile(r"\b([a-zA-Z0-9@$]{3,})\b")

# Direct second-person / third-person targeting for scare-quote bypass check
_DIRECT_TARGETING_PAT = re.compile(
    r"\b(?:you\s+are|you\'?re|you\s+r|ur\s+(?:a|an)?|u\s+r|tu\s+.*?hai|tum\s+.*?ho|you\s+(?:absolute|certified|total|complete|real|little)|he\s+is\s+(?:an?\s+|just\s+an?\s+)?|she\s+is\s+(?:an?\s+|just\s+an?\s+)?)\b",
    re.IGNORECASE,
)
_REPORTING_OR_EDU_PAT = re.compile(
    r"\b(?:word|words|words\s+like|term|terms|slur|slurs|means|meaning|translates|quoted|quote|said|shouted|yelled|screamed|called\s+me|called\s+him|called\s+her|called\s+us|named|referred\s+to\s+as|yell|write|writes|wrote|asked|record|recorded|archive|memoir|memoirs)\b",
    re.IGNORECASE,
)

# First-person self-reference framing (applied ONLY to 'insult' category)
_SELF_REF_PAT = re.compile(
    r"\b(?:main\s+khud(?:\s+hi)?|main\s+hi|meri\s+hi\s+galti|i\s+am\s+(?:such\s+an?|an?|being\s+an?|feeling\s+like\s+an?|a\s+total|an?\s+absolute|acting\s+like\s+an?)|i\'?m\s+(?:such\s+an?|an?|being\s+an?|feeling\s+like\s+an?|a\s+total|an?\s+absolute|acting\s+like\s+an?)|i\s+feel\s+like(?:\s+an?)?|myself(?:\s+an?)?|main\s+kitna\s+bada)(?:\s+[\w\']+){0,4}\s*$",
    re.IGNORECASE,
)


def get_default_hinglish_lexicon_path() -> Path:
    """Returns the default path to data/hinglish_lexicon.txt."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "hinglish_lexicon.txt"


def _normalize_for_hinglish(text: str, collapse_repeats: bool = False) -> str:
    cleaned = _URL_RE.sub(" ", text.lower())
    cleaned = cleaned.translate(_LEET_MAP)
    cleaned = cleaned.translate(_PUNCT_SPACE_TABLE)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    if collapse_repeats:
        cleaned = _REPEAT_ANY_RE.sub(r"\1", cleaned)
    return cleaned


@lru_cache(maxsize=4)
def _get_lexicon_terms_set(lexicon_path_str: str) -> Set[str]:
    """Returns a set of lowercase lexicon terms and significant sub-words for de-obfuscation validation."""
    path = Path(lexicon_path_str)
    terms: Set[str] = set()
    if not path.is_file():
        return terms
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "|" not in line:
                continue
            _, term = line.split("|", 1)
            term_clean = term.strip().lower()
            if term_clean:
                terms.add(term_clean)
                for w in term_clean.split():
                    if len(w) >= 3 and w not in {"hai", "tha", "mein", "yeh", "kar", "mat", "par", "toh", "aur"}:
                        terms.add(w)
    return terms


def _deobfuscate_text(text: str, lexicon_terms: Set[str]) -> str:
    """
    De-obfuscates single-letter punctuated runs (k.u.t.t.a -> kutta, g_a_n_d_u -> gandu),
    spaced runs (c h u t i y a -> chutiya), and inline leetspeak tokens (ch00tya -> chootya)
    ONLY when the collapsed token matches a lexicon term.
    """
    # 1. Punctuated runs: k.u.t.t.a, g_a_n_d_u, k-u-t-t-e
    def _rep_punct(m):
        raw = m.group(1)
        collapsed = re.sub(r"[\.\-_]+", "", raw).lower().translate(_LEET_MAP)
        collapsed_rep = _REPEAT_ANY_RE.sub(r"\1", collapsed)
        if collapsed in lexicon_terms or collapsed_rep in lexicon_terms:
            return collapsed
        return m.group(0)

    text = _PUNCT_RUN_RE.sub(_rep_punct, text)

    # 2. Spaced runs: c h u t i y a, s a a l e, a u k a t
    def _rep_spaced(m):
        raw = m.group(1)
        collapsed = re.sub(r"\s+", "", raw).lower().translate(_LEET_MAP)
        collapsed_rep = _REPEAT_ANY_RE.sub(r"\1", collapsed)
        if collapsed in lexicon_terms or collapsed_rep in lexicon_terms:
            return collapsed
        return m.group(0)

    text = _SPACED_RUN_RE.sub(_rep_spaced, text)

    # 3. Inline leetspeak words: ch00tya, k@mina
    def _rep_leet(m):
        raw = m.group(1)
        if any(ch in "@$01345" for ch in raw):
            collapsed = raw.lower().translate(_LEET_MAP)
            collapsed_rep = _REPEAT_ANY_RE.sub(r"\1", collapsed)
            if collapsed in lexicon_terms or collapsed_rep in lexicon_terms:
                return collapsed
        return m.group(0)

    text = _LEET_TOKEN_RE.sub(_rep_leet, text)

    return text


@lru_cache(maxsize=4)
def _load_compiled_lexicon(lexicon_path_str: str) -> Tuple[Tuple[str, str, re.Pattern], ...]:
    """
    Loads and compiles boundary-safe regexes with character stretching support
    for all categorized terms in the Hinglish lexicon.
    """
    path = Path(lexicon_path_str)
    if not path.is_file():
        raise FileNotFoundError(f"[ERROR] Hinglish lexicon file not found at: {path}")

    entries: List[Tuple[str, str, re.Pattern]] = []
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

            # Build regex supporting character stretching while enforcing word boundaries
            words = term.split()
            word_patterns = []
            for w in words:
                parts = [f"{re.escape(ch)}+" if ch.isalpha() else re.escape(ch) for ch in w]
                word_patterns.append("".join(parts))
            full_pattern = rf"\b{r'\s+'.join(word_patterns)}\b"
            compiled_pat = re.compile(full_pattern, re.IGNORECASE)
            entries.append((category, term, compiled_pat))

    return tuple(entries)


def load_hinglish_lexicon(
    lexicon_path: Optional[Union[str, Path]] = None,
) -> List[Tuple[str, str]]:
    """
    Loads the categorized Hinglish lexicon as a list of (category, term) tuples.
    """
    path = Path(lexicon_path).resolve() if lexicon_path else get_default_hinglish_lexicon_path()
    compiled = _load_compiled_lexicon(str(path))
    return [(cat, term) for cat, term, _ in compiled]


def _get_clauses(text: str) -> List[Tuple[int, int, str]]:
    """Splits text into clauses preserving character offsets."""
    clauses: List[Tuple[int, int, str]] = []
    last_end = 0
    for match in _CLAUSE_SPLIT_RE.finditer(text):
        start = match.start()
        if start > last_end:
            segment = text[last_end:start].strip()
            if segment:
                clauses.append((last_end, start, segment))
        last_end = match.end()
    if last_end < len(text):
        segment = text[last_end:].strip()
        if segment:
            clauses.append((last_end, len(text), segment))
    return clauses if clauses else [(0, len(text), text.strip())]


def _classify_match_context(
    raw_text: str,
    term: str,
    category: str,
    clause_text: str,
    rel_start: int,
    rel_end: int,
) -> Optional[str]:
    """
    Determines if a matched term is contextually suppressed.
    Returns:
        reason string ('quoted', 'educational_mention', 'reported_speech', 'negated', 'self_directed')
        or None if the term represents an active, unsuppressed abusive match.
    """
    before_in_clause = clause_text[:rel_start]
    after_in_clause = clause_text[rel_end:]
    before_stripped = before_in_clause.rstrip()
    after_stripped = after_in_clause.lstrip()

    # 1. Quotation Check (with scare-quote bypass protection)
    term_is_quoted = False
    if before_stripped and after_stripped:
        if before_stripped[-1] in _QUOTE_CHARS and after_stripped[0] in _QUOTE_CHARS:
            term_is_quoted = True

    raw_match = re.search(rf"['\"“”‘’`]\s*{re.escape(term)}\s*['\"“”‘’`]", raw_text, re.IGNORECASE)
    if raw_match:
        term_is_quoted = True

    if term_is_quoted:
        has_targeting = bool(_DIRECT_TARGETING_PAT.search(clause_text) or _DIRECT_TARGETING_PAT.search(raw_text))
        has_edu_reporting = bool(_REPORTING_OR_EDU_PAT.search(clause_text) or _REPORTING_OR_EDU_PAT.search(raw_text))
        # Suppress as quotation ONLY if it is not an active scare-quote attack
        if not (has_targeting and not has_edu_reporting):
            return "quoted"

    # 2. Educational / Meta-linguistic Framing
    edu_before_pat = re.compile(
        rf"\b(?:word|words|words\s+like|term|terms|slur|slurs|in\s+hindi|hindi\s+for|hindi\s+word|meaning\s+of)\s+(?:['\"“”‘’]?\w+['\"“”‘’]?\s+){{0,3}}{re.escape(term)}\b",
        re.IGNORECASE,
    )
    edu_after_pat = re.compile(
        rf"\b{re.escape(term)}\s+(?:in\s+hindi|means|meaning|translates\s+to|is\s+hindi\s+for|was\s+quoted|quoted|connects\s+to)\b",
        re.IGNORECASE,
    )
    if edu_before_pat.search(clause_text) or edu_after_pat.search(clause_text) or edu_before_pat.search(raw_text) or edu_after_pat.search(raw_text):
        return "educational_mention"

    if re.search(rf"\bwords?\s+like\s+.*?\b{re.escape(term)}\b", raw_text, re.IGNORECASE):
        return "educational_mention"

    if re.search(rf"\bin\s+hindi\s+.*?\b{re.escape(term)}\b", raw_text, re.IGNORECASE):
        return "educational_mention"

    # 3. Reported Speech
    for rep_pat in _REPORTED_SPEECH_RES:
        if rep_pat.search(before_in_clause) or rep_pat.search(raw_text[:raw_text.lower().find(term)] if term in raw_text.lower() else ""):
            return "reported_speech"

    # 4. Pre-Negation & Speech Prohibition in the same clause (5-token window)
    tokens_before = before_stripped.split()
    recent_tokens_before = " ".join(tokens_before[-5:]) if len(tokens_before) > 5 else before_stripped
    for pre_neg_pat in _PRE_NEGATION_RES:
        if pre_neg_pat.search(recent_tokens_before):
            return "negated"

    # 4b. Targeted distant belief/speech negation frames (clause-wide search before term)
    for dist_pat in _DISTANT_BELIEF_SPEECH_NEGATION_RES:
        if dist_pat.search(before_in_clause):
            return "negated"

    # 5. Post-Negation in the same clause (tightly bound within 1-2 tokens)
    tokens_after = after_stripped.split()
    first_two_after = " ".join(tokens_after[:3]) if tokens_after else ""
    for post_neg_pat in _POST_NEGATION_RES:
        if post_neg_pat.search(first_two_after):
            return "negated"

    # 6. First-Person Self-Reference Framing (Insult category ONLY)
    if category == "insult":
        if _SELF_REF_PAT.search(before_in_clause.rstrip()):
            return "self_directed"

    return None


def detect_hinglish_abuse(
    text: Any,
    lexicon_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Checks raw comment text against the Roman-script Hinglish lexicon using
    de-obfuscation and clause-scoped contextual analysis.

    Returns:
        dict with backward-compatible and upgraded keys:
            - hinglish_flagged (bool): True ONLY if at least one active match exists.
            - matched_terms (list[str]): Active abusive matches only.
            - active_matches (list[str]): Alias of matched_terms.
            - matched_categories (list[str]): Active abusive categories.
            - suppressed_matches (list[dict]): Contextual matches with term, category, reason.
            - needs_human_review (bool): True if suppressed match has high severity or reported speech.
            - rule_score (float in [0.0, 1.0]): Severity score from active matches only (0.0 if none).
    """
    if text is None or not str(text).strip():
        return {
            "hinglish_flagged": False,
            "matched_terms": [],
            "active_matches": [],
            "matched_categories": [],
            "suppressed_matches": [],
            "needs_human_review": False,
            "rule_score": 0.0,
        }

    path = Path(lexicon_path).resolve() if lexicon_path else get_default_hinglish_lexicon_path()
    path_str = str(path)
    compiled_entries = _load_compiled_lexicon(path_str)
    lex_terms = _get_lexicon_terms_set(path_str)

    raw_str = str(text)
    # De-obfuscate spaced/punctuated letter runs and leetspeak prior to clause splitting
    deobf_str = _deobfuscate_text(raw_str, lex_terms)
    clauses = _get_clauses(deobf_str)

    active_terms: List[str] = []
    active_categories: List[str] = []
    suppressed_matches: List[Dict[str, str]] = []
    max_score = 0.0

    seen_active = set()
    seen_suppressed = set()

    for category, term, pat in compiled_entries:
        for _, _, clause_str in clauses:
            for match in pat.finditer(clause_str):
                rel_start = match.start()
                rel_end = match.end()

                suppress_reason = _classify_match_context(
                    raw_text=deobf_str,
                    term=term,
                    category=category,
                    clause_text=clause_str,
                    rel_start=rel_start,
                    rel_end=rel_end,
                )

                if suppress_reason is not None:
                    if (term, suppress_reason) not in seen_suppressed:
                        seen_suppressed.add((term, suppress_reason))
                        suppressed_matches.append({
                            "term": term,
                            "category": category,
                            "reason": suppress_reason,
                        })
                else:
                    if term not in seen_active:
                        seen_active.add(term)
                        active_terms.append(term)
                        if category not in active_categories:
                            active_categories.append(category)
                        max_score = max(max_score, _CATEGORY_SCORES.get(category, 0.88))

    needs_review = any(
        m["category"] in ("threat", "obscene") or m["reason"] == "reported_speech"
        for m in suppressed_matches
    )

    is_flagged = len(active_terms) > 0

    return {
        "hinglish_flagged": is_flagged,
        "matched_terms": active_terms,
        "active_matches": active_terms,
        "matched_categories": active_categories,
        "suppressed_matches": suppressed_matches,
        "needs_human_review": bool(needs_review),
        "rule_score": float(max_score) if is_flagged else 0.0,
    }
