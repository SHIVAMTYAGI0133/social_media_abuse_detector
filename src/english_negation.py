"""
english_negation.py - Clause-Scoped English Negation & Scoping Layer for AbuseGuard AI.

Implements Solution A: Clause-Scoped English Negation & Scoping Layer (Fix 2.2).
Eliminates false positives where explicit negation, denials, prohibitions,
or reported inquiries govern insult terms (e.g., "I am not calling you an idiot"),
while strictly preserving:
- Unambiguous violent threats (e.g., "I am not joking, I will hurt you", "I won't hesitate to kill you")
- Double negations (e.g., "You are not not an idiot")
- Intensified insults (e.g., "I cannot believe you are such an idiot")
- Mixed multi-clause sentences with independent unnegated abuse
"""

import re
from typing import Any, Dict, List, Optional, Set

from .text_preprocessing import preprocess_text

# Target insult and profanity vocabulary commonly subject to false-positive negation
INSULT_TERMS: Set[str] = {
    "idiot", "idiots", "idiotic",
    "stupid", "stupidity",
    "moron", "morons", "moronic",
    "imbecile", "imbeciles",
    "fool", "fools", "foolish",
    "loser", "losers",
    "asshole", "assholes", "ashole",
    "bitch", "bitches",
    "bastard", "bastards",
    "cunt", "cunts",
    "dick", "dicks",
    "jerk", "jerks",
    "douchebag", "douchebags",
    "clown", "clowns",
    "trash", "garbage", "scum",
    "retard", "retards", "retarded",
    "dumb", "ugly", "pig", "pigs"
}

# Violent threats that MUST NEVER be suppressed by negation
_THREAT_PATTERNS = [
    re.compile(r"\b(?:will|gonna|going to|wanna|want to)\s+(?:kill|murder|hunt|shoot|stab|slit|punch|beat|destroy|choke|strangle|break)\s+(?:you|your|u|him|her|them)\b", re.I),
    re.compile(r"\b(?:break|smash)\s+(?:your|u|ur)\s+(?:neck|face|head|bones|legs|teeth)\b", re.I),
    re.compile(r"\b(?:won'?t|wont|will not)\s+hesitate\s+to\s+(?:kill|hurt|shoot|destroy|attack)\b", re.I),
    re.compile(r"\b(?:will not|won'?t)\s+stop\s+until\s+(?:you|u)\s+(?:are|re)?\s*(?:destroyed|dead|killed|hurt|gone)\b", re.I),
    re.compile(r"\b(?:i will|i'll)\s+(?:hurt|kill|murder|harm|attack|hunt|destroy|smash|ruin)\s+(?:you|u)\b", re.I),
    re.compile(r"\b(?:kill\s+yourself|kys|hang\s+yourself)\b", re.I),
    re.compile(r"\b(?:i am not joking|not playing around)[,\s]+.*(?:will\s+hurt|will\s+kill|break)\b", re.I),
]

# Affirmative intensifiers containing negation words that are NOT denials
_INTENSIFIER_PATTERNS = [
    re.compile(r"\b(?:cannot|can'?t)\s+believe\s+(?:that\s+)?(?:you|u|he|she|they)?\s*(?:are|re|is)?\s*(?:such\s+an?|an?|that)?\s*(?:" + "|".join(INSULT_TERMS) + r")\b", re.I),
    re.compile(r"\bnot\s+just\s+(?:a|an)?\s*(?:" + "|".join(INSULT_TERMS) + r")\b", re.I),
    re.compile(r"\bnot\s+only\s+(?:a|an)?\s*(?:" + "|".join(INSULT_TERMS) + r")\b", re.I),
    re.compile(r"\b(?:not|ain'?t)\s+(?:gonna|going to)\s+lie\s*[,:]?\s*(?:you|u)\s*(?:are|re)?\s*(?:a|an)?\s*(?:" + "|".join(INSULT_TERMS) + r")\b", re.I),
    re.compile(r"\bnot\s+to\s+mention\s*(?:you|u)\s*(?:are|re)?\s*(?:a|an)?\s*(?:" + "|".join(INSULT_TERMS) + r")\b", re.I),
]

# Double negation patterns (result in affirmative insult)
_DOUBLE_NEGATION_PATTERNS = [
    re.compile(r"\b(?:not|never)\s+(?:not|never)\b", re.I),
    re.compile(r"\b(?:cannot|can'?t)\s+say\s+(?:you|u|he|she)?\s*(?:are|re|is)?\s*not\b", re.I),
    re.compile(r"\bnot\s+that\s+(?:you|u|he|she)?\s*(?:are|re|is)?\s*not\b", re.I),
]


def _build_negation_patterns(term: str) -> List[re.Pattern]:
    """Builds clause-scoped regexes where negation governs a specific insult term."""
    escaped_term = re.escape(term)
    return [
        # Prohibitions & Admonitions
        re.compile(rf"\b(?:don'?t|do\s+not|never|stop|quit)\s+(?:call|calling|label|labeling|say|saying|shout|shouting|use|using)\s+(?:(?:him|her|them|us|me|anyone|anybody|people|users|someone)\s+)?(?:a|an|that)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\bno\s+reason\s+to\s+call\s+(?:anyone|him|her|them|anybody)?\s*(?:a|an)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\bstop\s+(?:shouting|saying|calling)\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\b(?:calling|labeling)\s+(?:you|him|her|them|anyone)?\s*(?:a|an)?\s*{escaped_term}\s+is\s+(?:wrong|unacceptable|inappropriate|not\s+(?:okay|ok|right|cool)|bad)\b", re.I),

        # Speech & Act Denial
        re.compile(rf"\b(?:i|we)\s+(?:am|are|was|were)\s+not\s+(?:calling|labeling|saying|insulting)\s+(?:you|him|her|them|anyone)?\s*(?:a|an|such an?)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\b(?:i|we)\s+never\s+(?:called|said|labeled|thought)\s+(?:you|him|her|them)?\s*(?:were|was|are|is)?\s*(?:a|an|such an?)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\b(?:i|we)\s+(?:do|did)\s+not\s+(?:call|say|label)\s+(?:you|him|her|them)?\s*(?:a|an)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\b(?:nobody|no\s+one)\s+(?:called|is\s+calling|said|thinks)\s+(?:you|anyone)?\s*(?:a|an)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\b(?:do|did)\s+not\s+(?:believe|think)\s+(?:that\s+)?(?:anyone|anybody|someone|people)?\s*(?:should|would)?\s*(?:call|label)\s+(?:you|him|her|them)?\s*(?:a|an)?\s*{escaped_term}\b", re.I),
        re.compile(rf"\bnot\s+going\s+to\s+.*(?:let\s+you\s+be\s+called|allow\s+anyone\s+to\s+call\s+you)\s+(?:a|an)?\s*{escaped_term}\b", re.I),

        # Direct Copular & Predicate Negation
        re.compile(rf"\b(?:you|u|he|she|they|i|we)\s+(?:are|re|is|am|was|were)\s+(?:not|never|definitely\s+not|certainly\s+not|obviously\s+not|hardly)\s+(?:a|an|such an?|that much of an?)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\b(?:you|u|he|she|they)\s*(?:aren'?t|isn'?t|wasn'?t|weren'?t)\s+(?:a|an|such an?)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\b(?:i|we)\s+(?:don'?t|do\s+not)\s+think\s+(?:that\s+)?(?:you|u|he|she|they)?\s*(?:are|re|is|were|was)?\s*(?:a|an)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\b(?:it\s+is|it'?s)\s+not\s+(?:true|right|fair|accurate)\s+(?:that\s+)?(?:you|u|he|she|they)?\s*(?:are|re|is)?\s*(?:a|an)?\s*[\"']?\s*{escaped_term}\b", re.I),
        re.compile(rf"\bnot\s+(?:an?|such an?)\s*{escaped_term}\b", re.I),
        re.compile(rf"\bdefinitely\s+not\s+(?:a|an)?\s*{escaped_term}\b", re.I),

        # Hinglish post-negation synergy on English term
        re.compile(rf"\b{escaped_term}\s+(?:nahi|nahin|nhi|na|mat\s+bol|mat\s+bolo)\b", re.I),

        # Quoted / Reported Mention (Meta-Discussion)
        re.compile(rf"\b(?:he|she|they|someone)\s+(?:told|said\s+to)\s+me\s+[\"'](?:you\s+are\s+an?\s+)?{escaped_term}[\"']", re.I),
        re.compile(rf"\bwhy\s+did\s+you\s+(?:tweet|write|say|post)\s+[\"'](?:you\s+are\s+an?\s+)?{escaped_term}[\"']", re.I),
    ]


def evaluate_english_negation_scoping(text: str) -> Dict[str, Any]:
    """
    Evaluates whether insult terms in the input text fall under valid negation scope.

    Returns:
        dict: Scoping metadata and neutralization instructions.
    """
    if not text or not text.strip():
        return {
            "should_neutralize": False,
            "has_threat_clause": False,
            "is_double_negation": False,
            "negated_terms": [],
            "reason": "Empty text"
        }

    # 1. Threat & Violence Gate
    for tp in _THREAT_PATTERNS:
        if tp.search(text):
            return {
                "should_neutralize": False,
                "has_threat_clause": True,
                "is_double_negation": False,
                "negated_terms": [],
                "reason": "Violent threat pattern detected; suppression prohibited"
            }

    # 2. Intensifier Gate
    for ip in _INTENSIFIER_PATTERNS:
        if ip.search(text):
            return {
                "should_neutralize": False,
                "has_threat_clause": False,
                "is_double_negation": False,
                "negated_terms": [],
                "reason": "Affirmative intensifier with negation detected"
            }

    # 3. Double Negation Gate
    for dnp in _DOUBLE_NEGATION_PATTERNS:
        if dnp.search(text):
            return {
                "should_neutralize": False,
                "has_threat_clause": False,
                "is_double_negation": True,
                "negated_terms": [],
                "reason": "Double negation detected (affirmative insult)"
            }

    # 4. Check for insult terms present
    words = set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))
    present_terms = words.intersection(INSULT_TERMS)
    if not present_terms:
        return {
            "should_neutralize": False,
            "has_threat_clause": False,
            "is_double_negation": False,
            "negated_terms": [],
            "reason": "No targeted insult terms present"
        }

    # 5. Split text into clauses for boundary checking
    clause_delimiters = r"[.!?;—–]|(?:\s+but\s+|\s+however\s+|\s+although\s+|\s+whereas\s+)"
    raw_clauses = re.split(clause_delimiters, text, flags=re.I)
    clauses = [c.strip() for c in raw_clauses if c.strip()]

    neutralized_text = text
    all_terms_negated = True
    negated_terms = []
    negated_clauses = []

    for term in present_terms:
        term_patterns = _build_negation_patterns(term)
        term_negated = False

        for clause in clauses:
            for pat in term_patterns:
                if pat.search(clause):
                    term_negated = True
                    negated_terms.append(term)
                    negated_clauses.append(clause)
                    break
            if term_negated:
                break

        if not term_negated:
            all_terms_negated = False
            break

    if all_terms_negated and negated_terms:
        # Neutralize negated insult occurrences using 'that' to maintain syntax
        for term in set(negated_terms):
            neutralized_text = re.sub(rf"\b{re.escape(term)}\b", "that", neutralized_text, flags=re.I)
        return {
            "should_neutralize": True,
            "has_threat_clause": False,
            "is_double_negation": False,
            "neutralized_text": neutralized_text,
            "negated_terms": list(set(negated_terms)),
            "negated_clauses": negated_clauses,
            "reason": f"Terms successfully negated: {list(set(negated_terms))}"
        }

    return {
        "should_neutralize": False,
        "has_threat_clause": False,
        "is_double_negation": False,
        "negated_terms": [],
        "reason": "Unnegated insult terms remain in text"
    }


def apply_english_negation_scoping(
    text: str,
    ml_abusive_prob: float,
    model: Any,
    vectorizer: Any,
    threshold: float = 0.3905,
) -> Dict[str, Any]:
    """
    Applies clause-scoped English negation analysis to adjust the abusive probability
    if and only if all insult tokens are legitimately negated and no threats exist.

    Preserves the original ML score for debugging.
    """
    # If model already deems text Safe, no false positive exists
    if ml_abusive_prob < threshold:
        return {
            "negation_detected": False,
            "negated_terms": [],
            "negated_clauses": [],
            "has_threat_clause": False,
            "is_double_negation": False,
            "adjustment_applied": False,
            "original_ml_score": float(ml_abusive_prob),
            "adjusted_score": float(ml_abusive_prob),
            "reason": "Initial ML prediction is below abusive threshold (Safe)"
        }

    scoping = evaluate_english_negation_scoping(text)

    if scoping.get("should_neutralize", False):
        neutralized = scoping["neutralized_text"]
        cleaned_neut = preprocess_text(neutralized)
        X_neut = vectorizer.transform([cleaned_neut])

        if hasattr(model, "predict_proba"):
            adjusted_score = float(model.predict_proba(X_neut)[0, 1])
        else:
            pred_raw = int(model.predict(X_neut)[0])
            adjusted_score = 1.0 if pred_raw == 1 else 0.0

        return {
            "negation_detected": True,
            "negated_terms": scoping.get("negated_terms", []),
            "negated_clauses": scoping.get("negated_clauses", []),
            "has_threat_clause": False,
            "is_double_negation": False,
            "adjustment_applied": True,
            "original_ml_score": float(ml_abusive_prob),
            "adjusted_score": float(adjusted_score),
            "reason": scoping.get("reason", "Negation scope successfully neutralized insult")
        }

    return {
        "negation_detected": len(scoping.get("negated_terms", [])) > 0,
        "negated_terms": scoping.get("negated_terms", []),
        "negated_clauses": scoping.get("negated_clauses", []),
        "has_threat_clause": scoping.get("has_threat_clause", False),
        "is_double_negation": scoping.get("is_double_negation", False),
        "adjustment_applied": False,
        "original_ml_score": float(ml_abusive_prob),
        "adjusted_score": float(ml_abusive_prob),
        "reason": scoping.get("reason", "No adjustment warranted")
    }

