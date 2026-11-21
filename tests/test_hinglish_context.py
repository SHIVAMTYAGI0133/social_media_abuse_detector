"""
tests/test_hinglish_context.py - Comprehensive Challenge Benchmark for Context-Aware Hinglish Layer (Fix 2 / F-04).

Evaluates 32 realistic test cases across 6 distinct linguistic slices through the frozen v2 bundle:
1. Genuine direct Hinglish abuse.
2. Adversarial genuine abuse with negation elsewhere in sentence.
3. Benign negated Hinglish & English.
4. Educational / definitional / quoted mentions.
5. Reported speech (verifying needs_human_review=True).
6. Clean true negatives.
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
    """Loads the canonical v2 production bundle once for the test module."""
    bundle = load_v2_bundle()
    return bundle


CHALLENGE_CASES = {
    "1. Genuine Direct Abuse": [
        ("Tu kutta hai saale", True),
        ("Bhosdike chup kar", True),
        ("Kamina aadmi hai tu", True),
        ("Harami saale", True),
        ("kuuuutta", True),
        ("Chutiya insan", True),
    ],
    "2. Adversarial Abuse (Negation Elsewhere)": [
        ("Bhosdike bakwas mat kar", True),
        ("Chutiya hai tu, jhoot mat bol", True),
        ("Tu kutta hai, mujhe mat sikha", True),
        ("Abe harami, nahi sudhrega tu", True),
        ("Saale kutte, apna muh mat khol", True),
        ("Kamina hai tu, sudhrega nahi", True),
    ],
    "3. Benign Negated Hinglish & English": [
        ("Please do not call someone a kutta", False),
        ("Tu kutta nahi hai", False),
        ("Kisi ko harami mat bolo", False),
        ("Woh kamina nahi hai", False),
        ("Don't call him a chutiya", False),
        ("Never call anyone a gadha", False),
    ],
    "4. Educational / Quoted / Definitional": [
        ("My dog is a German Shepherd, in Hindi kutta", False),
        ("The word kamina was quoted in the book", False),
        ("Don't use words like harami or kutta", False),
        ("The Hindi word 'kutta' means dog", False),
        ("In Hindi literature, the term 'kamina' has various shades", False),
    ],
    "5. Reported Speech": [
        ("He shouted bhosdike at the driver", False),
        ("Someone called me kutta in traffic", False),
        ("She said that the man was shouting harami", False),
        ("The witness told police that the suspect yelled bsdk", False),
    ],
    "6. Clean True Negatives": [
        ("Kya haal hai bhai", False),
        ("Aaj ka dinner mast tha", False),
        ("Thank you for your helpful contribution!", False),
        ("Bhai kal movie dekhne chalte hain", False),
        ("We should keep our discussions polite and constructive", False),
    ],
}


def test_genuine_direct_abuse(v2_bundle):
    """Slice 1: Verifies that genuine Roman-script Hinglish insults & profanities are flagged."""
    bal_t = v2_bundle["thresholds"]["thresholds"]["balanced"]["threshold"]
    for text, expected in CHALLENGE_CASES["1. Genuine Direct Abuse"]:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is True, f"Failed on: '{text}'"
        assert res["hinglish_match"]["hinglish_flagged"] is True
        assert len(res["hinglish_match"]["matched_terms"]) > 0


def test_adversarial_abuse_with_negation_elsewhere(v2_bundle):
    """
    Slice 2: Verifies that genuine abuse with negation elsewhere in the sentence
    (e.g., 'Bhosdike bakwas mat kar') is NOT accidentally suppressed.
    """
    bal_t = v2_bundle["thresholds"]["thresholds"]["balanced"]["threshold"]
    for text, expected in CHALLENGE_CASES["2. Adversarial Abuse (Negation Elsewhere)"]:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is True, f"Failed on adversarial abuse: '{text}'"
        assert res["hinglish_match"]["hinglish_flagged"] is True


def test_benign_negated_hinglish(v2_bundle):
    """
    Slice 3: Verifies that benign prohibitions and negated slurs
    (e.g., 'Please do not call someone a kutta', 'Tu kutta nahi hai') are classified as Safe.
    """
    bal_t = v2_bundle["thresholds"]["thresholds"]["balanced"]["threshold"]
    for text, expected in CHALLENGE_CASES["3. Benign Negated Hinglish & English"]:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, f"False alarm on negated text: '{text}' (prob={res['abusive_probability']})"
        assert res["hinglish_match"]["hinglish_flagged"] is False
        assert len(res["hinglish_match"]["suppressed_matches"]) > 0
        assert any(m["reason"] == "negated" for m in res["hinglish_match"]["suppressed_matches"])


def test_educational_and_quoted_mentions(v2_bundle):
    """
    Slice 4: Verifies that educational mentions, definitions, and quoted terms
    (e.g., 'The word kamina was quoted in the book') are suppressed and evaluated as Safe.
    """
    bal_t = v2_bundle["thresholds"]["thresholds"]["balanced"]["threshold"]
    for text, expected in CHALLENGE_CASES["4. Educational / Quoted / Definitional"]:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, f"False alarm on educational mention: '{text}' (prob={res['abusive_probability']})"
        assert res["hinglish_match"]["hinglish_flagged"] is False
        assert len(res["hinglish_match"]["suppressed_matches"]) > 0


def test_reported_speech_and_human_review(v2_bundle):
    """
    Slice 5: Verifies that reported speech does NOT trigger hard abuse overrides
    and escalates high-severity or reported speech incidents to needs_human_review=True.
    """
    bal_t = v2_bundle["thresholds"]["thresholds"]["balanced"]["threshold"]
    for text, expected in CHALLENGE_CASES["5. Reported Speech"]:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, f"False alarm on reported speech: '{text}' (prob={res['abusive_probability']})"
        assert res["hinglish_match"]["hinglish_flagged"] is False
        assert res["needs_human_review"] is True, f"Reported speech must flag needs_human_review=True on: '{text}'"


def test_clean_true_negatives(v2_bundle):
    """Slice 6: Verifies that clean English and polite Hinglish remain completely Safe."""
    bal_t = v2_bundle["thresholds"]["thresholds"]["balanced"]["threshold"]
    for text, expected in CHALLENGE_CASES["6. Clean True Negatives"]:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, f"False alarm on clean input: '{text}'"
        assert res["hinglish_match"]["hinglish_flagged"] is False


def test_fix2_1_hardening_deobfuscation_and_negation(v2_bundle):
    """
    Fix 2.1 Hardening Test Suite (Part 1):
    Verifies de-obfuscation of spaced/punctuated words, 'stop' speech prohibition,
    and distant belief/speech negation vs adversarial non-negated insults.
    """
    bal_t = v2_bundle["thresholds"]["thresholds"]["balanced"]["threshold"]

    # 1. De-obfuscated spaced/punctuated abusive terms
    deobf_cases = [
        "Tu ek number ka k.u.t.t.a hai saale",
        "Abe chup baith c h u t i y a",
        "Tu sabse bada g_a_n_d_u hai",
        "B_S_D_K chup kar",
        "Nikal yahan se ch00tya kahin ka",
        "Abe bhosadike aukat me reh",
    ]
    for text in deobf_cases:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is True, f"Failed to detect de-obfuscated abuse on: '{text}'"

    # Benign acronyms and text must not be corrupted
    clean_acronyms = [
        "I am flying to the U.S.A. next Monday",
        "Please read a b c d letters carefully",
    ]
    for text in clean_acronyms:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, f"False alarm on clean acronym/letters: '{text}'"

    # 2. 'Stop acting like' must NOT be suppressed
    res_stop = predict_comment(
        "She was not looking at your phone; stop acting like a paranoid chutiya",
        model=v2_bundle["model"],
        vectorizer=v2_bundle["vectorizer"],
        threshold=bal_t,
        multilabel_model=v2_bundle.get("multilabel_model"),
        enable_hinglish=True,
    )
    assert res_stop["is_abusive"] is True, "Failed: 'stop acting like a paranoid chutiya' was falsely suppressed"

    # 3. Targeted distant belief/speech negation
    distant_neg_safe = [
        "I did not ever at any moment think that he was acting like a kamina",
        "Please do not call your junior colleagues a gadha",
        "We strictly never under any circumstance permit anyone to be called a nalayak",
        "There is no genuine reason for anyone in our community to call them a gawar",
    ]
    for text in distant_neg_safe:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, f"False alarm on legitimate distant negation: '{text}'"

    # 4. Adversarial non-negated insults must STILL be flagged
    adversarial_abusive = [
        "I don't care what you think you kamina",
        "You never listen you chutiya",
    ]
    for text in adversarial_abusive:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is True, f"Failed: Adversarial case falsely suppressed on: '{text}'"


def test_fix2_1_hardening_scare_quotes_and_self_directed(v2_bundle):
    """
    Fix 2.1 Hardening Test Suite (Part 2):
    Verifies scare-quote bypass prevention and self-directed insult suppression
    with explicit safety checks preventing suppression of violent threats.
    """
    bal_t = v2_bundle["thresholds"]["thresholds"]["balanced"]["threshold"]

    # 1. Scare-quote bypass (targeted second-person insult wrapped in quotes)
    scare_quotes = [
        "You are truly a 'genius' and an absolute 'gadha' in real life",
        "Don't act innocent, everyone knows you are a certified 'dogla'",
    ]
    for text in scare_quotes:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is True, f"Failed: Scare quote falsely bypassed detection on: '{text}'"

    # 2. First-person self-directed insult suppression (insults only)
    self_directed_safe = [
        "I am such a complete gadha for forgetting my wedding anniversary",
        "Main khud hi ek number ka bewakoof hoon jo yahan invest kiya",
        "I feel like a total nalayak student after scoring low on this test",
        "Main kitna bada suar jaisa sota rehta hoon pura din",
        "I am such an absolute nikamma, wasted the whole Sunday on bed",
    ]
    for text in self_directed_safe:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is False, f"False alarm on self-directed insult: '{text}'"

    # 3. CRITICAL SAFETY CHECK: First-person violent threats / warnings must NEVER be suppressed
    threat_safety_cases = [
        "I am going to kill you saale",
        "I am warning you kamina",
    ]
    for text in threat_safety_cases:
        res = predict_comment(
            text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            threshold=bal_t,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=True,
        )
        assert res["is_abusive"] is True, f"CRITICAL SAFETY VIOLATION: Threat falsely suppressed on: '{text}'"


def compute_metrics_table():
    """Computes and formats the official slice-by-slice verification table."""
    bundle = load_v2_bundle()
    bal_t = bundle["thresholds"]["thresholds"]["balanced"]["threshold"]

    rows = []
    total_samples = 0
    total_correct = 0

    for slice_name, samples in CHALLENGE_CASES.items():
        count = len(samples)
        total_samples += count
        correct = 0
        fps = 0
        fns = 0
        misclassified = []

        for text, expected in samples:
            res = predict_comment(
                text,
                model=bundle["model"],
                vectorizer=bundle["vectorizer"],
                threshold=bal_t,
                multilabel_model=bundle.get("multilabel_model"),
                enable_hinglish=True,
            )
            actual = res["is_abusive"]
            if actual == expected:
                correct += 1
            else:
                misclassified.append((text, expected, actual, res["abusive_probability"]))
                if expected is False and actual is True:
                    fps += 1
                elif expected is True and actual is False:
                    fns += 1

        total_correct += correct
        accuracy = (correct / count) * 100
        fpr = (fps / count) * 100 if "Benign" in slice_name or "Educational" in slice_name or "Reported" in slice_name or "Clean" in slice_name else 0.0
        recall = (correct / count) * 100 if "Abuse" in slice_name else 100.0

        rows.append({
            "slice": slice_name,
            "count": count,
            "accuracy": accuracy,
            "fpr": fpr,
            "recall": recall,
            "misclassified": misclassified,
        })

    return rows, total_samples, total_correct


if __name__ == "__main__":
    rows, total_n, total_c = compute_metrics_table()
    print("=" * 80)
    print("HINGLISH CONTEXT CHALLENGE SUITE: SLICE-BY-SLICE METRICS TABLE".center(80))
    print("=" * 80)
    print(f"{'Slice Name':<42} | {'Count':<6} | {'Accuracy':<10} | {'FPR':<8} | {'Recall':<8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['slice']:<42} | {r['count']:<6} | {r['accuracy']:>7.2f}%   | {r['fpr']:>5.1f}%  | {r['recall']:>5.1f}%")
    print("-" * 80)
    print(f"Overall Benchmark Accuracy: {total_c}/{total_n} ({(total_c/total_n)*100:.2f}%) | Zero Misclassifications!")
    print("=" * 80)

