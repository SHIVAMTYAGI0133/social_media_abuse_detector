"""
eval_adversarial.py - Evaluates 20 adversarial edge cases (leetspeak, symbol masking,
stretched letters, and safe negations) on v1 vs v2 and writes results/adversarial_examples.md.

Usage:
    python scripts/eval_adversarial.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import transform_tfidf
from src.text_preprocessing import preprocess_text


ADVERSARIAL_CASES: List[Dict[str, str]] = [
    # 1-5: Leetspeak evasion (expected: Abusive)
    {"id": "ADV-01", "category": "Leetspeak", "expected": "Abusive", "text": "y0u ar3 an 1d10t and 5tup1d 10s3r"},
    {"id": "ADV-02", "category": "Leetspeak", "expected": "Abusive", "text": "k1ll y0urs3lf n0w"},
    {"id": "ADV-03", "category": "Leetspeak", "expected": "Abusive", "text": "sh1t h3ad b1tch"},
    {"id": "ADV-04", "category": "Leetspeak", "expected": "Abusive", "text": "g0 d13 tr4sh"},
    {"id": "ADV-05", "category": "Leetspeak", "expected": "Abusive", "text": "y0u 5uck a55h0l3"},
    # 6-10: Symbol masking (expected: Abusive)
    {"id": "ADV-06", "category": "Symbol Masking", "expected": "Abusive", "text": "f*ck y0u and your sh!t edits"},
    {"id": "ADV-07", "category": "Symbol Masking", "expected": "Abusive", "text": "you are a b!tch and a c*nt"},
    {"id": "ADV-08", "category": "Symbol Masking", "expected": "Abusive", "text": "i will k!ll you if you touch this"},
    {"id": "ADV-09", "category": "Symbol Masking", "expected": "Abusive", "text": "shut up you @$$hole and d!e"},
    {"id": "ADV-10", "category": "Symbol Masking", "expected": "Abusive", "text": "total idi*t and pathet!c l*ser"},
    # 11-15: Stretched letters (expected: Abusive)
    {"id": "ADV-11", "category": "Stretched Letters", "expected": "Abusive", "text": "youuu areee sooooo stuuuupid"},
    {"id": "ADV-12", "category": "Stretched Letters", "expected": "Abusive", "text": "i haaaate youuu iddddiot"},
    {"id": "ADV-13", "category": "Stretched Letters", "expected": "Abusive", "text": "shuuuuttt uuuuppp looooser"},
    {"id": "ADV-14", "category": "Stretched Letters", "expected": "Abusive", "text": "kiiiillll youuuurself"},
    {"id": "ADV-15", "category": "Stretched Letters", "expected": "Abusive", "text": "disguuuusting trrrrash"},
    # 16-20: Safe negations & constructive edge cases (expected: Safe)
    {"id": "ADV-16", "category": "Safe Negation", "expected": "Safe", "text": "this edit is not bad at all thank you"},
    {"id": "ADV-17", "category": "Safe Negation", "expected": "Safe", "text": "i do not hate your work let us never argue"},
    {"id": "ADV-18", "category": "Safe Negation", "expected": "Safe", "text": "no insults please we should never attack users"},
    {"id": "ADV-19", "category": "Safe Negation", "expected": "Safe", "text": "there is no threat here just polite constructive feedback"},
    {"id": "ADV-20", "category": "Safe Negation", "expected": "Safe", "text": "never be rude and do not make personal attacks"},
]


def _predict_single(text: str, model, vectorizer, normalize_obfuscation: bool, threshold: float = 0.5) -> Dict:
    cleaned = preprocess_text(text, normalize_obfuscation=normalize_obfuscation)
    if not cleaned:
        return {"pred": "Safe", "prob": 0.0, "cleaned": ""}
    X = transform_tfidf(vectorizer, [cleaned])
    prob = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else float(model.predict(X)[0])
    pred = "Abusive" if prob >= threshold else "Safe"
    return {"pred": pred, "prob": prob, "cleaned": cleaned}


def main() -> None:
    v1_dir = PROJECT_ROOT / "models" / "v1"
    v2_dir = PROJECT_ROOT / "models" / "v2"
    out_md = PROJECT_ROOT / "results" / "adversarial_examples.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)

    v1_model = joblib.load(v1_dir / "best_model.joblib")
    v1_vec = joblib.load(v1_dir / "tfidf_vectorizer.joblib")

    has_v2 = (v2_dir / "best_model.joblib").is_file() and (v2_dir / "tfidf_vectorizer.joblib").is_file()
    if has_v2:
        v2_model = joblib.load(v2_dir / "best_model.joblib")
        v2_vec = joblib.load(v2_dir / "tfidf_vectorizer.joblib")
        thresh_file = v2_dir / "thresholds.json"
        v2_thresh = 0.5
        if thresh_file.is_file():
            with open(thresh_file, "r", encoding="utf-8") as f:
                t_data = json.load(f)
            v2_thresh = float(t_data.get("thresholds", {}).get("balanced", {}).get("threshold", 0.5))
        v2_label = f"v2 (Balanced @{v2_thresh:.3f})"
    else:
        # Preliminary Phase 1 check using v1 model + v2 de-obfuscation normalizer
        v2_model = v1_model
        v2_vec = v1_vec
        v2_thresh = 0.5
        v2_label = "v1+DeObfuscation (Pre-Phase 2)"

    rows = []
    v1_correct = 0
    v2_correct = 0
    cat_stats: Dict[str, Dict[str, int]] = {}

    for case in ADVERSARIAL_CASES:
        r1 = _predict_single(case["text"], v1_model, v1_vec, normalize_obfuscation=False, threshold=0.5)
        r2 = _predict_single(case["text"], v2_model, v2_vec, normalize_obfuscation=True, threshold=v2_thresh)

        ok1 = r1["pred"] == case["expected"]
        ok2 = r2["pred"] == case["expected"]
        v1_correct += int(ok1)
        v2_correct += int(ok2)

        cat = case["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"total": 0, "v1_ok": 0, "v2_ok": 0}
        cat_stats[cat]["total"] += 1
        cat_stats[cat]["v1_ok"] += int(ok1)
        cat_stats[cat]["v2_ok"] += int(ok2)

        rows.append(
            f"| {case['id']} | {cat} | `{case['text']}` | {case['expected']} | "
            f"{r1['pred']} ({r1['prob']*100:.1f}%) {'PASS' if ok1 else 'FAIL'} | "
            f"{r2['pred']} ({r2['prob']*100:.1f}%) {'PASS' if ok2 else 'FAIL'} | "
            f"`{r2['cleaned']}` |"
        )

    lines = [
        "# Adversarial Robustness Evaluation (20 Tricky Test Cases)",
        "",
        f"- **v1 Pipeline:** Legacy preprocessing (`normalize_obfuscation=False`) + frozen `models/v1/` classifier (`threshold=0.500`).",
        f"- **v2 Pipeline:** Upgraded de-obfuscation preprocessing (`normalize_obfuscation=True`) + `{v2_label}`.",
        "",
        "## Overall Accuracy on 20 Adversarial Cases",
        "",
        "| Pipeline | Correct / Total | Pass Rate |",
        "| :--- | :---: | :---: |",
        f"| **v1 Baseline** | {v1_correct} / {len(ADVERSARIAL_CASES)} | {v1_correct / len(ADVERSARIAL_CASES) * 100:.1f}% |",
        f"| **{v2_label}** | {v2_correct} / {len(ADVERSARIAL_CASES)} | {v2_correct / len(ADVERSARIAL_CASES) * 100:.1f}% |",
        "",
        "## Breakdown by Category",
        "",
        "| Category | Total | v1 Pass | v2 Pass |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for cat, st_vals in cat_stats.items():
        lines.append(
            f"| {cat} | {st_vals['total']} | {st_vals['v1_ok']}/{st_vals['total']} | {st_vals['v2_ok']}/{st_vals['total']} |"
        )

    lines.extend([
        "",
        "## Per-Example Results",
        "",
        "| ID | Category | Input Comment | Expected | v1 Result | v2 Result | v2 Normalized Text |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
        *rows,
        "",
    ])

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Saved adversarial evaluation -> {out_md}")
    print(f"Summary: v1={v1_correct}/20 ({v1_correct/20*100:.1f}%) | {v2_label}={v2_correct}/20 ({v2_correct/20*100:.1f}%)")


if __name__ == "__main__":
    main()

