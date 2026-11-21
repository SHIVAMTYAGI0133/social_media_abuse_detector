"""
app.py - AbuseGuard AI v2: Social Media Content Analyzer Dashboard

Upgraded Streamlit Web Application (v2):
- Interactive Sidebar: Engine selector (v2 Improved / v1 Original / Side-by-Side Compare),
  Sensitivity mode (Strict / Balanced / Lenient), and Hinglish Rule Layer toggle.
- Tab 1 (Live Analyzer): Pre-loaded demo dropdown, real-time classification, multi-label
  category badges, Hinglish rule alerts, "Why Was This Flagged?" coefficient attribution
  chart, preprocessing audit, and live v1 vs v2 side-by-side comparison.
- Tab 2 (Batch CSV Upload): Batch classify uploaded CSVs (or a one-click sample batch),
  inspect summary KPIs, and download annotated CSV results.
- Tab 3 (Evaluation & Benchmarks): Displays real measured v1 vs v2 metrics, 5-fold CV
  benchmark table & chart, PR threshold curves, confusion matrices, multi-label metrics,
  adversarial evaluation, and Davidson cross-domain Twitter results.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.model_training import get_default_models_dir, load_model_artifacts
from src.predict import load_v2_bundle, predict_comment


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = get_default_models_dir()


# ---------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AbuseGuard AI v2 — Content Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# 2. CLEAN SAAS CSS (SIDEBAR VISIBLE & HIGH-CONTRAST)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .main .block-container {
        max-width: 1240px !important;
        padding: 1.25rem 2rem 3.5rem 2rem !important;
    }

    .top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.65rem 0 1rem 0;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 1.25rem;
    }

    .brand-logo {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.35rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.02em;
    }

    .brand-badge {
        font-size: 0.72rem;
        font-weight: 700;
        background: #eef2ff;
        color: #4338ca;
        padding: 3px 9px;
        border-radius: 6px;
        border: 1px solid #c7d2fe;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #15803d;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 4px 12px;
        border-radius: 9999px;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #16a34a;
        border-radius: 50%;
        display: inline-block;
    }

    .result-badge-safe {
        background: #f0fdf4;
        border: 1.5px solid #86efac;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .result-badge-abusive {
        background: #fef2f2;
        border: 1.5px solid #fca5a5;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .result-status-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.15rem;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .title-safe { color: #15803d; }
    .title-abusive { color: #b91c1c; }

    .result-explanation {
        font-size: 0.88rem;
        color: #334155;
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    .cat-pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 0.5rem 0 0.85rem 0;
    }

    .cat-pill {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 3px 9px;
        border-radius: 9999px;
        border: 1px solid #fca5a5;
        background: #fee2e2;
        color: #991b1b;
    }

    .cat-pill-threat {
        border-color: #ef4444;
        background: #991b1b;
        color: #ffffff;
    }

    .cat-pill-hinglish {
        border-color: #f59e0b;
        background: #fef3c7;
        color: #92400e;
    }

    .cat-pill-safe {
        border-color: #bbf7d0;
        background: #dcfce7;
        color: #166534;
    }

    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.15rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    .stat-card-label {
        font-size: 0.73rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }

    .stat-card-num {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0.15rem 0;
    }

    .stat-card-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: #4f46e5;
    }

    .empty-state-box {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 2.5rem 1.5rem;
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        background: #ffffff;
        min-height: 260px;
    }

    .saas-footer {
        margin-top: 2.75rem;
        padding-top: 1.25rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: 0.82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 3. CACHED ARTIFACT & BENCHMARK LOADERS
# ---------------------------------------------------------------------------
def load_app_pipelines(base_models_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads v1 baseline (if present in models/v1/) and v2 production bundle (from models/v2/).
    Enforces canonical v2 loading without silent fallback to legacy root models/.
    Raises FileNotFoundError if canonical v2 artifacts are missing.
    """
    base_dir = Path(base_models_dir) if base_models_dir is not None else MODELS_DIR
    pipelines: Dict[str, Any] = {"v1": None, "v2": None}

    v1_dir = base_dir / "v1"
    if (v1_dir / "best_model.joblib").is_file() and (v1_dir / "tfidf_vectorizer.joblib").is_file():
        v1_model, v1_vec = load_model_artifacts(models_dir=v1_dir)
        pipelines["v1"] = {"model": v1_model, "vectorizer": v1_vec}

    v2_dir = base_dir / "v2"
    if (v2_dir / "best_model.joblib").is_file() and (v2_dir / "tfidf_vectorizer.joblib").is_file():
        pipelines["v2"] = load_v2_bundle(models_dir=v2_dir)
    else:
        raise FileNotFoundError(
            f"[CRITICAL ERROR] Canonical v2 production artifacts missing in '{v2_dir}'. "
            "AbuseGuard AI requires 'best_model.joblib' and 'tfidf_vectorizer.joblib' inside models/v2/. "
            "Silent fallback to legacy v1 artifacts in root models/ has been disabled."
        )

    return pipelines


@st.cache_resource(show_spinner="Loading AbuseGuard AI v1 & v2 pipelines...")
def load_all_pipelines(base_models_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    return load_app_pipelines(base_models_dir=base_models_dir)


def _read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _read_text_safe(path: Path) -> Optional[str]:
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return None


# ---------------------------------------------------------------------------
# 4. DEMO EXAMPLES & SESSION STATE
# ---------------------------------------------------------------------------
DEMO_EXAMPLES: Dict[str, str] = {
    "— Select a Demo Example —": "",
    "1. Constructive Wikipedia Edit (Safe)": (
        "Thank you so much for your constructive feedback and helpful contribution to this article."
    ),
    "2. Polite Negation Edge Case (Safe)": (
        "This edit is not bad at all; let us never attack users and keep discussions polite."
    ),
    "3. Leetspeak Evasion (Abusive)": "y0u ar3 an 1d10t and 5tup1d 10s3r",
    "4. Symbol Masking Evasion (Abusive)": "f*ck y0u and your sh!t edits",
    "5. Stretched-Letter Threat (Abusive)": "kiiiillll youuuurself looooser",
    "6. Multi-Label Violent Threat (Abusive)": (
        "You are an absolute idiot and I will punch your face if you revert this again."
    ),
    "7. Roman Hinglish Abuse (Rule Layer)": (
        "Tu bahut bada kameena aur harami hai, chup kar chutiya"
    ),
}

if "user_comment" not in st.session_state:
    st.session_state["user_comment"] = ""
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False
if "last_outputs" not in st.session_state:
    st.session_state["last_outputs"] = None
if "batch_results_df" not in st.session_state:
    st.session_state["batch_results_df"] = None


def on_demo_select() -> None:
    choice = st.session_state.get("demo_selector", "")
    text = DEMO_EXAMPLES.get(choice, "")
    if text:
        st.session_state["user_comment"] = text
        st.session_state["analyzed"] = False
        st.session_state["last_outputs"] = None


def reset_form() -> None:
    st.session_state["user_comment"] = ""
    st.session_state["analyzed"] = False
    st.session_state["last_outputs"] = None
    st.session_state["demo_selector"] = "— Select a Demo Example —"


# ---------------------------------------------------------------------------
# 5. LOAD PIPELINES & SIDEBAR CONTROLS
# ---------------------------------------------------------------------------
try:
    pipelines = load_all_pipelines()
except FileNotFoundError as err:
    st.error(f"⚠️ Model Loading Error: {err}")
    st.stop()
except Exception as err:
    st.error(f"⚠️ Unexpected Model Loading Failure: {err}")
    st.stop()

v1_bundle = pipelines.get("v1")
v2_bundle = pipelines.get("v2")

with st.sidebar:
    st.markdown("### ⚙️ Analyzer Controls")
    available_engines = []
    if v2_bundle is not None:
        available_engines.append("v2 Improved")
    if v1_bundle is not None:
        available_engines.append("v1 Original")
    if v1_bundle is not None and v2_bundle is not None:
        available_engines.append("Side-by-Side Compare")

    engine_choice = st.radio(
        "Engine Mode",
        options=available_engines or ["v2 Improved"],
        index=0,
        help="Switch between upgraded v2, frozen v1 baseline, or live side-by-side comparison.",
    )

    sensitivity_choice = st.radio(
        "Sensitivity Mode (v2)",
        options=["Balanced", "Strict", "Lenient"],
        index=0,
        help="Strict maximizes Recall; Balanced maximizes F1; Lenient maximizes Precision.",
    )

    enable_hinglish = st.toggle(
        "Enable Hinglish Rule Layer",
        value=True,
        help="Flags Roman-script Hinglish insults, profanity, threats, and hate terms.",
    )

    st.divider()
    thresholds_meta = (v2_bundle or {}).get("thresholds", {}).get("thresholds", {})
    sens_key = sensitivity_choice.lower()
    active_threshold = float(thresholds_meta.get(sens_key, {}).get("threshold", 0.5))

    st.markdown("**Active Thresholds (Val-Calibrated)**")
    if thresholds_meta:
        for mode_k, mode_label in [("strict", "Strict"), ("balanced", "Balanced"), ("lenient", "Lenient")]:
            m_info = thresholds_meta.get(mode_k, {})
            t_val = m_info.get("threshold", 0.5)
            prefix = "👉 " if mode_k == sens_key else ""
            st.caption(f"{prefix}**{mode_label}:** `{t_val:.3f}` — {m_info.get('target', '')}")
    else:
        st.caption("Default threshold: `0.500`")

    st.divider()
    st.markdown("**Architecture Snapshot**")
    st.caption(
        "- **v2 Features:** Word (1,2) 10k + Char (3,5) 15k\n"
        "- **v2 De-Obfuscation:** Leetspeak, masking & stretch\n"
        "- **Multi-Label:** 6 Jigsaw sub-types (OvR)\n"
        "- **v1 Vault:** Frozen in `models/v1/`"
    )


# ---------------------------------------------------------------------------
# 6. HEADER NAVIGATION
# ---------------------------------------------------------------------------
v2_model_name = (
    type(v2_bundle["model"]).__name__ if v2_bundle is not None else "Not Trained"
)
st.markdown(
    f"""
    <div class="top-nav">
        <div class="brand-logo">
            <span style="font-size: 1.55rem;">🛡️</span>
            <span class="brand-title">AbuseGuard AI</span>
            <span class="brand-badge">v2 Upgrade</span>
        </div>
        <div class="status-pill">
            <span class="status-dot"></span> Engine: {engine_choice} ({v2_model_name})
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 7. HELPER RENDERERS FOR TAB 1
# ---------------------------------------------------------------------------
def _plot_why_flagged(top_features: List[Dict[str, Any]], title: str = "Why Was This Flagged?") -> None:
    if not top_features:
        st.info("No active vocabulary features matched for attribution.")
        return

    feats = [f["feature"] for f in top_features][::-1]
    vals = [f["contribution"] for f in top_features][::-1]
    colors = ["#dc2626" if v > 0 else "#16a34a" for v in vals]

    fig, ax = plt.subplots(figsize=(6.2, max(2.4, 0.36 * len(feats) + 0.8)))
    ax.barh(feats, vals, color=colors, height=0.62)
    ax.axvline(0.0, color="#475569", linewidth=1.0, linestyle="--")
    ax.set_xlabel("TF-IDF × Model Weight (+ Abusive | − Safe)", fontsize=8.5, fontweight="bold")
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    ax.tick_params(axis="both", labelsize=8.5)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _render_result_panel(res: Dict[str, Any], engine_label: str, is_v2: bool = True) -> None:
    is_abusive = res["is_abusive"]
    abusive_pct = res["abusive_probability"] * 100.0
    safe_pct = res["safe_probability"] * 100.0
    thresh = res.get("threshold_used", 0.5)
    hinglish_hit = res.get("hinglish_match", {}).get("hinglish_flagged", False)

    if is_abusive:
        extra = " (Hinglish Rule Triggered)" if hinglish_hit else ""
        st.markdown(
            f"""
            <div class="result-badge-abusive">
                <div class="result-status-title title-abusive">
                    <span>⚠️</span> {engine_label}: ABUSIVE{extra}
                </div>
                <div class="result-explanation">
                    Abusive score <strong>{abusive_pct:.1f}%</strong> meets/exceeds threshold <code>{thresh:.3f}</code>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        neg_applied = res.get("english_negation_info", {}).get("adjustment_applied", False)
        extra_note = ""
        if neg_applied:
            orig_pct = res.get("ml_abusive_probability", 0.0) * 100.0
            extra_note = f"<div style='margin-top: 4px; font-size: 0.82rem; color: #22c55e;'>ℹ️ English Negation Scope Layer neutralized negated insult (raw ML model score: {orig_pct:.1f}%).</div>"
        st.markdown(
            f"""
            <div class="result-badge-safe">
                <div class="result-status-title title-safe">
                    <span>✅</span> {engine_label}: SAFE
                </div>
                <div class="result-explanation">
                    Abusive score <strong>{abusive_pct:.1f}%</strong> is below threshold <code>{thresh:.3f}</code>.{extra_note}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Safe Probability", f"{safe_pct:.1f}%")
        st.progress(min(max(safe_pct / 100.0, 0.0), 1.0))
    with c2:
        st.metric("Abusive Probability", f"{abusive_pct:.1f}%")
        st.progress(min(max(abusive_pct / 100.0, 0.0), 1.0))

    # Multi-label & Hinglish Badges
    if is_v2:
        active_cats = res.get("active_categories", [])
        hinglish_terms = res.get("hinglish_match", {}).get("matched_terms", [])
        st.markdown("**Detected Sub-Type Badges:**")
        if active_cats or hinglish_terms:
            pills_html = ['<div class="cat-pill-row">']
            for cat in active_cats:
                cls = "cat-pill cat-pill-threat" if cat == "threat" else "cat-pill"
                pills_html.append(f'<span class="{cls}">{cat.replace("_", " ")}</span>')
            for term in hinglish_terms:
                pills_html.append(f'<span class="cat-pill cat-pill-hinglish">Hinglish: {term}</span>')
            pills_html.append("</div>")
            st.markdown("".join(pills_html), unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="cat-pill-row"><span class="cat-pill cat-pill-safe">No Toxicity Sub-Types Flagged</span></div>',
                unsafe_allow_html=True,
            )

        cat_probs = res.get("category_probabilities", {})
        if cat_probs and (is_abusive or any(v >= 0.25 for v in cat_probs.values())):
            with st.expander("🏷️ All 6 Multi-Label Category Probabilities"):
                prob_df = pd.DataFrame(
                    [{"Category": k, "Probability (%)": round(v * 100, 1)} for k, v in cat_probs.items()]
                ).sort_values("Probability (%)", ascending=False)
                st.dataframe(prob_df, hide_index=True, use_container_width=True)

    # Why Was This Flagged? Chart
    _plot_why_flagged(
        res.get("top_features", []),
        title=f"Why Was This Flagged? ({engine_label})",
    )

    with st.expander(f"🔍 Preprocessing Audit ({engine_label})"):
        st.caption("Normalized text fed to the TF-IDF vectorizer:")
        st.code(res["cleaned_text"] if res["cleaned_text"] else "(Empty)", language="text")


def _run_analysis(text: str) -> Dict[str, Any]:
    outputs: Dict[str, Any] = {}
    if v1_bundle is not None and engine_choice in ("v1 Original", "Side-by-Side Compare"):
        outputs["v1"] = predict_comment(
            raw_text=text,
            model=v1_bundle["model"],
            vectorizer=v1_bundle["vectorizer"],
            normalize_obfuscation=False,
            threshold=0.5,
            multilabel_model=None,
            enable_hinglish=False,
            explain=True,
            top_k=8,
        )

    if v2_bundle is not None and engine_choice in ("v2 Improved", "Side-by-Side Compare"):
        outputs["v2"] = predict_comment(
            raw_text=text,
            model=v2_bundle["model"],
            vectorizer=v2_bundle["vectorizer"],
            normalize_obfuscation=True,
            threshold=active_threshold,
            multilabel_model=v2_bundle.get("multilabel_model"),
            enable_hinglish=enable_hinglish,
            explain=True,
            top_k=8,
        )
    return outputs


# ---------------------------------------------------------------------------
# 8. MAIN TABS (LIVE ANALYZER | BATCH CSV | EVALUATION & BENCHMARKS)
# ---------------------------------------------------------------------------
tab_live, tab_batch, tab_eval = st.tabs([
    "💬 Live Analyzer",
    "📂 Batch CSV Upload",
    "📊 Evaluation & Benchmarks",
])

# ==========================
# TAB 1: LIVE ANALYZER
# ==========================
with tab_live:
    if v1_bundle is None and v2_bundle is None:
        st.error("⚠️ No model artifacts found in `models/v1/` or `models/v2/`. Run `scripts/benchmark_models.py` first.")
    else:
        st.markdown("#### Test & Inspect Comments in Real Time")
        st.selectbox(
            "Pre-loaded Demo Examples",
            options=list(DEMO_EXAMPLES.keys()),
            key="demo_selector",
            on_change=on_demo_select,
        )

        user_input = st.text_area(
            "Comment Text",
            key="user_comment",
            height=120,
            placeholder="Paste any social media comment (English, leetspeak, masked profanity, or Roman Hinglish)...",
        )

        col_act1, col_act2, col_meta = st.columns([1.4, 0.8, 2.0])
        with col_act1:
            analyze_clicked = st.button("🔍 Analyze Comment", type="primary", use_container_width=True)
        with col_act2:
            st.button("🔄 Clear", on_click=reset_form, use_container_width=True)
        with col_meta:
            st.caption(
                f"Chars: **{len(st.session_state['user_comment']):,}** | "
                f"Engine: **{engine_choice}** | Sensitivity: **{sensitivity_choice}** (`t={active_threshold:.3f}`) | "
                f"Hinglish Layer: **{'ON' if enable_hinglish else 'OFF'}**"
            )

        if analyze_clicked:
            if not user_input or not user_input.strip():
                st.warning("⚠️ Please enter a non-empty comment before analyzing.")
                st.session_state["analyzed"] = False
                st.session_state["last_outputs"] = None
            else:
                st.session_state["last_outputs"] = _run_analysis(user_input)
                st.session_state["analyzed"] = True

        st.divider()
        if not st.session_state["analyzed"] or not st.session_state["last_outputs"]:
            st.markdown(
                """
                <div class="empty-state-box">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🛡️</div>
                    <div style="font-weight: 700; font-size: 1.05rem; color: #334155;">Awaiting Comment Input</div>
                    <div style="font-size: 0.88rem; color: #64748b; max-width: 440px; margin-top: 0.35rem;">
                        Choose a demo example from the dropdown above or type your own comment, then click <strong>Analyze Comment</strong>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            outputs = st.session_state["last_outputs"]
            if engine_choice == "Side-by-Side Compare" and "v1" in outputs and "v2" in outputs:
                col_left, col_right = st.columns(2, gap="large")
                with col_left:
                    _render_result_panel(outputs["v1"], "v1 Original", is_v2=False)
                with col_right:
                    _render_result_panel(outputs["v2"], f"v2 Improved ({sensitivity_choice})", is_v2=True)
            elif "v2" in outputs:
                _render_result_panel(outputs["v2"], f"v2 Improved ({sensitivity_choice})", is_v2=True)
            elif "v1" in outputs:
                _render_result_panel(outputs["v1"], "v1 Original", is_v2=False)


# ==========================
# TAB 2: BATCH CSV UPLOAD
# ==========================
with tab_batch:
    st.markdown("#### Batch Comment Moderation (CSV)")
    st.caption(
        f"Uses active sidebar controls: **Sensitivity = {sensitivity_choice}** (`t={active_threshold:.3f}`), "
        f"**Hinglish Layer = {'ON' if enable_hinglish else 'OFF'}**."
    )

    col_up, col_sample = st.columns([2.2, 1.0])
    with col_up:
        uploaded_file = st.file_uploader("Upload a CSV file containing comments", type=["csv"])
    with col_sample:
        st.write("")
        st.write("")
        use_sample_batch = st.button("🧪 Load Sample 8-Comment Batch", use_container_width=True)

    batch_df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif use_sample_batch:
        batch_df = pd.DataFrame({
            "id": [f"demo_{i}" for i in range(1, 9)],
            "comment_text": [
                "Thank you for fixing the citations on this page!",
                "This edit is not bad at all, let us never argue.",
                "y0u ar3 an 1d10t and 5tup1d 10s3r",
                "f*ck y0u and your sh!t edits",
                "kiiiillll youuuurself looooser",
                "I will punch your face if you revert this again.",
                "Tu bahut bada kameena aur harami hai",
                "Check out the reference section for verified data.",
            ],
        })
        st.session_state["sample_batch_df"] = batch_df
    elif "sample_batch_df" in st.session_state and uploaded_file is None:
        batch_df = st.session_state["sample_batch_df"]

    if batch_df is not None and len(batch_df) > 0:
        default_col_idx = 0
        for preferred in ["comment_text", "text", "tweet", "comment"]:
            if preferred in batch_df.columns:
                default_col_idx = list(batch_df.columns).index(preferred)
                break

        text_col = st.selectbox("Select Text Column", options=list(batch_df.columns), index=default_col_idx)
        run_batch_btn = st.button("🚀 Run Batch Prediction", type="primary")

        if run_batch_btn:
            active_bundle = v2_bundle if (engine_choice != "v1 Original" and v2_bundle is not None) else v1_bundle
            use_v2_mode = active_bundle is v2_bundle
            thresh_batch = active_threshold if use_v2_mode else 0.5
            records = []
            for raw_val in batch_df[text_col].fillna("").astype(str):
                if not raw_val.strip():
                    records.append({
                        "predicted_label": 0,
                        "verdict": "Safe",
                        "abusive_probability": 0.0,
                        "safe_probability": 1.0,
                        "active_categories": "",
                        "hinglish_flagged": False,
                    })
                else:
                    r = predict_comment(
                        raw_text=raw_val,
                        model=active_bundle["model"],
                        vectorizer=active_bundle["vectorizer"],
                        normalize_obfuscation=use_v2_mode,
                        threshold=thresh_batch,
                        multilabel_model=active_bundle.get("multilabel_model") if use_v2_mode else None,
                        enable_hinglish=enable_hinglish if use_v2_mode else False,
                        explain=False,
                    )
                    records.append({
                        "predicted_label": r["prediction_label"],
                        "verdict": r["prediction_text"],
                        "abusive_probability": round(r["abusive_probability"], 4),
                        "safe_probability": round(r["safe_probability"], 4),
                        "active_categories": ", ".join(r["active_categories"]),
                        "hinglish_flagged": r["hinglish_match"]["hinglish_flagged"],
                    })

            out_df = pd.concat([batch_df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
            st.session_state["batch_results_df"] = out_df

        if st.session_state["batch_results_df"] is not None:
            res_df = st.session_state["batch_results_df"]
            total_n = len(res_df)
            abusive_n = int((res_df["predicted_label"] == 1).sum())
            safe_n = total_n - abusive_n
            hinglish_n = int(res_df["hinglish_flagged"].sum())

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Rows", f"{total_n:,}")
            k2.metric("Flagged Abusive", f"{abusive_n:,}")
            k3.metric("Verified Safe", f"{safe_n:,}")
            k4.metric("Hinglish Rule Hits", f"{hinglish_n:,}")

            st.dataframe(res_df, use_container_width=True)
            st.download_button(
                label="⬇️ Download Annotated Results CSV",
                data=res_df.to_csv(index=False).encode("utf-8"),
                file_name="abuseguard_batch_results.csv",
                mime="text/csv",
            )


# ==========================
# TAB 3: EVALUATION & BENCHMARKS
# ==========================
with tab_eval:
    st.markdown("#### Verified Evaluation & Before/After Benchmarks")
    v1_json = _read_json_safe(RESULTS_DIR / "v1_baseline.json")
    v2_json = _read_json_safe(MODELS_DIR / "v2" / "evaluation_summary.json")
    cv_json = _read_json_safe(RESULTS_DIR / "model_comparison_cv.json")
    ml_json = _read_json_safe(RESULTS_DIR / "multilabel_metrics.json")
    cd_json = _read_json_safe(RESULTS_DIR / "cross_domain_tweets.json")

    if v1_json and v2_json:
        v1_m = v1_json["v1_test_metrics"]
        v2_modes = v2_json["test_metrics_by_mode"]
        v2_bal = v2_modes["balanced"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-card-label">Abusive F1 (v1 → v2 Balanced)</div>
                    <div class="stat-card-num" style="color: #4f46e5;">{v2_bal['f1']*100:.2f}%</div>
                    <div class="stat-card-badge">From {v1_m['abusive_f1']*100:.2f}% (+{(v2_bal['f1']-v1_m['abusive_f1'])*100:.2f} pts)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-card-label">Abusive Precision</div>
                    <div class="stat-card-num">{v2_bal['precision']*100:.2f}%</div>
                    <div class="stat-card-badge">From {v1_m['abusive_precision']*100:.2f}% ({(v2_bal['precision']-v1_m['abusive_precision'])*100:+.2f} pts)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-card-label">PR-AUC (Imbalance Metric)</div>
                    <div class="stat-card-num">{v2_json['pr_auc']:.4f}</div>
                    <div class="stat-card-badge">From {v1_m['pr_auc']:.4f} ({v2_json['pr_auc']-v1_m['pr_auc']:+.4f})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c4:
            fp_delta = v2_bal["false_positives"] - v1_m["false_positives"]
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-card-label">Test Accuracy (n=31,915)</div>
                    <div class="stat-card-num">{v2_bal['accuracy']*100:.2f}%</div>
                    <div class="stat-card-badge">False Positives: {v1_m['false_positives']:,} → {v2_bal['false_positives']:,} ({fp_delta:+,})</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")
        comp_rows = [
            {
                "Model / Operating Mode": "v1 Baseline (Word 10k, t=0.500)",
                "Accuracy (%)": round(v1_m["accuracy"] * 100, 2),
                "Precision (%)": round(v1_m["abusive_precision"] * 100, 2),
                "Recall (%)": round(v1_m["abusive_recall"] * 100, 2),
                "Abusive F1 (%)": round(v1_m["abusive_f1"] * 100, 2),
                "False Positives": v1_m["false_positives"],
                "False Negatives": v1_m["false_negatives"],
            }
        ]
        for mode_name, label in [
            ("strict", "v2 Strict (High Recall)"),
            ("balanced", "v2 Balanced (Max Val F1)"),
            ("lenient", "v2 Lenient (High Precision)"),
        ]:
            m = v2_modes[mode_name]
            comp_rows.append({
                "Model / Operating Mode": f"{label} (t={m['threshold']:.3f})",
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "Recall (%)": round(m["recall"] * 100, 2),
                "Abusive F1 (%)": round(m["f1"] * 100, 2),
                "False Positives": m["false_positives"],
                "False Negatives": m["false_negatives"],
            })
        st.markdown("##### Held-Out Jigsaw Test Set Comparison (n = 31,915)")
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)
    else:
        st.info("Run `scripts/verify_v1_baseline.py` and `scripts/benchmark_models.py` to populate metrics.")

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        if (RESULTS_DIR / "pr_curve_thresholds.png").is_file():
            st.image(str(RESULTS_DIR / "pr_curve_thresholds.png"), caption="Validation Precision-Recall Curve & Threshold Modes")
    with col_img2:
        if (RESULTS_DIR / "confusion_matrices_v1_vs_v2.png").is_file():
            st.image(str(RESULTS_DIR / "confusion_matrices_v1_vs_v2.png"), caption="Confusion Matrices (v1 vs v2 Balanced)")

    if cv_json:
        st.markdown("##### 4-Model 5-Fold Stratified Cross-Validation (127,656 Train)")
        cv_df = pd.DataFrame([
            {
                "Model": r["model"],
                "Accuracy (%)": f"{r['accuracy_mean']*100:.2f} ± {r['accuracy_std']*100:.2f}",
                "Precision (%)": f"{r['precision_mean']*100:.2f} ± {r['precision_std']*100:.2f}",
                "Recall (%)": f"{r['recall_mean']*100:.2f} ± {r['recall_std']*100:.2f}",
                "Abusive F1 (%)": f"{r['f1_mean']*100:.2f} ± {r['f1_std']*100:.2f}",
                "ROC-AUC": round(r["roc_auc_mean"], 4),
                "PR-AUC": round(r["pr_auc_mean"], 4),
            }
            for r in cv_json["cv_results"]
        ])
        st.dataframe(cv_df, hide_index=True, use_container_width=True)
        if (RESULTS_DIR / "model_comparison.png").is_file():
            st.image(str(RESULTS_DIR / "model_comparison.png"), caption="5-Fold CV Benchmark Across 4 Model Families")

    if ml_json:
        st.markdown(
            f"##### Multi-Label Sub-Type Performance (Micro-F1: {ml_json['micro_f1']*100:.2f}%, Macro-F1: {ml_json['macro_f1']*100:.2f}%)"
        )
        ml_df = pd.DataFrame([
            {
                "Category": r["label"],
                "Test Positives": r["test_positives"],
                "Precision (%)": round(r["precision"] * 100, 2),
                "Recall (%)": round(r["recall"] * 100, 2),
                "F1-Score (%)": round(r["f1"] * 100, 2),
                "ROC-AUC": round(r["roc_auc"], 4),
                "PR-AUC": round(r["pr_auc"], 4),
            }
            for r in ml_json["per_label"]
        ])
        st.dataframe(ml_df, hide_index=True, use_container_width=True)

    if cd_json:
        st.markdown("##### Cross-Domain Twitter Evaluation (Davidson 20% Test Split, n = 4,957)")
        zs_bal = cd_json["zero_shot_test_20"]["balanced"]
        da_dav = cd_json["domain_adapted_test_20"]["davidson_test_050"]
        cd_df = pd.DataFrame([
            {
                "Setting": f"Zero-Shot Jigsaw v2 (t={zs_bal['threshold']:.3f})",
                "Accuracy (%)": round(zs_bal["accuracy"] * 100, 2),
                "Precision (%)": round(zs_bal["abusive_precision"] * 100, 2),
                "Recall (%)": round(zs_bal["abusive_recall"] * 100, 2),
                "Abusive F1 (%)": round(zs_bal["abusive_f1"] * 100, 2),
                "ROC-AUC": round(zs_bal["roc_auc"], 4),
            },
            {
                "Setting": "Domain-Adapted (Jigsaw + 80% Davidson, t=0.500)",
                "Accuracy (%)": round(da_dav["accuracy"] * 100, 2),
                "Precision (%)": round(da_dav["abusive_precision"] * 100, 2),
                "Recall (%)": round(da_dav["abusive_recall"] * 100, 2),
                "Abusive F1 (%)": round(da_dav["abusive_f1"] * 100, 2),
                "ROC-AUC": round(da_dav["roc_auc"], 4),
            },
        ])
        st.dataframe(cd_df, hide_index=True, use_container_width=True)

    adv_md = _read_text_safe(RESULTS_DIR / "adversarial_examples.md")
    if adv_md:
        with st.expander("🛡️ Adversarial Edge-Case Audit (20 Cases: v1 vs v2)"):
            st.markdown(adv_md)


st.markdown(
    """
    <div class="saas-footer">
        <strong>AbuseGuard AI v2</strong> • Final Year Engineering Evaluation • Python, Scikit-learn, FastAPI & Streamlit<br>
        <em>Assistive content moderation prototype. Automated predictions should support human moderation workflows.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
