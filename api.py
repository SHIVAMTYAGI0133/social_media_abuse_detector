"""
api.py - FastAPI REST Service for AbuseGuard AI v2.

Provides:
- GET /health  : Service and loaded model status
- POST /predict: Single-comment moderation inference with sensitivity calibration
                 (strict / balanced / lenient), multi-label sub-types, Hinglish rule
                 layer toggle, and token/subword feature explanations.

Usage:
    uvicorn api:app --host 127.0.0.1 --port 8000
"""

from functools import lru_cache
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.predict import load_v2_bundle, predict_comment


app = FastAPI(
    title="AbuseGuard AI API",
    description="v2 Social Media Violent & Abusive Content Moderation API",
    version="2.0.0",
)


@lru_cache(maxsize=1)
def get_v2_bundle() -> Dict[str, Any]:
    """Loads and caches v2 model, FeatureUnion vectorizer, multi-label classifier, and thresholds."""
    return load_v2_bundle()


class PredictRequest(BaseModel):
    text: str = Field(..., description="Raw social media comment text to analyze.")
    sensitivity: Literal["strict", "balanced", "lenient"] = Field(
        default="balanced",
        description="Threshold calibration mode: strict (high recall), balanced (max F1), or lenient (high precision).",
    )
    enable_hinglish: bool = Field(
        default=True,
        description="Enable the explicit Roman-script Hinglish abuse rule layer.",
    )
    explain: bool = Field(
        default=True,
        description="Include top words/subwords contributing to the prediction.",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=25,
        description="Number of top contributing features to return.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Comment text must not be empty or whitespace-only.")
        return value


class FeatureContribution(BaseModel):
    feature: str
    raw_feature: str
    tfidf_value: float
    coefficient: float
    contribution: float
    direction: str


class PredictResponse(BaseModel):
    verdict: str
    is_abusive: bool
    prediction_label: int
    abusive_probability: float
    safe_probability: float
    ml_abusive_probability: float
    sensitivity_mode: str
    threshold_used: float
    cleaned_text: str
    active_categories: List[str]
    category_probabilities: Dict[str, float]
    hinglish_match: Dict[str, Any]
    top_features: List[FeatureContribution]


def _resolve_threshold(bundle: Dict[str, Any], sensitivity: str) -> float:
    thresholds_dict = bundle.get("thresholds", {}).get("thresholds", {})
    if sensitivity in thresholds_dict:
        return float(thresholds_dict[sensitivity]["threshold"])
    return 0.5


@app.get("/health")
def health_check() -> Dict[str, Any]:
    try:
        bundle = get_v2_bundle()
        model_type = type(bundle["model"]).__name__
        has_multilabel = bundle.get("multilabel_model") is not None
        modes = list(bundle.get("thresholds", {}).get("thresholds", {}).keys()) or ["balanced"]
        return {
            "status": "ok",
            "model_version": "v2",
            "model_type": model_type,
            "multilabel_enabled": has_multilabel,
            "available_sensitivities": modes,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    try:
        bundle = get_v2_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model artifacts unavailable: {exc}") from exc

    threshold = _resolve_threshold(bundle, payload.sensitivity)
    result = predict_comment(
        raw_text=payload.text,
        model=bundle["model"],
        vectorizer=bundle["vectorizer"],
        normalize_obfuscation=True,
        threshold=threshold,
        multilabel_model=bundle.get("multilabel_model"),
        enable_hinglish=payload.enable_hinglish,
        explain=payload.explain,
        top_k=payload.top_k,
    )

    return PredictResponse(
        verdict=result["prediction_text"],
        is_abusive=result["is_abusive"],
        prediction_label=result["prediction_label"],
        abusive_probability=result["abusive_probability"],
        safe_probability=result["safe_probability"],
        ml_abusive_probability=result["ml_abusive_probability"],
        sensitivity_mode=payload.sensitivity,
        threshold_used=result["threshold_used"],
        cleaned_text=result["cleaned_text"],
        active_categories=result["active_categories"],
        category_probabilities=result["category_probabilities"],
        hinglish_match=result["hinglish_match"],
        top_features=[FeatureContribution(**f) for f in result.get("top_features", [])],
    )

