"""
tests/test_api.py - Automated tests for FastAPI AbuseGuard AI v2 service (api.py).
"""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "v2"
    assert data["multilabel_enabled"] is True


def test_predict_safe_comment():
    resp = client.post(
        "/predict",
        json={"text": "Thank you for your constructive feedback on this article.", "sensitivity": "balanced"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "Safe"
    assert data["is_abusive"] is False
    assert data["prediction_label"] == 0
    assert data["safe_probability"] > data["abusive_probability"]


def test_predict_obfuscated_abusive_comment_with_explain_and_multilabel():
    resp = client.post(
        "/predict",
        json={
            "text": "You are an 1d10t and I will k!ll you!",
            "sensitivity": "balanced",
            "explain": True,
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["verdict"] == "Abusive"
    assert data["is_abusive"] is True
    assert data["cleaned_text"] == "you are an idiot and i will kill you"
    assert len(data["active_categories"]) >= 1
    assert "threat" in data["category_probabilities"]
    assert len(data["top_features"]) > 0


def test_predict_hinglish_toggle():
    hinglish_text = "Tu bahut bada kameena aur harami hai"
    resp_on = client.post(
        "/predict",
        json={"text": hinglish_text, "enable_hinglish": True},
    )
    assert resp_on.status_code == 200
    data_on = resp_on.json()
    assert data_on["verdict"] == "Abusive"
    assert data_on["hinglish_match"]["hinglish_flagged"] is True
    assert "kameena" in data_on["hinglish_match"]["matched_terms"]

    resp_off = client.post(
        "/predict",
        json={"text": hinglish_text, "enable_hinglish": False},
    )
    assert resp_off.status_code == 200
    assert resp_off.json()["hinglish_match"]["hinglish_flagged"] is False


def test_predict_sensitivity_thresholds():
    t1 = client.post("/predict", json={"text": "Nice edit thanks", "sensitivity": "strict"}).json()["threshold_used"]
    t2 = client.post("/predict", json={"text": "Nice edit thanks", "sensitivity": "balanced"}).json()["threshold_used"]
    t3 = client.post("/predict", json={"text": "Nice edit thanks", "sensitivity": "lenient"}).json()["threshold_used"]
    assert t1 < t2 < t3


def test_predict_blank_text_returns_422():
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422

