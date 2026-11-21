"""
tests/test_artifact_integrity.py - Regression Tests for Fix 1 (F-05: Artifact Integrity & Loading Consistency).

Verifies:
1. Canonical artifact loading (models/v2/) is strictly enforced across api.py, app.py, and src.predict.
2. Silent fallbacks to legacy v1 artifacts in root models/ are completely eliminated.
3. Loading from a missing or empty directory raises an explicit FileNotFoundError.
4. train.py CLI argument and output directory resolution prevents accidental overwrite of models/v2/.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import app, get_v2_bundle
from app import load_app_pipelines
from fastapi.testclient import TestClient
from src.predict import get_prediction_pipeline, load_v2_bundle
from train import parse_args, resolve_training_output_dir


def test_get_prediction_pipeline_defaults_to_v2():
    """Verifies that get_prediction_pipeline() without arguments loads the canonical v2 bundle."""
    model, vectorizer = get_prediction_pipeline()
    assert type(model).__name__ == "CalibratedClassifierCV", "Default pipeline must use v2 CalibratedClassifierCV"
    assert hasattr(vectorizer, "transformer_list"), "Default vectorizer must be v2 FeatureUnion"
    transformed = vectorizer.transform(["test comment"])
    assert transformed.shape[1] == 25000, f"Expected 25,000 sparse dimensions, got {transformed.shape[1]}"


def test_load_v2_bundle_defaults_to_v2():
    """Verifies that load_v2_bundle() without arguments loads all v2 components."""
    bundle = load_v2_bundle()
    assert "model" in bundle and "vectorizer" in bundle
    assert "multilabel_model" in bundle and "thresholds" in bundle
    assert bundle["multilabel_model"] is not None, "v2 bundle must include multilabel model"
    assert type(bundle["model"]).__name__ == "CalibratedClassifierCV"
    assert "thresholds" in bundle["thresholds"]
    assert "balanced" in bundle["thresholds"]["thresholds"]


def test_api_defaults_to_v2():
    """Verifies that api.py enforces v2 bundle loading and health reports v2."""
    bundle = get_v2_bundle()
    assert type(bundle["model"]).__name__ == "CalibratedClassifierCV"

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "v2"
    assert data["multilabel_enabled"] is True


def test_app_loader_helper_defaults_to_v2():
    """Verifies that app.py's load_app_pipelines() loads v2 without silent fallback."""
    pipelines = load_app_pipelines()
    assert pipelines["v2"] is not None, "app loader helper must load v2 bundle"
    assert type(pipelines["v2"]["model"]).__name__ == "CalibratedClassifierCV"
    assert pipelines["v2"]["multilabel_model"] is not None


def test_loaders_raise_on_missing_directory(tmp_path: Path):
    """
    Verifies that get_prediction_pipeline, load_v2_bundle, and load_app_pipelines
    all raise FileNotFoundError when the target directory lacks v2 artifacts,
    strictly preventing silent fallback to legacy models.
    """
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir()

    with pytest.raises(FileNotFoundError) as exc_p:
        get_prediction_pipeline(models_dir=empty_dir)
    assert "Trained model file not found" in str(exc_p.value)

    with pytest.raises(FileNotFoundError) as exc_b:
        load_v2_bundle(models_dir=empty_dir)
    assert "Trained model file not found" in str(exc_b.value)

    with pytest.raises(FileNotFoundError) as exc_a:
        load_app_pipelines(base_models_dir=empty_dir)
    assert "Canonical v2 production artifacts missing" in str(exc_a.value)


def test_app_loader_refuses_silent_fallback_when_root_has_v1_only(tmp_path: Path):
    """
    Simulates the exact hazard: root models/ has legacy best_model.joblib, but models/v2 is missing.
    Verifies that load_app_pipelines raises FileNotFoundError instead of silently serving v1.
    """
    mock_base = tmp_path / "mock_models"
    mock_base.mkdir()

    # Create dummy file in root only (simulating legacy v1 file)
    dummy_file = mock_base / "best_model.joblib"
    dummy_file.write_text("dummy")

    with pytest.raises(FileNotFoundError) as exc:
        load_app_pipelines(base_models_dir=mock_base)
    assert "Canonical v2 production artifacts missing" in str(exc.value)


def test_train_path_resolution_default_run_dir(tmp_path: Path):
    """Verifies that train.py resolve_training_output_dir defaults to a timestamped run directory."""
    resolved = resolve_training_output_dir(base_models_dir=tmp_path)
    runs_dir = (tmp_path / "runs").resolve()
    assert str(resolved).startswith(str(runs_dir))
    assert "run_" in resolved.name
    assert resolved != (tmp_path / "v2").resolve()
    assert resolved != tmp_path.resolve()


def test_train_refuses_overwrite_v2_without_flag(tmp_path: Path):
    """Verifies that train.py raises PermissionError when targeting models/v2 without --allow-overwrite-production."""
    v2_target = tmp_path / "v2"
    with pytest.raises(PermissionError) as exc:
        resolve_training_output_dir(
            output_dir=v2_target,
            allow_overwrite_production=False,
            base_models_dir=tmp_path,
        )
    assert "canonical production directory (models/v2)" in str(exc.value)
    assert "--allow-overwrite-production" in str(exc.value)


def test_train_refuses_overwrite_root_without_flag(tmp_path: Path):
    """Verifies that train.py raises PermissionError when targeting root models without --allow-overwrite-production."""
    with pytest.raises(PermissionError) as exc:
        resolve_training_output_dir(
            output_dir=tmp_path,
            allow_overwrite_production=False,
            base_models_dir=tmp_path,
        )
    assert "root models directory" in str(exc.value)
    assert "--allow-overwrite-production" in str(exc.value)


def test_train_allows_overwrite_v2_with_explicit_flag(tmp_path: Path):
    """Verifies that train.py permits targeting models/v2 when allow_overwrite_production=True."""
    v2_target = tmp_path / "v2"
    resolved = resolve_training_output_dir(
        output_dir=v2_target,
        allow_overwrite_production=True,
        base_models_dir=tmp_path,
    )
    assert resolved == v2_target.resolve()


def test_train_cli_argument_parsing():
    """Verifies train.py CLI parser defaults and flags."""
    args_default = parse_args([])
    assert args_default.pipeline == "v2"
    assert args_default.output_dir is None
    assert args_default.allow_overwrite_production is False
    assert args_default.dataset is None

    args_custom = parse_args([
        "--dataset", "custom/data.csv",
        "--output-dir", "models/v2",
        "--allow-overwrite-production",
        "--pipeline", "v2",
    ])
    assert args_custom.dataset == "custom/data.csv"
    assert args_custom.output_dir == "models/v2"
    assert args_custom.allow_overwrite_production is True
    assert args_custom.pipeline == "v2"

