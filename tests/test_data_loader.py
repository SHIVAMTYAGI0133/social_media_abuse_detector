"""
test_data_loader.py - Unit tests for data loading and preprocessing.

NOTE ON TEST DATA:
=============================================================================
All data fixtures used in this test file are SYNTHETIC TOY EXAMPLES created
strictly for automated unit testing purposes. They do NOT contain or represent
real comments or records from the Jigsaw Toxic Comment Classification dataset.
=============================================================================
"""

import os
import sys
import tempfile
import pytest
import pandas as pd
from pathlib import Path

# Add project root directory to sys.path to ensure src is discoverable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import (
    load_data,
    check_file_exists,
    validate_columns,
    validate_data_content,
    REQUIRED_COLUMNS,
    TOXICITY_COLUMNS,
)
from src.preprocessing import (
    validate_labels,
    handle_missing_comments,
    remove_duplicates,
    create_abusive_label,
    prepare_dataset,
)


@pytest.fixture
def synthetic_valid_df() -> pd.DataFrame:
    """
    Creates a synthetic toy DataFrame for unit testing.
    CLEARLY LABELED: This is synthetic test fixture data, NOT real data.
    """
    return pd.DataFrame({
        "id": ["test_001", "test_002", "test_003", "test_004", "test_005"],
        "comment_text": [
            "Hello, thank you for your helpful contribution to Wikipedia!",
            "I strongly disagree with this edit, let us discuss on talk page.",
            "You are an absolute idiot and cannot write anything good.",
            "I will find you and harm you for reverting my edits.",
            "Your ethnic background makes you completely unqualified for this.",
        ],
        "toxic":         [0, 0, 1, 1, 1],
        "severe_toxic":  [0, 0, 0, 1, 0],
        "obscene":       [0, 0, 1, 0, 0],
        "threat":        [0, 0, 0, 1, 0],
        "insult":        [0, 0, 1, 0, 1],
        "identity_hate": [0, 0, 0, 0, 1],
    })


@pytest.fixture
def synthetic_csv_file(synthetic_valid_df: pd.DataFrame, tmp_path: Path) -> Path:
    """
    Writes synthetic test data to a temporary CSV file.
    """
    temp_file = tmp_path / "synthetic_train.csv"
    synthetic_valid_df.to_csv(temp_file, index=False)
    return temp_file


# ---------------------------------------------------------------------------
# DATA LOADER TESTS
# ---------------------------------------------------------------------------

def test_load_data_success_with_synthetic_csv(synthetic_csv_file: Path):
    """Verifies that load_data loads a valid synthetic CSV correctly."""
    df = load_data(synthetic_csv_file)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in REQUIRED_COLUMNS:
        assert col in df.columns


def test_load_data_file_not_found():
    """Verifies that attempting to load a non-existent file raises FileNotFoundError."""
    non_existent_path = Path("non_existent_directory_12345") / "train.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        check_file_exists(non_existent_path)
    assert "Dataset file not found" in str(exc_info.value)
    assert "https://www.kaggle.com" in str(exc_info.value)


def test_load_data_missing_columns(tmp_path: Path):
    """Verifies that missing required columns triggers a ValueError."""
    bad_df = pd.DataFrame({
        "id": ["1", "2"],
        "comment_text": ["Sample comment A", "Sample comment B"],
        # Missing all 6 toxicity label columns
    })
    bad_csv = tmp_path / "missing_cols.csv"
    bad_df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError) as exc_info:
        load_data(bad_csv)
    assert "missing required column(s)" in str(exc_info.value)


def test_validate_data_content_empty_df():
    """Verifies that an empty DataFrame triggers a ValueError."""
    empty_df = pd.DataFrame(columns=REQUIRED_COLUMNS)
    with pytest.raises(ValueError) as exc_info:
        validate_data_content(empty_df)
    assert "contains 0 rows" in str(exc_info.value)


def test_validate_data_content_all_null_comments():
    """Verifies that all-null comments trigger a ValueError."""
    null_comments_df = pd.DataFrame({
        "comment_text": [None, None],
        "toxic": [0, 0],
        "severe_toxic": [0, 0],
        "obscene": [0, 0],
        "threat": [0, 0],
        "insult": [0, 0],
        "identity_hate": [0, 0],
    })
    with pytest.raises(ValueError) as exc_info:
        validate_data_content(null_comments_df)
    assert "only null/empty values" in str(exc_info.value)


# ---------------------------------------------------------------------------
# LABEL VALIDATION & PREPROCESSING TESTS
# ---------------------------------------------------------------------------

def test_validate_labels_success(synthetic_valid_df: pd.DataFrame):
    """Verifies that strictly binary labels pass validation without error."""
    # Should not raise any exception
    validate_labels(synthetic_valid_df, TOXICITY_COLUMNS)


def test_validate_labels_invalid_value():
    """Verifies that non-binary label values (e.g., 2, -1) raise ValueError."""
    invalid_df = pd.DataFrame({
        "comment_text": ["Test comment"],
        "toxic": [2],  # Invalid value!
        "severe_toxic": [0],
        "obscene": [0],
        "threat": [0],
        "insult": [0],
        "identity_hate": [0],
    })
    with pytest.raises(ValueError) as exc_info:
        validate_labels(invalid_df, TOXICITY_COLUMNS)
    assert "invalid non-binary value" in str(exc_info.value)


def test_validate_labels_nan_value():
    """Verifies that NaN labels raise ValueError without silent conversion."""
    nan_df = pd.DataFrame({
        "comment_text": ["Test comment"],
        "toxic": [None],  # Null value
        "severe_toxic": [0],
        "obscene": [0],
        "threat": [0],
        "insult": [0],
        "identity_hate": [0],
    })
    with pytest.raises(ValueError) as exc_info:
        validate_labels(nan_df, TOXICITY_COLUMNS)
    assert "null/missing value" in str(exc_info.value)


def test_create_abusive_label_logic():
    """
    Verifies that the combined 'abusive' column is computed accurately:
    - Row with all 0s -> abusive = 0
    - Row with any 1 -> abusive = 1
    - Original columns are preserved
    """
    test_df = pd.DataFrame({
        "comment_text": ["Safe comment", "Threat only", "Multiple labels", "Identity hate only"],
        "toxic":         [0, 0, 1, 0],
        "severe_toxic":  [0, 0, 1, 0],
        "obscene":       [0, 0, 1, 0],
        "threat":        [0, 1, 0, 0],  # Threat=1, but toxic=0
        "insult":        [0, 0, 1, 0],
        "identity_hate": [0, 0, 0, 1],  # Hate=1, but toxic=0
    })

    result_df = create_abusive_label(test_df)

    # Check that abusive column exists
    assert "abusive" in result_df.columns

    # Row 0: All zeros -> abusive must be 0
    assert result_df.loc[0, "abusive"] == 0

    # Row 1: Threat only (toxic=0) -> abusive must be 1
    assert result_df.loc[1, "abusive"] == 1

    # Row 2: Multiple toxic subcategories -> abusive must be 1
    assert result_df.loc[2, "abusive"] == 1

    # Row 3: Identity hate only (toxic=0) -> abusive must be 1
    assert result_df.loc[3, "abusive"] == 1

    # Original columns must be preserved
    for col in TOXICITY_COLUMNS:
        assert col in result_df.columns
        assert list(result_df[col]) == list(test_df[col])


def test_handle_missing_comments():
    """Verifies that missing or whitespace-only comments are handled properly."""
    df_with_missing = pd.DataFrame({
        "comment_text": ["Valid text", None, "   ", "Another valid text"],
        "toxic": [0, 0, 0, 1],
    })

    # Test drop strategy (default)
    cleaned_drop = handle_missing_comments(df_with_missing, drop_missing=True)
    assert len(cleaned_drop) == 2
    assert list(cleaned_drop["comment_text"]) == ["Valid text", "Another valid text"]

    # Test fill strategy
    cleaned_fill = handle_missing_comments(df_with_missing, drop_missing=False)
    assert len(cleaned_fill) == 4
    assert cleaned_fill.loc[1, "comment_text"] == ""


def test_remove_duplicates():
    """Verifies that duplicate comments are removed, keeping the first occurrence."""
    df_with_dups = pd.DataFrame({
        "comment_text": [
            "Great work!",
            "Great work!",  # Duplicate
            "Terrible work!",
            "Great work!",  # Duplicate
        ],
        "toxic": [0, 0, 1, 0],
    })

    cleaned = remove_duplicates(df_with_dups, subset=["comment_text"])
    assert len(cleaned) == 2
    assert list(cleaned["comment_text"]) == ["Great work!", "Terrible work!"]


def test_prepare_dataset_full_pipeline(synthetic_valid_df: pd.DataFrame):
    """Verifies the complete end-to-end preprocessing pipeline on synthetic data."""
    # Add a duplicate and a null comment to synthetic fixture to test full pipeline
    dirty_synthetic = pd.concat([
        synthetic_valid_df,
        pd.DataFrame({
            "id": ["test_dup", "test_null"],
            "comment_text": [synthetic_valid_df.iloc[0]["comment_text"], None],
            "toxic": [0, 0],
            "severe_toxic": [0, 0],
            "obscene": [0, 0],
            "threat": [0, 0],
            "insult": [0, 0],
            "identity_hate": [0, 0],
        })
    ], ignore_index=True)

    prepared_df = prepare_dataset(dirty_synthetic)

    # Should have dropped the null row and the duplicate row
    assert len(prepared_df) == len(synthetic_valid_df)
    assert "abusive" in prepared_df.columns
    assert set(prepared_df["abusive"].unique()).issubset({0, 1})
