"""
test_feature_extraction.py - Unit tests for TF-IDF feature extraction.

NOTE ON TEST DATA:
=============================================================================
All data fixtures used in this test file are small SYNTHETIC TOY EXAMPLES created
strictly for automated unit testing purposes. They do NOT contain or represent
real comments from the Kaggle Jigsaw Toxic Comment Classification dataset.
=============================================================================
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import (
    create_tfidf_vectorizer,
    create_feature_union_vectorizer,
    fit_tfidf_vectorizer,
    transform_tfidf,
    fit_transform_tfidf,
    get_feature_names,
)


@pytest.fixture
def synthetic_train_comments() -> list:
    """Small synthetic training comments for unit testing."""
    return [
        "this is a clean and helpful comment",
        "you are rude and disrespectful",
        "i do not agree with your edits",
        "another clean and constructive message",
    ]


@pytest.fixture
def synthetic_test_comments() -> list:
    """Small synthetic unseen test comments for unit testing."""
    return [
        "this is disrespectful and not helpful",
        "a completely new sentence with unlearned words",
    ]


def test_create_vectorizer_defaults():
    """Verifies that vectorizer is initialized with expected NLP configuration."""
    vec = create_tfidf_vectorizer(max_features=1000, ngram_range=(1, 2), min_df=1)
    assert vec.max_features == 1000
    assert vec.ngram_range == (1, 2)
    assert vec.min_df == 1
    assert not hasattr(vec, "vocabulary_")


def test_fit_and_matrix_shape(synthetic_train_comments):
    """
    Verifies that fitting on synthetic comments produces a vocabulary and that
    the transformed sparse matrix has the expected number of rows.
    """
    vec = create_tfidf_vectorizer(max_features=50, ngram_range=(1, 2), min_df=1)
    fitted_vec, matrix = fit_transform_tfidf(vec, synthetic_train_comments)

    assert hasattr(fitted_vec, "vocabulary_")
    assert len(fitted_vec.vocabulary_) > 0

    # Number of rows must equal number of input comments
    assert matrix.shape[0] == len(synthetic_train_comments)
    # Output must be a Scipy sparse matrix
    assert issparse(matrix)


def test_transform_unseen_comments_does_not_refit_vocabulary(
    synthetic_train_comments, synthetic_test_comments
):
    """
    CRITICAL CHECK: Verifies that transforming new comments uses the fixed
    learned vocabulary and does NOT refit or change the vocabulary size.
    """
    vec = create_tfidf_vectorizer(max_features=100, ngram_range=(1, 2), min_df=1)
    fitted_vec = fit_tfidf_vectorizer(vec, synthetic_train_comments)

    initial_vocab_size = len(fitted_vec.vocabulary_)
    initial_vocab_snapshot = dict(fitted_vec.vocabulary_)

    # Transform new, previously unseen comments
    test_matrix = transform_tfidf(fitted_vec, synthetic_test_comments)

    # 1. Row count matches test comments
    assert test_matrix.shape[0] == len(synthetic_test_comments)
    # 2. Column count strictly matches training vocabulary size
    assert test_matrix.shape[1] == initial_vocab_size
    # 3. Vocabulary has NOT mutated or grown
    assert fitted_vec.vocabulary_ == initial_vocab_snapshot


def test_transform_before_fit_raises_error(synthetic_test_comments):
    """Verifies that attempting to transform with an un-fitted vectorizer raises an error."""
    unfitted_vec = create_tfidf_vectorizer()
    with pytest.raises(NotFittedError) as exc_info:
        transform_tfidf(unfitted_vec, synthetic_test_comments)
    assert "not fitted yet" in str(exc_info.value)


def test_fit_empty_input_raises_value_error():
    """Verifies that passing an empty collection to fit raises ValueError."""
    vec = create_tfidf_vectorizer()
    with pytest.raises(ValueError) as exc_info:
        fit_tfidf_vectorizer(vec, [])
    assert "empty collection" in str(exc_info.value)


def test_transform_empty_input_raises_value_error(synthetic_train_comments):
    """Verifies that passing an empty collection to transform raises ValueError."""
    vec = create_tfidf_vectorizer(min_df=1)
    fitted_vec = fit_tfidf_vectorizer(vec, synthetic_train_comments)
    with pytest.raises(ValueError) as exc_info:
        transform_tfidf(fitted_vec, [])
    assert "empty collection" in str(exc_info.value)


def test_end_to_end_preprocessing_and_tfidf_integration():
    """
    Verifies that raw synthetic comments can be preprocessed with text_preprocessing
    and subsequently transformed with feature_extraction.
    """
    from src.text_preprocessing import preprocess_text_series

    raw_comments = [
        "YOU are being very RUDE!!! Visit http://spam.com",
        "This is NOT acceptable, never do it!",
        "hello.world constructive comment here",
    ]

    # Step 1: Preprocess
    cleaned_comments = preprocess_text_series(raw_comments)
    assert cleaned_comments.iloc[0] == "you are being very rude visit"
    assert "http" not in cleaned_comments.iloc[0]
    assert "not" in cleaned_comments.iloc[1]
    assert "hello world" in cleaned_comments.iloc[2]

    # Step 2: TF-IDF Feature Extraction
    vectorizer = create_tfidf_vectorizer(max_features=20, min_df=1)
    fitted_vec, matrix = fit_transform_tfidf(vectorizer, cleaned_comments)

    assert matrix.shape[0] == 3
    assert matrix.shape[1] <= 20
    assert issparse(matrix)


def test_create_feature_union_defaults():
    """Verifies v2 FeatureUnion combines word (1,2) and char_wb (3,5) TF-IDF branches."""
    from sklearn.pipeline import FeatureUnion

    fu = create_feature_union_vectorizer()
    assert isinstance(fu, FeatureUnion)
    transformers = dict(fu.transformer_list)
    assert "word" in transformers and "char" in transformers

    word_vec = transformers["word"]
    char_vec = transformers["char"]
    assert word_vec.ngram_range == (1, 2)
    assert word_vec.max_features == 10000
    assert word_vec.min_df == 2
    assert word_vec.sublinear_tf is True

    assert char_vec.analyzer == "char_wb"
    assert char_vec.ngram_range == (3, 5)
    assert char_vec.max_features == 15000
    assert char_vec.min_df == 5
    assert char_vec.sublinear_tf is True


def test_feature_union_fit_transform_and_feature_names(
    synthetic_train_comments, synthetic_test_comments
):
    """Verifies fitting, transforming, unfitted guard, and feature names on FeatureUnion."""
    unfitted_fu = create_feature_union_vectorizer(
        word_max_features=100, word_min_df=1, char_max_features=200, char_min_df=1
    )
    with pytest.raises(NotFittedError):
        transform_tfidf(unfitted_fu, synthetic_test_comments)

    fitted_fu, train_mat = fit_transform_tfidf(unfitted_fu, synthetic_train_comments)
    test_mat = transform_tfidf(fitted_fu, synthetic_test_comments)

    assert issparse(train_mat)
    assert issparse(test_mat)
    assert train_mat.shape[0] == len(synthetic_train_comments)
    assert test_mat.shape[0] == len(synthetic_test_comments)
    assert train_mat.shape[1] == test_mat.shape[1]

    feat_names = get_feature_names(fitted_fu)
    assert len(feat_names) == train_mat.shape[1]
    assert any(name.startswith("word__") for name in feat_names)
    assert any(name.startswith("char__") for name in feat_names)
