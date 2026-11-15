"""
feature_extraction.py - TF-IDF Feature Extraction for NLP (v1 & v2 FeatureUnion).

This module converts preprocessed text comments into numerical sparse matrices using:
- v1: Single word-level TfidfVectorizer (unigrams & bigrams).
- v2: FeatureUnion combining Word TF-IDF (1,2)-grams + Character-boundary TF-IDF
      (analyzer='char_wb', ngram_range=(3,5)) to capture typos, obfuscations,
      morphological variants, and subword abusive roots.
"""

from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.utils.validation import check_is_fitted


def create_tfidf_vectorizer(
    max_features: Optional[int] = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: Union[int, float] = 2,
    sublinear_tf: bool = True,
    **kwargs: Any,
) -> TfidfVectorizer:
    """
    Creates and configures a Scikit-learn TfidfVectorizer with recommended NLP defaults.
    Preserved intact for backward compatibility and v1 compatibility.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        **kwargs,
    )
    return vectorizer


def create_feature_union_vectorizer(
    word_max_features: Optional[int] = 10000,
    word_ngram_range: Tuple[int, int] = (1, 2),
    word_min_df: Union[int, float] = 2,
    char_max_features: Optional[int] = 15000,
    char_ngram_range: Tuple[int, int] = (3, 5),
    char_min_df: Union[int, float] = 5,
    sublinear_tf: bool = True,
) -> FeatureUnion:
    """
    Creates a v2 Scikit-learn FeatureUnion combining:
    - Word TF-IDF: ngram_range=(1, 2), max_features=10000, min_df=2, sublinear_tf=True
    - Character-boundary TF-IDF: analyzer='char_wb', ngram_range=(3, 5),
      max_features=15000, min_df=5, sublinear_tf=True

    This hybrid representation captures full-word/phrase context (including negations)
    alongside subword character patterns resilient to slang, typos, and masking.
    """
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=word_ngram_range,
        max_features=word_max_features,
        min_df=word_min_df,
        sublinear_tf=sublinear_tf,
    )
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        max_features=char_max_features,
        min_df=char_min_df,
        sublinear_tf=sublinear_tf,
    )
    return FeatureUnion(
        transformer_list=[
            ("word", word_tfidf),
            ("char", char_tfidf),
        ]
    )


def _is_vectorizer_fitted(vectorizer: Any) -> bool:
    """Checks whether a TfidfVectorizer or FeatureUnion has been fitted."""
    if isinstance(vectorizer, FeatureUnion):
        try:
            for _, transformer in vectorizer.transformer_list:
                if not hasattr(transformer, "vocabulary_") or transformer.vocabulary_ is None:
                    return False
                check_is_fitted(transformer, "idf_")
            return True
        except Exception:
            return False

    return hasattr(vectorizer, "vocabulary_") and vectorizer.vocabulary_ is not None


def get_feature_names(vectorizer: Any) -> np.ndarray:
    """
    Returns ordered feature names from either a fitted TfidfVectorizer or FeatureUnion.
    """
    if not _is_vectorizer_fitted(vectorizer):
        raise NotFittedError(
            "[ERROR] Vectorizer is not fitted yet. Fit on training data first."
        )

    if hasattr(vectorizer, "get_feature_names_out"):
        return vectorizer.get_feature_names_out()
    return np.array(vectorizer.get_feature_names())


def _validate_text_collection(texts: Any) -> List[str]:
    """
    Internal helper to validate that the input collection is non-empty
    and contains text-compatible items.
    """
    if texts is None:
        raise ValueError("[ERROR] Input text collection cannot be None.")

    if isinstance(texts, (pd.Series, np.ndarray)):
        text_list = texts.tolist()
    elif isinstance(texts, (list, tuple)):
        text_list = list(texts)
    else:
        raise ValueError(
            f"[ERROR] Expected a collection (list, Series, array), got {type(texts).__name__}."
        )

    if len(text_list) == 0:
        raise ValueError("[ERROR] Cannot extract features from an empty collection of comments.")

    return [str(t) if t is not None else "" for t in text_list]


def fit_tfidf_vectorizer(
    vectorizer: Union[TfidfVectorizer, FeatureUnion],
    train_texts: Union[pd.Series, List[str], np.ndarray],
) -> Union[TfidfVectorizer, FeatureUnion]:
    """
    Fits the vectorizer (TfidfVectorizer or FeatureUnion) strictly on training comments.
    """
    validated_texts = _validate_text_collection(train_texts)

    try:
        vectorizer.fit(validated_texts)
    except Exception as exc:
        raise ValueError(f"[ERROR] Failed to fit TfidfVectorizer: {exc}") from exc

    return vectorizer


def transform_tfidf(
    vectorizer: Union[TfidfVectorizer, FeatureUnion],
    texts: Union[pd.Series, List[str], np.ndarray],
) -> csr_matrix:
    """
    Transforms text comments into a numerical sparse TF-IDF matrix using an
    ALREADY-FITTED TfidfVectorizer or FeatureUnion.
    """
    if not _is_vectorizer_fitted(vectorizer):
        raise NotFittedError(
            "[ERROR] TfidfVectorizer is not fitted yet. Call 'fit_tfidf_vectorizer' "
            "with training data before calling 'transform_tfidf'."
        )

    validated_texts = _validate_text_collection(texts)
    return vectorizer.transform(validated_texts)


def fit_transform_tfidf(
    vectorizer: Union[TfidfVectorizer, FeatureUnion],
    train_texts: Union[pd.Series, List[str], np.ndarray],
) -> Tuple[Union[TfidfVectorizer, FeatureUnion], csr_matrix]:
    """
    Fits the vectorizer on training comments AND transforms them into a sparse
    feature matrix in a single step.
    """
    fitted_vec = fit_tfidf_vectorizer(vectorizer, train_texts)
    features = transform_tfidf(fitted_vec, train_texts)
    return fitted_vec, features
