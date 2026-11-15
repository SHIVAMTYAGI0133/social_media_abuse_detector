"""
Social Media Abuse Detector - Source Package

Contains modules for:
- Part 1: Data loading, validation, and label preparation
- Part 2: Text preprocessing, normalization, and TF-IDF feature extraction
- Part 3: Model training, evaluation, comparison, and persistence
- Part 4: Live prediction inference and web application integration
"""

from .data_loader import (
    load_data,
    get_default_dataset_path,
    REQUIRED_COLUMNS,
    TOXICITY_COLUMNS,
)
from .preprocessing import (
    validate_labels,
    handle_missing_comments,
    remove_duplicates,
    create_abusive_label,
    prepare_dataset,
)
from .text_preprocessing import (
    preprocess_text,
    preprocess_text_series,
    normalize_obfuscation_text,
)
from .feature_extraction import (
    create_tfidf_vectorizer,
    create_feature_union_vectorizer,
    fit_tfidf_vectorizer,
    transform_tfidf,
    fit_transform_tfidf,
    get_feature_names,
)
from .model_training import (
    split_data,
    train_logistic_regression,
    train_naive_bayes,
    evaluate_model,
    compare_and_select_best_model,
    plot_confusion_matrix,
    save_model_artifacts,
    load_model_artifacts,
    run_training_pipeline,
    get_default_models_dir,
)
from .predict import (
    get_prediction_pipeline,
    predict_comment,
)
from .hinglish_detector import (
    load_hinglish_lexicon,
    detect_hinglish_abuse,
)

__all__ = [
    # Part 1: Data Loading & Labeling
    "load_data",
    "get_default_dataset_path",
    "REQUIRED_COLUMNS",
    "TOXICITY_COLUMNS",
    "validate_labels",
    "handle_missing_comments",
    "remove_duplicates",
    "create_abusive_label",
    "prepare_dataset",
    # Part 2: Text Normalization & Feature Extraction
    "preprocess_text",
    "preprocess_text_series",
    "normalize_obfuscation_text",
    "create_tfidf_vectorizer",
    "create_feature_union_vectorizer",
    "fit_tfidf_vectorizer",
    "transform_tfidf",
    "fit_transform_tfidf",
    "get_feature_names",
    # Part 3: Model Training & Evaluation
    "split_data",
    "train_logistic_regression",
    "train_naive_bayes",
    "evaluate_model",
    "compare_and_select_best_model",
    "plot_confusion_matrix",
    "save_model_artifacts",
    "load_model_artifacts",
    "run_training_pipeline",
    "get_default_models_dir",
    # Part 4: Live Prediction & Inference
    "get_prediction_pipeline",
    "predict_comment",
    "load_hinglish_lexicon",
    "detect_hinglish_abuse",
]
