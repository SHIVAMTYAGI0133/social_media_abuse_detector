"""
test_model_training.py - Unit tests for Part 3 Model Training and Evaluation.

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
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extraction import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
    transform_tfidf,
)
from src.model_training import (
    split_data,
    train_logistic_regression,
    train_naive_bayes,
    evaluate_model,
    compare_and_select_best_model,
    save_model_artifacts,
    load_model_artifacts,
    run_training_pipeline,
)


@pytest.fixture
def synthetic_training_df() -> pd.DataFrame:
    """
    Creates a synthetic toy dataset for unit testing training and evaluation.
    Contains 20 balanced/imbalanced sample comments with known binary labels.
    CLEARLY LABELED: Synthetic test fixture data, NOT real data.
    """
    safe_comments = [
        f"Thank you for this very helpful Wikipedia contribution #{i}" for i in range(14)
    ]
    abusive_comments = [
        "You are an absolute idiot and a complete loser.",
        "I will find you and hurt you for reverting this.",
        "This is disgusting and hateful trash.",
        "You are so stupid and incompetent.",
        "Shut your mouth and get lost.",
        "I hate you and your family.",
    ]

    all_comments = safe_comments + abusive_comments
    all_labels = [0] * len(safe_comments) + [1] * len(abusive_comments)

    return pd.DataFrame({
        "id": [f"syn_{i}" for i in range(len(all_comments))],
        "comment_text": all_comments,
        "toxic": [0] * 14 + [1, 1, 1, 1, 1, 1],
        "severe_toxic": [0] * 20,
        "obscene": [0] * 14 + [1, 0, 1, 0, 0, 0],
        "threat": [0] * 14 + [0, 1, 0, 0, 0, 0],
        "insult": [0] * 14 + [1, 0, 0, 1, 1, 0],
        "identity_hate": [0] * 14 + [0, 0, 0, 0, 0, 1],
    })


@pytest.fixture
def synthetic_csv_file(synthetic_training_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Writes synthetic dataset to a temporary CSV file."""
    temp_csv = tmp_path / "synthetic_train.csv"
    synthetic_training_df.to_csv(temp_csv, index=False)
    return temp_csv


def test_split_data_reproducibility():
    """Verifies that split_data produces identical splits when given random_state=42."""
    texts = [f"Comment number {i}" for i in range(20)]
    labels = [0] * 15 + [1] * 5

    X_train_1, X_test_1, y_train_1, y_test_1 = split_data(
        texts, labels, test_size=0.25, random_state=42
    )
    X_train_2, X_test_2, y_train_2, y_test_2 = split_data(
        texts, labels, test_size=0.25, random_state=42
    )

    pd.testing.assert_series_equal(X_train_1, X_train_2)
    pd.testing.assert_series_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)


def test_split_data_stratification():
    """Verifies that stratified splitting preserves the safe/abusive label proportions."""
    texts = [f"Comment {i}" for i in range(20)]
    labels = [0] * 16 + [1] * 4  # 80% safe (0), 20% abusive (1)

    X_train, X_test, y_train, y_test = split_data(
        texts, labels, test_size=0.25, random_state=42, stratify=True
    )

    # Total test count = 5
    assert len(X_test) == 5
    assert len(y_test) == 5

    # Check that abusive label (1) exists in both splits
    assert 1 in set(y_train)
    assert 1 in set(y_test)
    assert 0 in set(y_train)
    assert 0 in set(y_test)


def test_split_data_mismatched_length():
    """Verifies that passing mismatched X and y lengths raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        split_data(["Text 1", "Text 2"], [1])
    assert "Features length (2) != labels length (1)" in str(exc_info.value)


def test_no_data_leakage_vectorizer_scope():
    """
    CRITICAL CHECK: Verifies that words exclusive to the test set NEVER enter the
    vectorizer vocabulary when fitting strictly on training data.
    """
    train_texts = ["safe comment here", "helpful and clean message"]
    test_texts = ["unseen_exclusive_word_xyz safe comment"]

    vec = create_tfidf_vectorizer(min_df=1)
    fitted_vec = fit_tfidf_vectorizer(vec, train_texts)

    # Word 'unseen_exclusive_word_xyz' MUST NOT be in vocabulary
    assert "unseen_exclusive_word_xyz" not in fitted_vec.vocabulary_

    # Transforming test texts should work using only known vocabulary
    test_vec = transform_tfidf(fitted_vec, test_texts)
    assert test_vec.shape[0] == 1
    assert test_vec.shape[1] == len(fitted_vec.vocabulary_)


def test_train_and_evaluate_logistic_regression():
    """Verifies that Logistic Regression trains and evaluates properly on sparse data."""
    train_texts = ["clean text one", "clean text two", "idiot jerk", "rude loser"]
    y_train = pd.Series([0, 0, 1, 1])

    test_texts = ["clean text", "jerk"]
    y_test = pd.Series([0, 1])

    vec = create_tfidf_vectorizer(min_df=1)
    fitted_vec = fit_tfidf_vectorizer(vec, train_texts)
    X_train = transform_tfidf(fitted_vec, train_texts)
    X_test = transform_tfidf(fitted_vec, test_texts)

    model = train_logistic_regression(X_train, y_train, random_state=42)
    assert isinstance(model, LogisticRegression)

    eval_result = evaluate_model(model, X_test, y_test, model_name="LogReg")
    assert "accuracy" in eval_result
    assert "abusive_precision" in eval_result
    assert "abusive_recall" in eval_result
    assert "abusive_f1" in eval_result
    assert len(eval_result["confusion_matrix"]) == 2


def test_train_and_evaluate_naive_bayes():
    """Verifies that Multinomial Naive Bayes trains and evaluates properly."""
    train_texts = ["good job", "nice edit", "terrible idiot", "bad jerk"]
    y_train = pd.Series([0, 0, 1, 1])

    test_texts = ["good edit", "idiot"]
    y_test = pd.Series([0, 1])

    vec = create_tfidf_vectorizer(min_df=1)
    fitted_vec = fit_tfidf_vectorizer(vec, train_texts)
    X_train = transform_tfidf(fitted_vec, train_texts)
    X_test = transform_tfidf(fitted_vec, test_texts)

    model = train_naive_bayes(X_train, y_train, alpha=0.5)
    assert isinstance(model, MultinomialNB)

    eval_result = evaluate_model(model, X_test, y_test, model_name="NaiveBayes")
    assert eval_result["model_name"] == "NaiveBayes"
    assert 0.0 <= eval_result["accuracy"] <= 1.0


def test_compare_and_select_best_model():
    """Verifies that compare_and_select_best_model chooses the candidate with higher abusive_f1."""
    results = {
        "Model_Low_Recall": {
            "model_object": "DummyModelA",
            "accuracy": 0.95,
            "abusive_precision": 0.90,
            "abusive_recall": 0.50,
            "abusive_f1": 0.64,
        },
        "Model_High_Recall": {
            "model_object": "DummyModelB",
            "accuracy": 0.91,
            "abusive_precision": 0.85,
            "abusive_recall": 0.88,
            "abusive_f1": 0.86,
        },
    }

    best_name, best_model, best_metrics = compare_and_select_best_model(
        results, primary_metric="abusive_f1"
    )
    assert best_name == "Model_High_Recall"
    assert best_model == "DummyModelB"
    assert best_metrics["abusive_f1"] == 0.86


def test_save_and_load_model_artifacts(tmp_path: Path):
    """Verifies saving and loading model artifacts using joblib."""
    # Create simple dummy model and vectorizer
    vec = create_tfidf_vectorizer(min_df=1)
    fitted_vec = fit_tfidf_vectorizer(vec, ["hello world", "helpful comment"])
    X = transform_tfidf(fitted_vec, ["hello world", "helpful comment"])
    y = pd.Series([0, 1])
    model = train_logistic_regression(X, y)

    dummy_results = {
        "Logistic Regression": {
            "model_object": model,
            "accuracy": 1.0,
            "abusive_precision": 1.0,
            "abusive_recall": 1.0,
            "abusive_f1": 1.0,
            "confusion_matrix": [[1, 0], [0, 1]],
        }
    }

    saved_paths = save_model_artifacts(
        best_model=model,
        vectorizer=fitted_vec,
        evaluation_results=dummy_results,
        best_model_name="Logistic Regression",
        output_dir=tmp_path,
    )

    assert saved_paths["model_path"].is_file()
    assert saved_paths["vectorizer_path"].is_file()
    assert saved_paths["metadata_path"].is_file()
    assert saved_paths["cm_plot_path"].is_file()

    # Load artifacts back and verify
    loaded_model, loaded_vec = load_model_artifacts(models_dir=tmp_path)
    assert hasattr(loaded_model, "predict")
    assert hasattr(loaded_vec, "transform")
    assert loaded_vec.vocabulary_ == fitted_vec.vocabulary_


def test_load_model_artifacts_missing_file(tmp_path: Path):
    """Verifies that load_model_artifacts raises FileNotFoundError when files do not exist."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_model_artifacts(models_dir=tmp_path)
    assert "Trained model file not found" in str(exc_info.value)


def test_run_training_pipeline_end_to_end_on_synthetic_csv(
    synthetic_csv_file: Path, tmp_path: Path
):
    """
    End-to-end integration test verifying that the complete training pipeline runs
    from synthetic CSV to model persistence without errors.
    """
    models_out = tmp_path / "models_output"

    run_summary = run_training_pipeline(
        dataset_path=synthetic_csv_file,
        models_dir=models_out,
        max_features=50,
        test_size=0.25,
        random_state=42,
    )

    assert run_summary["best_model_name"] in [
        "Logistic Regression",
        "Multinomial Naive Bayes",
    ]
    assert run_summary["train_samples"] == 15
    assert run_summary["test_samples"] == 5
    assert (models_out / "best_model.joblib").is_file()
    assert (models_out / "tfidf_vectorizer.joblib").is_file()
    assert (models_out / "evaluation_summary.json").is_file()
    assert (models_out / "confusion_matrix.png").is_file()


def test_run_training_pipeline_missing_dataset_raises_error():
    """Verifies that attempting to run training on a non-existent CSV raises FileNotFoundError."""
    fake_path = Path("non_existent_folder_xyz") / "train.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        run_training_pipeline(dataset_path=fake_path)
    assert "Dataset file not found" in str(exc_info.value)
