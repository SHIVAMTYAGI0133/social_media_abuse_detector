"""
model_training.py - Model Training, Evaluation, and Persistence Module.

This module is responsible for:
1. Loading and splitting the dataset into train and test sets using reproducible stratification.
2. Extracting TF-IDF features strictly on the training set to prevent data leakage.
3. Training two baseline classifiers: Logistic Regression (balanced) and Multinomial Naive Bayes.
4. Accounting for class imbalance and evaluating models on the held-out test set.
5. Computing comprehensive metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
6. Selecting the best model based on Abusive-Class F1-Score and Recall.
7. Persisting the winning model, fitted TF-IDF vectorizer, and evaluation artifacts to disk.

=============================================================================
PREVENTING DATA LEAKAGE (Key Concept for Viva / Exam)
-----------------------------------------------------------------------------
Data leakage occurs when information from outside the training dataset (such as
the test dataset) is accidentally used to create or tune the model.
In NLP, fitting TF-IDF on the entire dataset before splitting is a common mistake:
it allows the vectorizer to "peek" at the vocabulary and IDF weights of test comments.

To strictly avoid data leakage:
1. We split raw comments into Train and Test sets FIRST.
2. We fit the TfidfVectorizer ONLY on the Training comments.
3. We transform the Test comments using the already-fitted vectorizer.
=============================================================================
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .data_loader import get_default_dataset_path, load_data
from .feature_extraction import (
    create_tfidf_vectorizer,
    fit_tfidf_vectorizer,
    transform_tfidf,
)
from .preprocessing import prepare_dataset
from .text_preprocessing import preprocess_text_series


def get_default_models_dir() -> Path:
    """Returns the default directory for saving models relative to project root."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "models"


def split_data(
    X: Union[pd.Series, List[str]],
    y: Union[pd.Series, List[int], np.ndarray],
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits text comments and binary labels into reproducible training and testing sets.

    Parameters:
        X (pd.Series or list): Cleaned text comments.
        y (pd.Series or list): Binary abusive target labels (0 = Safe, 1 = Abusive).
        test_size (float): Proportion of data reserved for testing (default: 0.2, i.e., 20%).
        random_state (int): Seed for reproducible random splitting (default: 42).
        stratify (bool): If True, preserves the exact percentage of safe vs abusive
            comments across both train and test splits (essential for imbalanced data).

    Returns:
        tuple (X_train, X_test, y_train, y_test): Four Pandas Series.

    Raises:
        ValueError: If input lengths do not match or collection is empty.
    """
    if len(X) == 0 or len(y) == 0:
        raise ValueError("[ERROR] Cannot split empty data or labels.")

    if len(X) != len(y):
        raise ValueError(f"[ERROR] Features length ({len(X)}) != labels length ({len(y)}).")

    # Ensure inputs are Pandas Series
    X_series = pd.Series(X).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)

    # Check that both classes exist if stratify is requested
    unique_classes = y_series.unique()
    if stratify and len(unique_classes) > 1:
        stratify_labels = y_series
    else:
        stratify_labels = None

    X_train, X_test, y_train, y_test = train_test_split(
        X_series,
        y_series,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def train_logistic_regression(
    X_train: csr_matrix,
    y_train: pd.Series,
    class_weight: Optional[str] = "balanced",
    max_iter: int = 1000,
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    """
    Trains a Logistic Regression classifier for binary abuse detection.

    Why Logistic Regression?
    - Highly effective for high-dimensional sparse TF-IDF text features.
    - Fast training and low memory footprint.
    - With class_weight='balanced', it automatically assigns higher loss penalties
      to the minority 'abusive' class, preventing the model from ignoring rare abuse.

    Parameters:
        X_train (csr_matrix): Sparse TF-IDF feature matrix of training comments.
        y_train (pd.Series): Binary labels for training comments.
        class_weight (str or dict, optional): 'balanced' handles class imbalance.
        max_iter (int): Maximum solver iterations to ensure convergence.
        random_state (int): Reproducibility seed.

    Returns:
        LogisticRegression: The trained model.
    """
    if X_train.shape[0] != len(y_train):
        raise ValueError(
            f"[ERROR] Sample count mismatch: X_train has {X_train.shape[0]} rows, "
            f"y_train has {len(y_train)} labels."
        )

    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs,
    )
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(
    X_train: csr_matrix,
    y_train: pd.Series,
    alpha: float = 0.5,
    **kwargs: Any,
) -> MultinomialNB:
    """
    Trains a Multinomial Naive Bayes classifier for binary abuse detection.

    Why Multinomial Naive Bayes?
    - The classic probabilistic baseline for text classification.
    - Assumes conditional independence between features given the class label.
    - Extremely fast to train and evaluate.

    Parameters:
        X_train (csr_matrix): Sparse TF-IDF feature matrix of training comments.
        y_train (pd.Series): Binary labels for training comments.
        alpha (float): Laplace smoothing parameter (default 0.5 to prevent zero probabilities).

    Returns:
        MultinomialNB: The trained model.
    """
    if X_train.shape[0] != len(y_train):
        raise ValueError(
            f"[ERROR] Sample count mismatch: X_train has {X_train.shape[0]} rows, "
            f"y_train has {len(y_train)} labels."
        )

    model = MultinomialNB(alpha=alpha, **kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: csr_matrix,
    y_test: pd.Series,
    model_name: str = "Classifier",
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on the held-out test set.

    Positive Class (1): Abusive
    Negative Class (0): Safe

    Metrics Explained:
    - Accuracy: Total correct predictions over total comments (can be misleading under class imbalance).
    - Precision: When the model predicts 'abusive', how often is it right?
    - Recall: What fraction of all actual 'abusive' comments did the model successfully catch?
    - F1-Score: Harmonic mean of precision and recall.

    Parameters:
        model: Trained Scikit-learn classifier.
        X_test (csr_matrix): Test feature matrix.
        y_test (pd.Series): Ground-truth test labels.
        model_name (str): Label for display and reporting.

    Returns:
        dict: Complete evaluation metrics dictionary.
    """
    y_pred = model.predict(X_test)

    # Core scalar metrics (targeting the abusive class = 1)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    # Detailed text classification report
    report_dict = classification_report(
        y_test, y_pred, target_names=["Safe (0)", "Abusive (1)"], output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_test, y_pred, target_names=["Safe (0)", "Abusive (1)"], zero_division=0
    )

    return {
        "model_name": model_name,
        "model_object": model,
        "accuracy": acc,
        "abusive_precision": prec,
        "abusive_recall": rec,
        "abusive_f1": f1,
        "confusion_matrix": cm.tolist(),
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
    }


def compare_and_select_best_model(
    results: Dict[str, Dict[str, Any]],
    primary_metric: str = "abusive_f1",
) -> Tuple[str, Any, Dict[str, Any]]:
    """
    Compares evaluated models and selects the top-performing candidate.

    Evaluation Criterion:
    We prioritize 'abusive_f1' (or abusive recall) over overall accuracy.
    In social media moderation, missing abusive content (false negatives) is
    dangerous, but spamming safe comments with false alarms (false positives) is
    also undesirable. The F1-score balances both trade-offs.

    Parameters:
        results (dict): Dictionary mapping model names to their evaluation dicts.
        primary_metric (str): Key to rank models by (default: 'abusive_f1').

    Returns:
        tuple (best_model_name, best_model_object, best_metrics_dict).
    """
    if not results:
        raise ValueError("[ERROR] No model evaluation results provided for comparison.")

    best_name = None
    best_score = -1.0
    best_metrics = None

    for name, metrics in results.items():
        score = metrics.get(primary_metric, 0.0)
        # Tie-breaker: if F1 is equal, choose higher recall
        if (score > best_score) or (
            score == best_score
            and metrics.get("abusive_recall", 0.0) > best_metrics.get("abusive_recall", 0.0)
        ):
            best_score = score
            best_name = name
            best_metrics = metrics

    best_model = best_metrics["model_object"]
    return best_name, best_model, best_metrics


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """
    Generates and saves an annotated heatmap confusion matrix plot.

    Parameters:
        cm (np.ndarray): 2x2 confusion matrix array [[TN, FP], [FN, TP]].
        model_name (str): Name of the model.
        output_path (Path): Path where the PNG image will be saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    classes = ["Safe (0)", "Abusive (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")

    # Annotate cells with counts and descriptions
    labels_matrix = [
        [f"True Neg (Safe)\n{cm[0, 0]:,}", f"False Pos\n{cm[0, 1]:,}"],
        [f"False Neg\n{cm[1, 0]:,}", f"True Pos (Abusive)\n{cm[1, 1]:,}"],
    ]

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                labels_matrix[i][j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {output_path}")


def save_model_artifacts(
    best_model: Any,
    vectorizer: Any,
    evaluation_results: Dict[str, Dict[str, Any]],
    best_model_name: str,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """
    Persists the winning model, fitted TF-IDF vectorizer, and evaluation reports using joblib.

    Parameters:
        best_model: Trained winning classifier.
        vectorizer: Fitted TfidfVectorizer.
        evaluation_results (dict): Dictionary of evaluation metrics for all models.
        best_model_name (str): Name of winning model.
        output_dir (Path, optional): Directory to save artifacts. Defaults to 'models/'.

    Returns:
        dict: Mapping of artifact names to their saved file paths.
    """
    if output_dir is None:
        target_dir = get_default_models_dir()
    else:
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save trained model
    model_path = target_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    # 2. Save fitted vectorizer
    vectorizer_path = target_dir / "tfidf_vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)

    # 3. Save evaluation summary as JSON (sanitize by removing model_object)
    summary_data = {}
    for name, res in evaluation_results.items():
        clean_entry = {k: v for k, v in res.items() if k != "model_object"}
        summary_data[name] = clean_entry

    metadata_path = target_dir / "evaluation_summary.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_best_model": best_model_name,
                "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "results": summary_data,
            },
            f,
            indent=2,
        )

    # 4. Generate and save confusion matrix chart for best model
    best_cm = np.array(evaluation_results[best_model_name]["confusion_matrix"])
    cm_plot_path = target_dir / "confusion_matrix.png"
    plot_confusion_matrix(best_cm, best_model_name, cm_plot_path)

    print(f"[INFO] Saved trained model to:      {model_path}")
    print(f"[INFO] Saved TF-IDF vectorizer to: {vectorizer_path}")
    print(f"[INFO] Saved evaluation summary to: {metadata_path}")

    return {
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "metadata_path": metadata_path,
        "cm_plot_path": cm_plot_path,
    }


def load_model_artifacts(
    models_dir: Optional[Union[str, Path]] = None,
) -> Tuple[Any, Any]:
    """
    Loads the trained model and fitted TF-IDF vectorizer from disk.

    Parameters:
        models_dir (Path, optional): Directory containing saved artifacts.

    Returns:
        tuple (model, vectorizer).

    Raises:
        FileNotFoundError: If the model or vectorizer files are missing.
    """
    if models_dir is None:
        target_dir = get_default_models_dir()
    else:
        target_dir = Path(models_dir)

    model_path = target_dir / "best_model.joblib"
    vec_path = target_dir / "tfidf_vectorizer.joblib"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"[ERROR] Trained model file not found at: {model_path}. "
            "Please run 'python -m src.model_training' first to train and save the model."
        )

    if not vec_path.is_file():
        raise FileNotFoundError(
            f"[ERROR] TF-IDF vectorizer file not found at: {vec_path}. "
            "Please run 'python -m src.model_training' first to fit and save the vectorizer."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    return model, vectorizer


def run_training_pipeline(
    dataset_path: Optional[Union[str, Path]] = None,
    models_dir: Optional[Union[str, Path]] = None,
    max_features: int = 10000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Orchestrates the complete Part 3 Model Training & Evaluation pipeline.

    Steps:
    1. Loads and prepares the dataset using Part 1 modules.
    2. Cleans text using Part 2 preprocessing.
    3. Splits data into Train and Test sets using stratified sampling.
    4. Fits TF-IDF vectorizer strictly on training text; transforms test text.
    5. Trains Logistic Regression (balanced) and Multinomial Naive Bayes.
    6. Evaluates both models on held-out test data.
    7. Selects the best model prioritizing Abusive-class F1-Score and Recall.
    8. Persists the best model, vectorizer, confusion matrix plot, and summary.

    Parameters:
        dataset_path (Path, optional): Custom path to train.csv.
        models_dir (Path, optional): Custom directory to save models.
        max_features (int): Maximum vocabulary size for TF-IDF.
        test_size (float): Proportion of data for test set.
        random_state (int): Reproducibility seed.

    Returns:
        dict: Training run summary including best model name and metrics.
    """
    print("\n" + "=" * 70)
    print("PART 3: MODEL TRAINING & EVALUATION PIPELINE".center(70))
    print("Project: Detection of Violent and Abusive Content in Social Media".center(70))
    print("=" * 70)

    # Step 1: Load and prepare dataset
    if dataset_path is None:
        target_path = get_default_dataset_path()
    else:
        target_path = Path(dataset_path)

    print(f"\n[STEP 1] Loading raw dataset from: {target_path}")
    raw_df = load_data(target_path)
    print(f"[OK] Raw dataset loaded: {len(raw_df):,} comments.")

    print("\n[STEP 2] Running Part 1 data cleaning and 'abusive' label preparation...")
    prepared_df = prepare_dataset(raw_df)
    print(f"[OK] Prepared dataset: {len(prepared_df):,} clean comments.")

    # Class balance audit
    counts = prepared_df["abusive"].value_counts()
    safe_cnt = counts.get(0, 0)
    abusive_cnt = counts.get(1, 0)
    total_cnt = len(prepared_df)
    print(f"\nClass Distribution:")
    print(f"  - Safe comments (0):    {safe_cnt:,} ({safe_cnt / total_cnt * 100:.2f}%)")
    print(f"  - Abusive comments (1): {abusive_cnt:,} ({abusive_cnt / total_cnt * 100:.2f}%)")
    print("Note: The abusive class is the minority class. High accuracy alone is insufficient;")
    print("we must evaluate Abusive Recall and F1-Score.")

    # Step 3: Text Preprocessing
    print("\n[STEP 3] Normalizing comments with Part 2 text preprocessing...")
    clean_texts = preprocess_text_series(prepared_df["comment_text"])
    labels = prepared_df["abusive"]
    print("[OK] Text normalization complete.")

    # Step 4: Stratified Train/Test Split (BEFORE any TF-IDF fitting!)
    print(
        f"\n[STEP 4] Splitting data into Train ({int((1-test_size)*100)}%) and Test ({int(test_size*100)}%)..."
    )
    print(f"Stratification: Preserving safe/abusive ratio (random_state={random_state}).")
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        clean_texts, labels, test_size=test_size, random_state=random_state, stratify=True
    )
    print(f"  - Training samples: {len(X_train_raw):,}")
    print(f"  - Testing samples:  {len(X_test_raw):,}")

    # Step 5: TF-IDF Feature Extraction (Train fit only!)
    print("\n[STEP 5] Extracting TF-IDF features (strictly avoiding data leakage)...")
    print("Fitting TfidfVectorizer on training comments ONLY...")
    vectorizer = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
    )
    vectorizer = fit_tfidf_vectorizer(vectorizer, X_train_raw)
    vocab_size = len(vectorizer.vocabulary_)
    print(f"[OK] Learned vocabulary size: {vocab_size:,} unigram & bigram features.")

    print("Transforming training and test text into sparse matrices...")
    X_train_vec = transform_tfidf(vectorizer, X_train_raw)
    X_test_vec = transform_tfidf(vectorizer, X_test_raw)
    print(f"  - X_train sparse matrix: {X_train_vec.shape}")
    print(f"  - X_test sparse matrix:  {X_test_vec.shape}")

    # Step 6: Train Baseline Classifiers
    print("\n[STEP 6] Training Baseline Classifiers...")

    # Model A: Logistic Regression (balanced)
    print("  -> Training Logistic Regression (class_weight='balanced', max_iter=1000)...")
    log_reg = train_logistic_regression(X_train_vec, y_train, class_weight="balanced")
    print("     [OK] Logistic Regression trained.")

    # Model B: Multinomial Naive Bayes
    print("  -> Training Multinomial Naive Bayes (alpha=0.5)...")
    nb_model = train_naive_bayes(X_train_vec, y_train, alpha=0.5)
    print("     [OK] Multinomial Naive Bayes trained.")

    # Step 7: Evaluate on Held-Out Test Set
    print("\n[STEP 7] Evaluating Classifiers on Held-Out Test Data...")
    results = {}

    log_reg_eval = evaluate_model(log_reg, X_test_vec, y_test, model_name="Logistic Regression")
    results["Logistic Regression"] = log_reg_eval

    nb_eval = evaluate_model(nb_model, X_test_vec, y_test, model_name="Multinomial Naive Bayes")
    results["Multinomial Naive Bayes"] = nb_eval

    # Display comparison table
    print("\n" + "=" * 70)
    print("MODEL PERFORMANCE COMPARISON (Held-Out Test Set)".center(70))
    print("=" * 70)
    print(
        f"{'Model':<26} | {'Accuracy':<10} | {'Abusive Prec':<12} | {'Abusive Rec':<11} | {'Abusive F1':<10}"
    )
    print("-" * 75)
    for name, r in results.items():
        print(
            f"{name:<26} | {r['accuracy']*100:>8.2f}% | "
            f"{r['abusive_precision']*100:>10.2f}% | "
            f"{r['abusive_recall']*100:>9.2f}% | "
            f"{r['abusive_f1']*100:>8.2f}%"
        )
    print("=" * 70)

    # Print detailed classification reports
    for name, r in results.items():
        print(f"\n--- Detailed Classification Report: {name} ---")
        print(r["classification_report_text"])
        print(f"Confusion Matrix (labels [Safe, Abusive]):")
        print(f"  [TN={r['true_negatives']:,}, FP={r['false_positives']:,}]")
        print(f"  [FN={r['false_negatives']:,}, TP={r['true_positives']:,}]")

    # Step 8: Select Best Model
    best_name, best_model, best_eval = compare_and_select_best_model(
        results, primary_metric="abusive_f1"
    )
    print("\n[STEP 8] Model Selection Decision:")
    print(f"  >> Selected Champion Model: {best_name}")
    print(
        f"  >> Selection Rationale: Achieved highest Abusive F1-Score ({best_eval['abusive_f1']*100:.2f}%) "
        f"and Abusive Recall ({best_eval['abusive_recall']*100:.2f}%)."
    )

    # Step 9: Save Artifacts
    print("\n[STEP 9] Persisting Model & Evaluation Artifacts...")
    saved_paths = save_model_artifacts(
        best_model=best_model,
        vectorizer=vectorizer,
        evaluation_results=results,
        best_model_name=best_name,
        output_dir=models_dir,
    )

    print("\n" + "=" * 70)
    print("PART 3 MODEL TRAINING COMPLETE".center(70))
    print("=" * 70)

    return {
        "best_model_name": best_name,
        "best_model_object": best_model,
        "best_metrics": best_eval,
        "all_results": results,
        "saved_paths": saved_paths,
        "train_samples": len(X_train_raw),
        "test_samples": len(X_test_raw),
        "vocab_size": vocab_size,
    }


if __name__ == "__main__":
    # Command-line entry point: supports custom dataset path as optional argument
    cli_dataset = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None

    try:
        run_training_pipeline(dataset_path=cli_dataset)
    except FileNotFoundError as fnf_err:
        print("\n" + "=" * 70)
        print("DATASET NOT FOUND".center(70))
        print("=" * 70)
        print(str(fnf_err))
        print("ACTION REQUIRED:")
        print("1. Download 'train.csv' from Kaggle (Jigsaw Toxic Comment Challenge).")
        print("2. Place it inside the 'data/' folder: data/train.csv")
        print("3. Run 'python -m src.model_training' again.")
        sys.exit(1)
    except Exception as err:
        print(f"\n[CRITICAL ERROR] Model training pipeline failed: {err}")
        sys.exit(1)
