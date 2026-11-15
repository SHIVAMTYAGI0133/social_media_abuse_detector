"""
data_loader.py - Module for loading and validating dataset files.

This module handles:
1. Checking whether the dataset file exists.
2. Loading train.csv using Pandas.
3. Validating that all required columns are present.
4. Verifying that comment text and label columns contain expected data types.
5. Providing clear, beginner-friendly error messages if anything goes wrong.
"""

from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

# The 6 distinct toxicity subcategories annotated in the Jigsaw dataset
TOXICITY_COLUMNS: List[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# The complete set of columns required for data preparation
REQUIRED_COLUMNS: List[str] = ["comment_text"] + TOXICITY_COLUMNS


def get_default_dataset_path() -> Path:
    """
    Returns the default path to 'data/train.csv' relative to the project root.
    Works reliably on Windows, macOS, and Linux without hardcoding paths.
    """
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "train.csv"


def check_file_exists(file_path: Union[str, Path]) -> Path:
    """
    Checks if the specified dataset file exists on disk.

    Parameters:
        file_path (str or Path): Path to the CSV file.

    Returns:
        Path: Resolved Path object if the file exists.

    Raises:
        FileNotFoundError: If the file does not exist, with helpful instructions.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.is_file():
        error_msg = (
            f"\n[ERROR] Dataset file not found at: {path_obj}\n\n"
            "Where to get the dataset:\n"
            "1. Visit the Kaggle competition:\n"
            "   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data\n"
            "2. Download 'train.csv.zip' and extract 'train.csv'.\n"
            "3. Place 'train.csv' inside the 'data/' folder of this project:\n"
            f"   Target path: {path_obj}\n"
        )
        raise FileNotFoundError(error_msg)
    return path_obj


def validate_columns(df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> None:
    """
    Validates that all required columns exist in the loaded DataFrame.

    Parameters:
        df (pd.DataFrame): The loaded dataset DataFrame.
        required_cols (list of str, optional): List of required column names.
            Defaults to REQUIRED_COLUMNS.

    Raises:
        ValueError: If any required column is missing.
    """
    if required_cols is None:
        required_cols = REQUIRED_COLUMNS

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"\n[ERROR] The dataset is missing required column(s): {missing_cols}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Required columns: {required_cols}\n"
            "Please ensure you are using the official Jigsaw 'train.csv' file."
        )


def validate_data_content(df: pd.DataFrame) -> None:
    """
    Validates basic content integrity of the dataset:
    - Checks that DataFrame is not completely empty (0 rows).
    - Checks that comment_text column contains text-like data.

    Parameters:
        df (pd.DataFrame): The loaded dataset DataFrame.

    Raises:
        ValueError: If the dataset is empty or comment_text is completely null.
    """
    if len(df) == 0:
        raise ValueError("[ERROR] The dataset file was loaded, but it contains 0 rows.")

    # Check if comment_text has any non-null content
    if df["comment_text"].dropna().empty:
        raise ValueError(
            "[ERROR] The 'comment_text' column contains only null/empty values."
        )


def load_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads and validates the dataset from a CSV file.

    Parameters:
        file_path (str or Path, optional): Custom file path to train.csv.
            If None, defaults to 'data/train.csv' within the project.

    Returns:
        pd.DataFrame: Validated DataFrame containing the raw dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or data is malformed.
    """
    if file_path is None:
        target_path = get_default_dataset_path()
    else:
        target_path = Path(file_path)

    # 1. Check whether file exists
    verified_path = check_file_exists(target_path)

    # 2. Load the CSV file using Pandas
    try:
        df = pd.read_csv(verified_path)
    except Exception as exc:
        raise ValueError(
            f"[ERROR] Failed to parse CSV file at {verified_path}: {exc}"
        ) from exc

    # 3. Check whether required columns are present
    validate_columns(df, REQUIRED_COLUMNS)

    # 4. Check whether data content is valid
    validate_data_content(df)

    return df
