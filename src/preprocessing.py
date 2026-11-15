"""
preprocessing.py - Data cleaning, label validation, and label preparation module.

This module is responsible for:
1. Handling missing comment text safely and transparently.
2. Removing duplicate comments with documented rationale.
3. Strictly validating that toxicity labels are binary (0 or 1) without silent coercion.
4. Creating a unified binary 'abusive' column (1 if any toxicity subcategory is 1, else 0).
5. Preserving all original toxicity columns for downstream analysis.
"""

from typing import List, Optional
import pandas as pd
from .data_loader import TOXICITY_COLUMNS


def validate_labels(df: pd.DataFrame, toxicity_cols: Optional[List[str]] = None) -> None:
    """
    Validates that the toxicity label columns contain strictly valid binary values (0 or 1).

    Parameters:
        df (pd.DataFrame): The DataFrame containing toxicity columns.
        toxicity_cols (list of str, optional): List of label columns to validate.
            Defaults to TOXICITY_COLUMNS.

    Raises:
        ValueError: If any label column has missing (NaN) values or non-binary values.
                    We explicitly avoid silent conversion (e.g. converting negative
                    numbers or arbitrary floats to 0 or 1).
    """
    if toxicity_cols is None:
        toxicity_cols = TOXICITY_COLUMNS

    for col in toxicity_cols:
        if col not in df.columns:
            raise ValueError(f"[ERROR] Toxicity column '{col}' is missing from the DataFrame.")

        # Check for missing/null values in labels
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise ValueError(
                f"[ERROR] Label column '{col}' contains {null_count} null/missing value(s). "
                "Toxicity labels must be explicitly 0 or 1."
            )

        # Check for valid binary values (0 or 1)
        unique_vals = set(df[col].unique())
        allowed_vals = {0, 1}

        # Allow numeric types like int or float where value equals 0 or 1 (e.g., 0.0, 1.0)
        invalid_vals = {val for val in unique_vals if val not in allowed_vals}
        if invalid_vals:
            raise ValueError(
                f"[ERROR] Label column '{col}' contains invalid non-binary value(s): {invalid_vals}. "
                "Toxicity labels must contain ONLY 0 or 1. Silent conversion is prohibited."
            )


def handle_missing_comments(
    df: pd.DataFrame, text_col: str = "comment_text", drop_missing: bool = True
) -> pd.DataFrame:
    """
    Handles missing, null, or blank comment text in the dataset.

    Design Decision Documentation:
    ------------------------------
    In Natural Language Processing (NLP), comments with null or empty text carry zero
    linguistic features. Imputing missing comments with artificial strings (like 'missing'
    or empty strings) introduces synthetic noise that can mislead text vectorizers (e.g.,
    TF-IDF) and cause false predictions.
    
    Therefore, the safest and cleanest strategy is to drop rows where the comment text is
    missing or contains only whitespace.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        text_col (str): Name of the text column (default: 'comment_text').
        drop_missing (bool): If True, drops rows with null/blank comments.
                             If False, fills NaN with an empty string.

    Returns:
        pd.DataFrame: Cleaned DataFrame with missing text handled.
    """
    df_cleaned = df.copy()

    if text_col not in df_cleaned.columns:
        raise ValueError(f"[ERROR] Text column '{text_col}' not found in DataFrame.")

    # Identify rows where text is null or empty whitespace
    is_null = df_cleaned[text_col].isnull()
    is_blank = df_cleaned[text_col].astype(str).str.strip().eq("")
    invalid_mask = is_null | is_blank

    dropped_count = invalid_mask.sum()

    if drop_missing:
        if dropped_count > 0:
            print(f"[INFO] Dropping {dropped_count} row(s) with missing/blank '{text_col}'.")
            df_cleaned = df_cleaned[~invalid_mask].reset_index(drop=True)
    else:
        if dropped_count > 0:
            print(f"[INFO] Filling {dropped_count} missing '{text_col}' row(s) with empty string.")
            df_cleaned[text_col] = df_cleaned[text_col].fillna("").astype(str)

    # Ensure comment_text is explicitly typed as string
    df_cleaned[text_col] = df_cleaned[text_col].astype(str)
    return df_cleaned


def remove_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Removes duplicate rows from the dataset.

    Design Decision Documentation:
    ------------------------------
    In social media datasets, repeated comments frequently arise from bot spam,
    viral copypastas, or duplicate scraping. If duplicate comments remain in the
    dataset, they can:
      1. Cause data leakage between train/test splits.
      2. Artificially inflate model accuracy metrics.
      3. Over-fit the model to specific recurring phrases.
    
    Therefore, we remove duplicate comments based on the comment text (keeping the
    first occurrence), preserving a diverse and representative dataset.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        subset (list of str, optional): Columns to check for duplicates.
            Defaults to ['comment_text'].

    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    df_cleaned = df.copy()

    if subset is None:
        subset = ["comment_text"]

    initial_count = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    duplicates_removed = initial_count - len(df_cleaned)

    if duplicates_removed > 0:
        print(f"[INFO] Removed {duplicates_removed} duplicate row(s) based on columns: {subset}.")

    return df_cleaned


def create_abusive_label(
    df: pd.DataFrame, toxicity_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Creates the unified binary target column 'abusive'.

    =============================================================================
    CONCEPTUAL DIFFERENCE: 'toxic' vs COMBINED 'abusive'
    -----------------------------------------------------------------------------
    In the raw Jigsaw dataset, 'toxic' is just ONE of six individual subcategories.
    The six annotated subcategories are:
      1. toxic: General rudeness, disrespect, or hostility.
      2. severe_toxic: Extremely aggressive, vicious, or venomous comments.
      3. obscene: Vulgarity, profanity, or sexually explicit remarks.
      4. threat: Statements expressing intent to inflict physical harm or violence.
      5. insult: Personal attacks, mockery, or derogatory remarks directed at someone.
      6. identity_hate: Hate speech targeting race, religion, gender, sexual orientation, etc.

    A comment might be labeled as a 'threat' or 'identity_hate' without being
    explicitly tagged as general 'toxic' by human annotators.

    The new combined 'abusive' column acts as an overarching binary target:
      - abusive = 1: The comment contains AT LEAST ONE form of harmful content
                     (toxic, severe_toxic, obscene, threat, insult, OR identity_hate).
      - abusive = 0: The comment is completely clean/safe across ALL six categories.

    Why this is beneficial:
      - It establishes a clean binary classification problem (Abusive vs Safe)
        for our primary NLP model.
      - It preserves all 6 original subcategory columns untouched for subsequent
        multi-label classification or granular statistical breakdown.
    =============================================================================

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the toxicity columns.
        toxicity_cols (list of str, optional): List of the 6 toxicity columns.
            Defaults to TOXICITY_COLUMNS.

    Returns:
        pd.DataFrame: DataFrame with the new 'abusive' column added and original
                      columns intact.
    """
    if toxicity_cols is None:
        toxicity_cols = TOXICITY_COLUMNS

    # 1. Strictly validate labels before any computation (no silent coercion)
    validate_labels(df, toxicity_cols)

    df_result = df.copy()

    # 2. Convert toxicity columns to integer type safely
    for col in toxicity_cols:
        df_result[col] = df_result[col].astype(int)

    # 3. Create 'abusive': 1 if ANY column equals 1, else 0
    # Any row where sum across toxicity columns is >= 1 is abusive
    df_result["abusive"] = (df_result[toxicity_cols].sum(axis=1) > 0).astype(int)

    return df_result


def prepare_dataset(
    df: pd.DataFrame,
    text_col: str = "comment_text",
    toxicity_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Executes the complete Part 1 preprocessing pipeline:
    1. Handles missing/blank comments.
    2. Removes duplicate comments.
    3. Validates toxicity label values.
    4. Computes the combined 'abusive' target column.

    Parameters:
        df (pd.DataFrame): Raw loaded DataFrame.
        text_col (str): Column name for comment text.
        toxicity_cols (list of str, optional): Toxicity label column names.

    Returns:
        pd.DataFrame: Fully prepared DataFrame ready for exploratory data analysis.
    """
    if toxicity_cols is None:
        toxicity_cols = TOXICITY_COLUMNS

    # Step 1: Handle missing comment text safely
    df_cleaned = handle_missing_comments(df, text_col=text_col, drop_missing=True)

    # Step 2: Remove duplicate comments
    df_deduped = remove_duplicates(df_cleaned, subset=[text_col])

    # Step 3 & 4: Validate binary labels and create unified 'abusive' column
    df_prepared = create_abusive_label(df_deduped, toxicity_cols=toxicity_cols)

    return df_prepared
