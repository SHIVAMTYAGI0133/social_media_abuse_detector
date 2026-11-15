"""
main.py - Entry point for Part 1: Data Preparation & Exploration.

This script executes the complete data preparation pipeline:
1. Loads the Jigsaw Toxic Comment dataset from 'data/train.csv'.
2. Performs data cleaning, deduplication, and label preparation.
3. Prints dataset dimensions, missing values, and head preview.
4. Analyzes class distribution (Safe vs Abusive).
5. Generates and saves a class distribution bar chart.
6. Displays reproducible sample comments for both classes (random_state=42).
7. Prints a summary of the prepared dataset.

If 'data/train.csv' is missing, it displays clear, actionable download instructions.
"""

import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import load_data, get_default_dataset_path
from src.preprocessing import prepare_dataset, TOXICITY_COLUMNS


def print_section_header(title: str) -> None:
    """Prints a styled visual header for console output."""
    separator = "=" * 70
    print(f"\n{separator}\n{title.center(70)}\n{separator}")


def plot_class_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """
    Creates and saves a bar chart showing the distribution of Safe vs Abusive comments.

    Parameters:
        df (pd.DataFrame): The prepared DataFrame containing the 'abusive' column.
        output_path (Path): Path where the chart image will be saved.
    """
    counts = df["abusive"].value_counts().rename(index={0: "Safe (0)", 1: "Abusive (1)"})
    total = len(df)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values, color=["#2ecc71", "#e74c3c"], width=0.5)

    plt.title("Distribution of Comments: Safe vs Abusive", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Category", fontsize=11, fontweight="bold")
    plt.ylabel("Number of Comments", fontsize=11, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars with counts and percentages
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (total * 0.01),
            f"{height:,}\n({percentage:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[INFO] Class distribution chart saved to: {output_path}")


def display_examples(df: pd.DataFrame, label_value: int, label_name: str, n_samples: int = 5) -> None:
    """
    Prints random sample comments for a given class using reproducible random_state=42.

    Parameters:
        df (pd.DataFrame): Prepared DataFrame.
        label_value (int): Class label (0 for safe, 1 for abusive).
        label_name (str): Human-readable name ("Safe" or "Abusive").
        n_samples (int): Number of examples to display.
    """
    subset = df[df["abusive"] == label_value]
    available = len(subset)

    print(f"\n--- {label_name.upper()} COMMENT EXAMPLES (Label abusive = {label_value}) ---")

    if available == 0:
        print(f"No comments found for category '{label_name}'.")
        return

    sample_count = min(n_samples, available)
    samples = subset.sample(n=sample_count, random_state=42)

    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        clean_text = row["comment_text"].replace("\n", " ").strip()
        # Truncate very long comments for clean readable console display
        truncated_text = clean_text[:140] + ("..." if len(clean_text) > 140 else "")
        
        if label_value == 1:
            active_labels = [col for col in TOXICITY_COLUMNS if row[col] == 1]
            label_tag = f" [Subcategories: {', '.join(active_labels)}]"
        else:
            label_tag = " [Subcategories: None (Clean)]"

        print(f"{idx}. {truncated_text}")
        print(f"   {label_tag}\n")


def run_pipeline() -> None:
    """
    Main execution pipeline for Part 1: Data Preparation.
    """
    print_section_header("PART 1: DATA PREPARATION & EXPLORATION")
    print("Project: Detection of Violent and Abusive Content in Social Media")
    print("Dataset: Jigsaw Toxic Comment Classification Challenge")

    # Allow custom dataset path via command line argument, or use default 'data/train.csv'
    dataset_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else get_default_dataset_path()

    # Step 1: Load and validate dataset
    print(f"\n[STEP 1] Checking and loading dataset from: {dataset_path}")
    try:
        raw_df = load_data(dataset_path)
    except FileNotFoundError as fnf_err:
        print_section_header("DATASET NOT FOUND")
        print(str(fnf_err))
        print("ACTION REQUIRED:")
        print("1. Please download the dataset as instructed above.")
        print("2. Ensure the file is saved as 'data/train.csv'.")
        print("3. Run 'python main.py' again after placing the file.")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[CRITICAL ERROR] Failed to load dataset: {exc}")
        sys.exit(1)

    print("[OK] Dataset successfully loaded into memory!")

    # Step 2: Preprocess dataset & prepare combined 'abusive' label
    print("\n[STEP 2] Running data cleaning and label preparation pipeline...")
    prepared_df = prepare_dataset(raw_df)
    print("[OK] Preprocessing completed successfully!")

    # Step 3: Print dimensions and first 5 rows
    print_section_header("DATASET DIMENSIONS & PREVIEW")
    print(f"Total Rows:    {len(prepared_df):,}")
    print(f"Total Columns: {len(prepared_df.columns)}")
    print(f"Columns:       {list(prepared_df.columns)}")

    print("\nFirst 5 Rows of Prepared Dataset:")
    preview_cols = ["comment_text"] + TOXICITY_COLUMNS + ["abusive"]
    display_df = prepared_df[preview_cols].copy()
    # Shorten comment text preview for tabular console display
    display_df["comment_text"] = display_df["comment_text"].apply(
        lambda t: (t[:50] + "...") if len(t) > 50 else t
    )
    print(display_df.head(5).to_string(index=True))

    # Step 4: Check missing values in important columns
    print_section_header("MISSING VALUES AUDIT")
    null_counts = prepared_df[preview_cols].isnull().sum()
    print("Missing values per column:")
    for col, count in null_counts.items():
        status = "[CLEAN]" if count == 0 else f"[WARNING: {count} NULLS]"
        print(f"  - {col.ljust(16)}: {count} missing values {status}")

    # Step 5: Class distribution breakdown
    print_section_header("CLASS DISTRIBUTION ANALYSIS")
    counts = prepared_df["abusive"].value_counts()
    safe_count = counts.get(0, 0)
    abusive_count = counts.get(1, 0)
    total_count = len(prepared_df)

    safe_pct = (safe_count / total_count) * 100 if total_count > 0 else 0
    abusive_pct = (abusive_count / total_count) * 100 if total_count > 0 else 0

    print(f"Total Comments:   {total_count:,}")
    print(f"Safe Comments:    {safe_count:,} ({safe_pct:.2f}%)")
    print(f"Abusive Comments: {abusive_count:,} ({abusive_pct:.2f}%)")

    print("\nBreakdown of Individual Toxicity Subcategories:")
    for col in TOXICITY_COLUMNS:
        col_count = (prepared_df[col] == 1).sum()
        col_pct = (col_count / total_count) * 100 if total_count > 0 else 0
        print(f"  - {col.ljust(14)}: {col_count:,} ({col_pct:.2f}%)")

    # Step 6: Generate and save distribution chart
    print_section_header("GENERATING DISTRIBUTION VISUALIZATION")
    chart_path = Path(__file__).resolve().parent / "data" / "class_distribution.png"
    plot_class_distribution(prepared_df, chart_path)

    # Step 7: Print reproducible sample comments (5 safe, 5 abusive)
    print_section_header("REPRODUCIBLE COMMENT EXAMPLES (random_state=42)")
    display_examples(prepared_df, label_value=0, label_name="Safe", n_samples=5)
    display_examples(prepared_df, label_value=1, label_name="Abusive", n_samples=5)

    # Step 8: Dataset preparation summary
    print_section_header("PART 1 DATA PREPARATION SUMMARY")
    print(
        "[OK] Data loading and schema validation: Complete\n"
        "[OK] Missing text and duplicate comment handling: Complete\n"
        "[OK] Strict binary toxicity label validation: Complete\n"
        "[OK] Unified 'abusive' target column creation: Complete\n"
        "[OK] Exploratory analysis and class distribution chart: Complete\n"
        "\nThe dataset is now fully cleaned and validated for Part 2 (Text Preprocessing & Feature Extraction)."
    )
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
