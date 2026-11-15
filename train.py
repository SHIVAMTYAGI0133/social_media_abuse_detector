"""
train.py - Project Entry Point for Part 3: Model Training and Evaluation.

Usage:
    python train.py                  # Trains on data/train.csv by default
    python train.py path/to/data.csv # Trains on a custom dataset CSV path
"""

import sys
from pathlib import Path
from src.model_training import run_training_pipeline


def main() -> None:
    # Optional CLI argument for custom dataset path
    custom_dataset = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None

    try:
        run_training_pipeline(dataset_path=custom_dataset)
    except FileNotFoundError as fnf_err:
        print("\n" + "=" * 70)
        print("DATASET NOT FOUND".center(70))
        print("=" * 70)
        print(str(fnf_err))
        print("ACTION REQUIRED:")
        print("1. Download 'train.csv' from Kaggle (Jigsaw Toxic Comment Challenge).")
        print("2. Place it inside the 'data/' folder: data/train.csv")
        print("3. Run 'python train.py' again after placing the file.")
        sys.exit(1)
    except Exception as err:
        print(f"\n[CRITICAL ERROR] Model training pipeline failed: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
