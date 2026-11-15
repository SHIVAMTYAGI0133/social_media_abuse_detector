# Dataset Information for Social Media Abuse Detector

## Overview
This project uses the **Jigsaw Toxic Comment Classification Challenge** dataset from Kaggle.
The dataset contains comments from Wikipedia talk page edits labeled by human raters for toxic behavior.

## How to Obtain the Dataset
1. Visit the Kaggle competition page:
   [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
2. You will need a free Kaggle account to download the data.
3. Download the `train.csv.zip` archive.
4. Unzip the archive to extract `train.csv`.
5. Place `train.csv` directly into this directory:
   `social_media_abuse_detector/data/train.csv`

## Dataset Columns
The real training dataset contains the following columns:
- `id`: Unique identifier for each comment
- `comment_text`: The text content of the social media / Wikipedia comment
- `toxic`: Binary label (0 or 1) indicating general toxicity
- `severe_toxic`: Binary label (0 or 1) indicating severe toxicity
- `obscene`: Binary label (0 or 1) indicating obscene or vulgar language
- `threat`: Binary label (0 or 1) indicating threats of violence or harm
- `insult`: Binary label (0 or 1) indicating insulting or demeaning remarks
- `identity_hate`: Binary label (0 or 1) indicating hate speech against protected identity groups

## Important Note
The actual `train.csv` file is large (~68 MB) and is excluded from version control via `.gitignore`.
Do not commit raw dataset files to Git.
