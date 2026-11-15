# Trained Models Directory

This directory stores the trained machine learning models, fitted TF-IDF vectorizers, and evaluation artifacts produced in **Part 3: Model Training and Evaluation**.

## Expected Artifacts:
- `best_model.joblib`: The selected high-performing classifier (Logistic Regression or Naive Bayes).
- `tfidf_vectorizer.joblib`: The fitted TF-IDF vectorizer containing the vocabulary learned strictly from training data.
- `evaluation_summary.json`: Detailed evaluation metrics (accuracy, precision, recall, F1-score) on the held-out test set for all trained models.
- `confusion_matrix.png`: Visual confusion matrix plot for the selected best model.
- `model_metadata.json`: Information on training date, hyperparameters, and feature dimensions.
