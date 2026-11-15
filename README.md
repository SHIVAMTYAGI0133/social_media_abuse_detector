# Detection of Violent and Abusive Content in Social Media Using Machine Learning and NLP

**Author:** Third-Year Engineering Student Project  
**Complete Project Status:** Parts 1, 2, 3, and 4 Fully Implemented & Tested

---

## 📌 Project Overview
This project detects violent, toxic, and abusive content in social media discussions using Natural Language Processing (NLP) and Machine Learning techniques.

The project is structured into four modular parts:
- **Part 1: Data Preparation & Exploration** (Loaded, cleaned, and explored 159,571 comments from the Jigsaw Toxic Comment Classification Challenge).
- **Part 2: Text Preprocessing & TF-IDF Feature Extraction** (Normalized text, removed URLs, preserved negations, and extracted 10,000 unigram/bigram features).
- **Part 3: Model Training, Evaluation & Persistence** (Stratified 80/20 train/test split, prevented data leakage, handled class imbalance, and selected Logistic Regression as champion model).
- **Part 4: Streamlit Web Application (AbuseGuard AI)** (Interactive browser application for live comment classification with model-estimated confidence scores).

---

## 📁 Complete Project Structure

```
social_media_abuse_detector/
│
├── data/
│   ├── README.md                  # Instructions on downloading train.csv from Kaggle
│   └── class_distribution.png     # Class distribution plot (89.83% Safe vs 10.17% Abusive)
│
├── models/
│   ├── README.md                  # Documentation on saved model artifacts
│   ├── best_model.joblib          # Selected champion Logistic Regression classifier (Part 3)
│   ├── tfidf_vectorizer.joblib    # Fitted TF-IDF vectorizer (10,000 learned vocabulary features)
│   ├── evaluation_summary.json    # Test set evaluation metrics, classification reports, and CM
│   └── confusion_matrix.png       # Confusion matrix heatmap visualization
│
├── src/
│   ├── __init__.py                # Exposes modules cleanly across Parts 1-4
│   ├── data_loader.py             # Loads dataset from disk with strict schema validation (Part 1)
│   ├── preprocessing.py           # Handles missing rows, deduplication, builds 'abusive' label (Part 1)
│   ├── text_preprocessing.py      # Normalizes text, removes URLs, handles punctuation, keeps negations (Part 2)
│   ├── feature_extraction.py      # Configures, fits, and transforms comments via TF-IDF (Part 2)
│   ├── model_training.py          # Stratified split, trains Logistic Regression & Naive Bayes, saves best model (Part 3)
│   └── predict.py                 # Live inference helper with identical preprocessing and probability scores (Part 4)
│
├── tests/
│   ├── test_data_loader.py        # 12 automated unit tests for loading and label preparation (Part 1)
│   ├── test_text_preprocessing.py # 9 automated unit tests for text normalization (Part 2)
│   ├── test_feature_extraction.py # 7 automated unit tests for TF-IDF feature extraction (Part 2)
│   ├── test_model_training.py     # 11 automated unit tests for training, metrics, and persistence (Part 3)
│   └── test_predict.py            # 6 automated unit tests for live prediction and probability scoring (Part 4)
│
├── .gitignore                     # Excludes virtual env, caches, and raw datasets
├── requirements.txt               # Dependencies (pandas, numpy, matplotlib, scikit-learn, joblib, streamlit, pytest)
├── pytest.ini                     # Pytest configuration with pythonpath set
├── README.md                      # Comprehensive project documentation & student viva guide
├── main.py                        # Entry point for running the Part 1 exploratory pipeline
├── train.py                       # Entry point for running the Part 3 model training pipeline
└── app.py                         # Streamlit web application: AbuseGuard AI (Part 4)
```

---

## 🛠️ Technology Stack
- **Python 3.10+** (Tested on Python 3.13)
- **Pandas**: Structured tabular data manipulation and validation
- **NumPy**: Numerical operations
- **Matplotlib**: Visualization generation
- **Scikit-learn**: TF-IDF feature vectorization and classifiers (`LogisticRegression`, `MultinomialNB`, metrics)
- **Joblib**: Model serialization
- **Streamlit**: Web application dashboard
- **Pytest**: Unit testing framework (45 tests passing)

---

## 🏆 Champion Model Performance (Held-Out Test Set: 31,915 comments)

Evaluated on the held-out test set using stratified sampling (`random_state=42`):

| Metric | Logistic Regression (`class_weight='balanced'`) | Multinomial Naive Bayes (`alpha=0.5`) | Winner |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 92.55% | **94.62%** | Naive Bayes |
| **Abusive Precision** | 59.09% | **86.22%** | Naive Bayes |
| **Abusive Recall** | **86.81%** | 56.09% | **Logistic Regression (+30.72%)** |
| **Abusive F1-Score** | **70.32%** | 67.96% | **Logistic Regression** |

### Why Logistic Regression Was Selected Over Naive Bayes:
In social media content moderation, **Recall for the Abusive class** is the primary safety objective. While Naive Bayes achieved 94.62% overall accuracy, it achieved an abusive recall of only **56.09%**, allowing **1,425 abusive comments (43.9%)** to slip through undetected.  
Logistic Regression with `class_weight='balanced'` caught **2,817 of 3,245 abusive comments (86.81% recall)** and achieved the highest overall **F1-score (70.32%)**, missing only 428 abusive comments.

---

## 🚀 How to Run on Windows (Step-by-Step)

### 1. Open Windows PowerShell in Project Directory
```powershell
cd c:\Users\lucky\OneDrive\Desktop\social_media_abuse_detector
```

### 2. Activate the Virtual Environment
```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Requirements
```powershell
pip install -r requirements.txt
```

### 4. Run the Automated Test Suite (All 45 Tests)
```powershell
pytest -v
```

### 5. Run Part 1 (Data Exploration)
```powershell
python main.py
```

### 6. Run Part 3 (Model Training & Evaluation)
```powershell
python train.py
```

### 7. Run Part 4 (Streamlit Web Application)
```powershell
streamlit run app.py
```
*(Streamlit will automatically open your default web browser at `http://localhost:8501`).*

---

## 🌐 Using the Web Application (AbuseGuard AI)

1. **Paste / Type a Comment:** Enter any social media text in the large text box.
2. **Try Sample Buttons:** Click any of the four example buttons (`Constructive Edit`, `Polite Disagreement`, `Violent Threat`, `Hostile Harassment`) to pre-fill the text area.
3. **Analyze Comment:** Click **🔍 Analyze Comment** to run the prediction pipeline.
4. **View Results:**
   - **Green Badge:** Classified as **Safe**.
   - **Red Badge:** Classified as **Abusive**.
   - **Probability Meter:** Shows model-estimated probabilities for both classes.
5. **Inspect Preprocessing:** Expand the *"View Text Preprocessing Pipeline"* dropdown to inspect how the raw text was normalized before vectorization.
6. **Reset:** Click **🔄 Clear / Reset** to reset the input and clear previous results.

---

## ⚠️ Known Limitations & Disclaimers

1. **Context and Sarcasm:** As an n-gram TF-IDF linear model, the classifier evaluates surface lexical patterns. It may misclassify subtle irony, complex sarcasm, or context-dependent humor.
2. **Slang & Evolving Language:** Online slang evolves rapidly. Words not present in the training vocabulary may not be recognized as abusive.
3. **Statistical Confidence:** Probabilities displayed are model-estimated logistic sigmoid scores based on the training distribution, not absolute guarantees of real-world toxicity.
4. **Human Review:** The system is intended as an automated screening tool to assist human moderators, not as a replacement for human judgment.
