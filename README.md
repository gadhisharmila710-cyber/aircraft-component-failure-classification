# Aircraft Component Failure Classification using Machine Learning

Classifies unstructured aircraft maintenance logs into failure categories
(**Avionics**, **Hydraulics**, **Landing Gear**) using NLP and machine learning,
to speed up fault diagnosis and support predictive maintenance.

## Problem
Aircraft maintenance logs are unstructured text and slow to analyze manually.
This project automates that classification, helping reduce aircraft downtime
and improve maintenance efficiency and safety.

## Approach
- **Text vectorization:** TF-IDF (Term Frequency–Inverse Document Frequency),
  using unigrams and bigrams
- **Model:** Multinomial Naive Bayes classifier
- **Pipeline:** `TfidfVectorizer` → `MultinomialNB` (scikit-learn `Pipeline`)
- **Evaluation:** 80/20 train-test split, stratified by class, scored with
  accuracy, precision, recall, and F1-score

## Tech Stack
- Python
- pandas, scikit-learn, joblib

## How It Works
1. Load and clean the maintenance log dataset (`aircraft_logs.csv`)
2. Convert log text to numerical features using TF-IDF
3. Train a Naive Bayes classifier on the training split
4. Evaluate on a held-out test split (unseen data)
5. Save the trained model (`aircraft_model.pkl`)
6. Run an interactive prompt that classifies a new maintenance log in real time

## Usage
```bash
pip install pandas scikit-learn joblib
python aircraft_failure_classifier.py
```

## Project Team
- G. Gouthami
- M. Rani
- B. Devi Sree
- G. Sharmila

**Internal Guide:** Mr. Md Ayaz Uddin, Assistant Professor, AI & ML Dept.

## Future Scope
- Real-time sensor data integration (vibration, temperature, pressure)
- Deployment as an API (Flask/FastAPI) for live predictions
- Integration with IoT-based aircraft monitoring systems
