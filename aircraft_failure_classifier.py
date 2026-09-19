Content is user-generated and unverified.
Learn about artifacts
# ==========================================
# Aircraft Component Failure Classification
# ==========================================

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

warnings.filterwarnings("ignore")

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------
df = pd.read_csv("aircraft_logs.csv")

df.dropna(inplace=True)
df["log"] = df["log"].str.lower()

X = df["log"]
y = df["component"]

# ------------------------------------------
# TRAIN-TEST SPLIT (this is what makes the
# accuracy score honest — the model is
# evaluated on logs it has never seen)
# ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------
# MODEL
# ------------------------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2)
    )),
    ("clf", MultinomialNB())
])

# ------------------------------------------
# TRAIN
# ------------------------------------------
model.fit(X_train, y_train)

# ------------------------------------------
# EVALUATE ON UNSEEN TEST DATA
# ------------------------------------------
y_pred = model.predict(X_test)

print("============== MODEL PERFORMANCE ==============\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ------------------------------------------
# SAVE MODEL
# ------------------------------------------
joblib.dump(model, "aircraft_model.pkl")

# ------------------------------------------
# PREDICTION SYSTEM
# ------------------------------------------
while True:

    print("\n==============================================")
    print("Aircraft Component Failure Prediction System")
    print("==============================================")

    user_input = input("\nEnter maintenance log description: ")

    if user_input.strip() == "":
        print("Please enter valid input!")
        continue

    prediction = model.predict([user_input])[0]

    print("\nPredicted Component:", prediction)

    choice = input("\nDo you want to test again? (yes/no): ")

    if choice.lower() != "yes":
        print("\nSystem Closed.")
        break
Cookie settings
We use cookies to deliver and improve our services, analyze site usage, and if you agree, to customize or personalize your experience and market our services to you. You can read our Cookie Policy here.

