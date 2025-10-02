# classifier/train_logreg.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# --- paths ---
PROCESSED_TRAIN = "data/processed/train.csv"
PROCESSED_TEST  = "data/processed/test.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- load data ---
print("Loading processed data...")
train_df = pd.read_csv(PROCESSED_TRAIN, encoding="ISO-8859-1")
test_df  = pd.read_csv(PROCESSED_TEST, encoding="ISO-8859-1")

# Expect a column named 'clean_text' and 'label'
if "clean_text" not in train_df.columns:
    raise RuntimeError("Column 'clean_text' not found in data/processed/train.csv")

X_train = train_df["clean_text"].astype(str).tolist()
y_train = train_df["label"].astype(int).tolist()
X_test  = test_df["clean_text"].astype(str).tolist()
y_test  = test_df["label"].astype(int).tolist()

print(f"Samples - Train: {len(X_train)}, Test: {len(X_test)}")

# --- vectorizer ---
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=2)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print("Vectorizer done. Vocab size:", len(vectorizer.vocabulary_))

# --- classifier ---
print("Training Logistic Regression (this can take a moment)...")
clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="saga", random_state=42)
clf.fit(X_train_vec, y_train)

# --- evaluation ---
print("Evaluating on test set...")
y_pred = clf.predict(X_test_vec)
y_proba = clf.predict_proba(X_test_vec)[:, 1]  # prob for class=1 (real)

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0))
}

print("Metrics:")
print(json.dumps(metrics, indent=2))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# --- save artifacts ---
model_path = os.path.join(MODEL_DIR, "logreg_model.pkl")
vec_path   = os.path.join(MODEL_DIR, "logreg_vectorizer.pkl")
meta_path  = os.path.join(MODEL_DIR, "logreg_metadata.json")

joblib.dump(clf, model_path)
joblib.dump(vectorizer, vec_path)

meta = {
    "vocab_size": len(vectorizer.vocabulary_),
    "metrics": metrics
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"\nSaved model -> {model_path}")
print(f"Saved vectorizer -> {vec_path}")
print(f"Saved metadata -> {meta_path}")
