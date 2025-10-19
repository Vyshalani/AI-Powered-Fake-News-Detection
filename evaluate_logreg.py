# evaluate_logreg.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# --- Load dataset ---
df = pd.read_csv("data/raw/Namibian_real_and_fake_news.csv", encoding="ISO-8859-1")
print(df.columns)



# Make sure column names match your CSV
X = df["content"]   # your text data
y = df["label"]     # your 0/1 labels


# --- Load model and vectorizer ---
model = joblib.load("models/logreg_model.pkl")
vectorizer = joblib.load("models/logreg_vectorizer.pkl")

# --- Split and transform data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_tfidf = vectorizer.transform(X_test)

# --- Predictions ---
y_pred = model.predict(X_test_tfidf)

# --- Metrics ---
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("📊 Evaluation Metrics:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
