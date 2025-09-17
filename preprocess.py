import pandas as pd
import re
from sklearn.model_selection import train_test_split

# Load raw dataset
raw_path = "data/raw/namibia_news_sample.csv"
df = pd.read_csv(raw_path)

# Clean text (lowercase, remove special chars)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\sáéíóúäëïöüñ]", "", text)  # keep Afrikaans accents
    return text

df["headline"] = df["headline"].astype(str).apply(clean_text)
df["body"] = df["body"].astype(str).apply(clean_text)

# Encode labels (real=0, fake=1)
df["label"] = df["label"].map({"real": 0, "fake": 1})

# Split into train/test (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# Save processed datasets
train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

print("✅ Data preprocessing complete!")
print(f"Train set: {train_df.shape[0]} samples")
print(f"Test set: {test_df.shape[0]} samples")
