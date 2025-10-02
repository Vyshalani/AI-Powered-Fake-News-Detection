import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

RAW_CSV = "data/raw/Namibian_real_and_fake_news.csv"
PROCESSED_DIR = "data/processed"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# --- Load CSV ---
df = pd.read_csv(RAW_CSV, encoding="ISO-8859-1")



print("Raw data loaded:", df.shape)
print(df.head())

# --- Clean text ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-zÄäÖöÜüßÉéÈèÊêÁáÀàÂâÍíÎîÓóÒòÔôÚúÛû\s]", "", text)  # keep English + Afrikaans chars
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["content"].apply(clean_text)

# --- Train/test split ---
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
df.to_csv(os.path.join(PROCESSED_DIR, "processed_full.csv"), index=False)

print("✅ Preprocessing complete.")
print("Train size:", train_df.shape, "Test size:", test_df.shape)
print("Saved to:", PROCESSED_DIR)
