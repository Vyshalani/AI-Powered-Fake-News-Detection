import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classifier.threehan_model import ThreeHANModel
from classifier.utils import TextProcessor

# --- SETTINGS ---
MAX_LEN = 100
BATCH_SIZE = 16
MODEL_PATH = "models/final_model.pt"
TEST_CSV = "data/processed/test.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LOAD TEST DATA ---
test_df = pd.read_csv(TEST_CSV)
texts = (test_df["headline"] + " " + test_df["body"]).tolist()
labels = torch.tensor(test_df["label"].values, dtype=torch.long)

# --- TEXT PROCESSOR ---
processor = TextProcessor(max_len=MAX_LEN)
processor.build_vocab(texts)  # In practice, load vocab from training for consistency
X_test = processor.texts_to_tensor(texts)

# --- LOAD MODEL ---
vocab_size = len(processor.vocab) + 1
model = ThreeHANModel(vocab_size=vocab_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# --- PREDICTIONS ---
X_test = X_test.to(DEVICE)
labels = labels.to(DEVICE)
with torch.no_grad():
    outputs = model(X_test)
    preds = torch.argmax(outputs, dim=1)

# --- METRICS ---
acc = accuracy_score(labels.cpu(), preds.cpu())
prec = precision_score(labels.cpu(), preds.cpu(), zero_division=0)
rec = recall_score(labels.cpu(), preds.cpu(), zero_division=0)
f1 = f1_score(labels.cpu(), preds.cpu(), zero_division=0)

print("------ Evaluation Results ------")
print(f"Samples tested: {len(labels)}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
print("-------------------------------")
