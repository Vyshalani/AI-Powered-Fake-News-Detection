import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from classifier.threehan_model import ThreeHANModel
from classifier.utils import TextProcessor

# ----------------- SETTINGS -----------------
MODEL_PATH = "models/final_model.pt"
BATCH_SIZE = 4
EPOCHS = 3
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 100

# ----------------- DATASET -----------------
class NewsDataset(Dataset):
    def __init__(self, csv_path, processor):
        df = pd.read_csv(csv_path)
        # Combine headline + body into text
        self.texts = (df["headline"].astype(str) + " " + df["body"].astype(str)).tolist()
        self.labels = df["label"].map({"real":0, "fake":1}).tolist()  # convert labels to 0/1
        processor.build_vocab(self.texts)
        self.processor = processor


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_tensor = self.processor.texts_to_tensor([self.texts[idx]]).squeeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text_tensor, label

processor = TextProcessor(max_len=MAX_LEN)
dataset = NewsDataset("data/raw/namibia_news_sample.csv", processor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

vocab_size = len(processor.vocab)  # your built vocab size
model = ThreeHANModel(vocab_size=vocab_size)
model.to(DEVICE)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for texts, labels in dataloader:
        texts = texts.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

import pickle

# Save vocab for inference
with open("models/vocab.pkl", "wb") as f:
    pickle.dump(processor.vocab, f)

print("Vocab saved to models/vocab.pkl")

