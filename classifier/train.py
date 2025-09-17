import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from classifier.3han_model import ThreeHANModel  # adjust import if needed
import numpy as np

# Hyperparameters
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 0.001
EMBEDDING_DIM = 100
HIDDEN_DIM = 64
NUM_CLASSES = 2

# Load processed train dataset
train_df = pd.read_csv("data/processed/train.csv")

# Combine headline + body
texts = (train_df["headline"] + " " + train_df["body"]).tolist()
labels = train_df["label"].values

# Encode labels
labels = torch.tensor(labels, dtype=torch.long)

# Simple tokenizer & vocabulary
vocab = {}
for text in texts:
    for word in text.split():
        if word not in vocab:
            vocab[word] = len(vocab)+1  # start from 1
vocab_size = len(vocab)+1

# Convert texts to sequences of indices
max_len = 50  # simple fixed length
sequences = []
for text in texts:
    seq = [vocab.get(word,0) for word in text.split()]
    if len(seq) < max_len:
        seq += [0]*(max_len - len(seq))
    else:
        seq = seq[:max_len]
    sequences.append(seq)
sequences = torch.tensor(sequences, dtype=torch.long)

# Create dataset
class NewsDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = NewsDataset(sequences, labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = ThreeHANModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/final_model.pt")
print("✅ Model training complete and saved to models/final_model.pt")
