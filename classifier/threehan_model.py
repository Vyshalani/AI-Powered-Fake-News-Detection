import torch
import torch.nn as nn
import torch.nn.functional as F

class WordAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        # x: batch_size x seq_len x embedding_dim
        H, _ = self.gru(x)  # H: batch_size x seq_len x 2*hidden_dim
        attn_weights = F.softmax(self.attention(H), dim=1)  # batch_size x seq_len x 1
        M = torch.sum(attn_weights * H, dim=1)  # batch_size x 2*hidden_dim
        return M

class ThreeHANModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.word_attention = WordAttention(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        # x: batch_size x seq_len
        embeds = self.embedding(x)  # batch_size x seq_len x embedding_dim
        attn_out = self.word_attention(embeds)  # batch_size x 2*hidden_dim
        out = self.fc(attn_out)  # batch_size x num_classes
        return out
