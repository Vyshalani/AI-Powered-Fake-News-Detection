import torch
import re
import numpy as np

class TextProcessor:
    def __init__(self, vocab=None, max_len=100):
        """
        vocab: dictionary mapping word -> index
        max_len: maximum sequence length for padding
        """
        self.vocab = vocab
        self.max_len = max_len

    def build_vocab(self, texts):
        """Build vocab from a list of texts"""
        self.vocab = {}
        idx = 1  # 0 reserved for padding
        for text in texts:
            for word in text.split():
                if word not in self.vocab:
                    self.vocab[word] = idx
                    idx += 1

    def clean_text(self, text):
        """Lowercase, remove unwanted characters, keep Afrikaans accents"""
        text = str(text).lower()
        text = re.sub(r"[^a-zA-Z0-9\sáéíóúäëïöüñ]", "", text)
        return text

    def text_to_sequence(self, text):
        """Convert text to list of indices with padding"""
        seq = [self.vocab.get(word, 0) for word in text.split()]
        if len(seq) < self.max_len:
            seq += [0] * (self.max_len - len(seq))
        else:
            seq = seq[:self.max_len]
        return torch.tensor(seq, dtype=torch.long)

    def texts_to_tensor(self, texts):
        """Convert list of texts to tensor of sequences"""
        sequences = [self.text_to_sequence(self.clean_text(t)) for t in texts]
        return torch.stack(sequences)
