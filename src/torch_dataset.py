import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
from torch.nn.utils.rnn import pad_sequence
import re

class ToxicCommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=200):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.texts[idx])
        tokens = tokens[:self.max_len]
        tensor_tokens = torch.tensor(tokens, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float)
        return tensor_tokens, label_tensor

def collate_fn(batch):
    texts, labels = zip(*batch)
    padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded, labels
