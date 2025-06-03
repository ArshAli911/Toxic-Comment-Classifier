import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizerFast

class ToxicityDataset(Dataset):
    def __init__(self, texts, labels=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length", 
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item
