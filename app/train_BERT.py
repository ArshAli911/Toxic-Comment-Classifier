import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertModel, AdamW
from tqdm import tqdm

# Constants
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }

    def __len__(self):
        return len(self.texts)

# Model Class
class ToxicClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(LABELS))

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0])

# Load Data
df = pd.read_csv("data/toxic_comments.csv")
df = df.fillna("")
texts = df["comment_text"].tolist()
labels = df[LABELS].values

# Split
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)

# Tokenizer and DataLoaders
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_ds = ToxicDataset(X_train, y_train, tokenizer)
val_ds = ToxicDataset(X_val, y_val, tokenizer)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16)

# Model and Optimizer
model = ToxicClassifier().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.BCEWithLogitsLoss()

# Training Loop
for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dl, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss:.4f}")

# Evaluation
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in val_dl:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].cpu().numpy()
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels)

preds = (np.vstack(all_preds) > 0.5).astype(int)
targets = np.vstack(all_targets)
print("\nClassification Report:\n")
print(classification_report(targets, preds, target_names=LABELS))

# Save the model
torch.save(model.state_dict(), "model_bert.pth")
print("\n✅ model_bert.pth saved successfully!")
