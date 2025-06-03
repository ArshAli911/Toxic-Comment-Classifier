import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# Config
MAX_LEN = 256
BATCH_SIZE = 8 # Reduced batch size further
EPOCHS = 1 # Reduced epochs for faster training on subset
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_LABELS = len(LABELS)
DATA_SUBSET_SIZE = 500 # Use a smaller subset of data

# Dataset
class ToxicBERTDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "targets": target
        }

# Model
class ToxicClassifier(nn.Module):
    def __init__(self, pretrained_model="distilbert-base-uncased", num_labels=6):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)

# Load data
df = pd.read_csv("data/toxic_comments.csv")
df = df.fillna("")
# Use a subset of the data
df = df.head(DATA_SUBSET_SIZE)
print(f"Using a subset of {DATA_SUBSET_SIZE} samples for training.")

X = df["comment_text"].tolist()
y = df[LABELS].values

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Dataloaders
train_ds = ToxicBERTDataset(X_train, y_train, tokenizer, MAX_LEN)
val_ds = ToxicBERTDataset(X_val, y_val, tokenizer, MAX_LEN)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# Model setup
model = ToxicClassifier().to(DEVICE)
optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)

        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            targets = batch["targets"].cpu().numpy()
            outputs = model(input_ids, attention_mask).cpu()
            preds = torch.sigmoid(outputs).numpy()
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    pred_labels = (all_preds > 0.5).astype(int)
    print(classification_report(all_targets, pred_labels, target_names=LABELS, zero_division=0))

# Save the trained model
MODEL_SAVE_PATH = "model_bert.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\n✅ Model saved to {MODEL_SAVE_PATH}")

