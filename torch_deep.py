import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.torch_model import ToxicCommentLSTM
from src.torch_dataset import ToxicCommentDataset, collate_fn
from src.torch_trainer import train_epoch, evaluate
from sklearn.feature_extraction.text import CountVectorizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
df = pd.read_csv("data/toxic_comments.csv")
texts = df["comment_text"].fillna("").tolist()
labels = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

# Tokenization
vectorizer = CountVectorizer(token_pattern=r'\b\w+\b', max_features=10000)
vectorizer.fit(texts)
tokenizer = lambda x: [vectorizer.vocabulary_.get(w, 0)+1 for w in x.lower().split()]

# Datasets and loaders
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.1, random_state=42)
train_ds = ToxicCommentDataset(X_train, y_train, tokenizer)
val_ds = ToxicCommentDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_ds, batch_size=64, collate_fn=collate_fn)

# Model
model = ToxicCommentLSTM(vocab_size=10001, embed_dim=128, hidden_dim=64, num_classes=6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.BCELoss()

# Train
for epoch in range(5):
    loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"📘 Epoch {epoch+1} - Loss: {loss:.4f}")
    evaluate(model, val_loader, device)