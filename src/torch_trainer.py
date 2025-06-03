import torch
from sklearn.metrics import classification_report
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds).numpy() > 0.5
    targets = torch.cat(all_targets).numpy()
    print(classification_report(targets, preds, target_names=[
        'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']))
