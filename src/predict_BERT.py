import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel
import numpy as np

# Configuration
MODEL_PATH = "model_bert.pth"  # Path to saved model
PRETRAINED_NAME = "distilbert-base-uncased"
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Define the same model as during training
class ToxicClassifier(nn.Module):
    def __init__(self, pretrained_model=PRETRAINED_NAME, num_labels=6):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)

# Load model and tokenizer
model = ToxicClassifier().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_NAME)

# Inference function
def predict_toxic_comment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(DEVICE)
    attention_mask = inputs['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    results = {label: float(prob) for label, prob in zip(LABELS, probs)}
    return results

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a comment (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break

        preds = predict_toxic_comment(user_input)
        print("\nToxicity Probabilities:")
        for label, prob in preds.items():
            print(f"{label:15s}: {prob:.3f}")

        print("\nLikely labels:")
        likely = [label for label, p in preds.items() if p > 0.5]
        print(likely if likely else "None")