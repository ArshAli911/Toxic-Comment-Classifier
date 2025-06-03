import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast

LABELS = [
    'toxic', 'severe_toxic', 'obscene',
    'threat', 'insult', 'identity_hate'
]
MODEL_PATH = "model_bert.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ToxicClassifier(nn.Module):
    def __init__(self, pretrained="distilbert-base-uncased", num_labels=6):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)
        return logits 


@torch.no_grad()
def load_model_and_tokenizer():
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = ToxicClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, tokenizer