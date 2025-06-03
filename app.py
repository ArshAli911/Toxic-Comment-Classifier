import streamlit as st
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertModel

st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")

# Config
MODEL_PATH = "model_bert.pth"
PRETRAINED_NAME = "distilbert-base-uncased"
MAX_LEN = 256
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class ToxicClassifier(nn.Module):
    def __init__(self, pretrained_model=PRETRAINED_NAME, num_labels=6):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        return self.classifier(pooled_output)

# Load model + tokenizer
@st.cache_resource
def load_model():
    model = ToxicClassifier().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return DistilBertTokenizerFast.from_pretrained(PRETRAINED_NAME)

model = load_model()
tokenizer = load_tokenizer()

# Inference
def predict(text):
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
    
    return {label: float(prob) for label, prob in zip(LABELS, probs)}

# Streamlit UI
st.title("🧠 Toxic Comment Classifier")
st.write("Enter a comment and check if it's toxic.")

user_input = st.text_area("Your comment", height=150)

if st.button("Classify"):
    if user_input.strip():
        results = predict(user_input)
        st.subheader("Toxicity Probabilities")
        for label, score in results.items():
            st.write(f"**{label}**: {score:.2f}")
        
        st.subheader("Likely Labels")
        likely = [label for label, score in results.items() if score > 0.5]
        st.success(", ".join(likely) if likely else "None")
    else:
        st.warning("Please enter a comment to analyze.")
