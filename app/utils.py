import torch
from app.model import load_model_and_tokenizer, LABELS, DEVICE

model, tokenizer = load_model_and_tokenizer()

def predict(text: str, threshold: float = 0.5):
	encoded = tokenizer.encode_plus(
		text,
		add_special_tokens=True,
		max_length=256,
		padding="max_length",
		truncation=True,
		return_attention_mask=True,
		return_tensors="pt"
	)
	
	return encoded
	input_ids = encoded["input_ids"].to(DEVICE)
	attention_mask = encoded["attention_mask"].to(DEVICE)