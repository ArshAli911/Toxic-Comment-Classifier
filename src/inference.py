import joblib
import re
import numpy as np

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def clean_text(text):
    """
    Basic cleaning to match training preprocessing.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_model_and_vectorizer (model_path="models/toxic_model.pkl",
                              vectorizer_path="models/tfidf_vectorizer.pkl"):
    """
    Loads the trained model and vectorizer from disk.
    """
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_text(text, model, vectorizer):
    """
    Predicts labels for a given input text.
    """
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    preds = model.predict(vec)[0]
    result = {label: bool(pred) for label, pred in zip(LABELS, preds)}
    return result
