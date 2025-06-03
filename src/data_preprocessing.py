import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text: str) -> str:
    """
    Cleans input text by removing punctuation, converting to lowercase, and normalizing whitespace.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)           # Remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()       # Normalize whitespace
    return text

def preprocess_dataframe(df: pd.DataFrame) -> tuple:
    """
    Applies cleaning and separates features and labels.
    Returns cleaned text list and label DataFrame.
    """
    df['comment_text'] = df['comment_text'].fillna('').apply(clean_text)
    X_text = df['comment_text']
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
    return X_text, y

def vectorize_text(X_text, max_features=10000):
    """
    Converts cleaned text to TF-IDF vector format.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_vectorized = vectorizer.fit_transform(X_text)
    return X_vectorized, vectorizer
