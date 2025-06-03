from src.preprocess import load_and_clean_data, vectorize_text
from src.train import train_model, save_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    df = load_and_clean_data("data/toxic_comments.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        df['comment_text'], df['label'], test_size=0.2, random_state=42)

    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    model = train_model(X_train_vec, y_train)

    evaluate_model(model, X_test_vec, y_test)

    save_model(model, vectorizer)
