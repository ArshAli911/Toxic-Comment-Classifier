import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_tfidf_model(X_vectorized, y, save_path="models/toxic_model.pkl"):
    """
    Trains a OneVsRest Logistic Regression model on TF-IDF vectors.
    Saves the trained model to disk.
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    # Define model
    model = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("🔍 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=y.columns))

    # Save model
    joblib.dump(model, save_path)
    print(f"\n✅ Model saved to {save_path}")
    return model
