from fastapi import FastAPI 
from .schemas import CommentRequest, PredictionResponse
from .utils import predict

app = FastAPI(
    title="Toxic Comment Classifier API",
    version="1.0",
    description="FastAPI backend for BERT-based toxicity classification"
)

@app.get("/")
def root():
    return {"message": "Toxic Comment Classifier API is up!"}

@app.post("/predict", response_model=PredictionResponse)
def classify_comment(request: CommentRequest):
    label_probs, predicted_labels = predict(request.text)
    return PredictionResponse(label_probs=label_probs, predicted_labels=predicted_labels)  