from pydantic import BaseModel

class CommentRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label_probs: dict
predicted_labels: list