from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from src.model import CryptoClassifier
import os

class FeatureInput(BaseModel):
    features: list[float]  

model = CryptoClassifier()
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models", "crypto_classifier.pth"))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(input_data: FeatureInput):
    try:
        x = torch.tensor([input_data.features], dtype=torch.float32)
        with torch.no_grad():
            output = model(x)
            _, predicted = torch.max(output, 1)
            label = int(predicted.item())
        return {"prediction": "UP" if label == 1 else "DOWN"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))