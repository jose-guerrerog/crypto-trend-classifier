from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn

# Define input schema
class FeatureInput(BaseModel):
    features: list[float]  # Expecting 5 input features

# Define model
class CryptoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

# Load the trained model
model = CryptoClassifier()
model.load_state_dict(torch.load("./src/model.pth", map_location=torch.device("cpu")))
model.eval()

# Create FastAPI app
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