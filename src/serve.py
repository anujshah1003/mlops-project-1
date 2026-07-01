from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

class InputData(BaseModel):
    features: list

MODEL_VERSION="v1"


app = FastAPI()

# Load model once at startup
model = joblib.load("models/model.pkl")

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputData):
    print("Received input:", data.features)
    data = np.array(data.features).reshape(1, -1)
    prediction = model.predict(data)

    return {"prediction": prediction.tolist(),
            "model_version": MODEL_VERSION}