from pathlib import Path

import joblib
import numpy as np 
from fastapi import FastAPI
from pydantic import BaseModel, Field

Model_path = Path(__file__).parent / "model.joblib"
artifact = joblib.load(Model_path)
model = artifact["model"]
target_names = artifact["target_names"]

#creating an app now 
app = FastAPI(
    title = "Iris Classification API",
    description="Predicts the iris species from 4 flower measurements",
    version="1.0.0"
)

class IrisFeatures(BaseModel):
    sepal_length: float = Field(gt=0, description="Sepal length in cm")
    sepal_width: float = Field(gt=0, description="Sepal width in cm")
    petal_length: float = Field(gt=0, description="Petal length in cm")
    petal_width: float = Field(gt=0, description="Petal width in cm")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"sepal_length": 5.1, "sepal_width": 3.5,
                 "petal_length": 1.4, "petal_width": 0.2}
            ]
        }
    }

class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict[str, float]

@app.get("/")
def root():
    return {"message": "ML API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
    # must be same order as train
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    return {
        "prediction": pred,
        "class_name": target_names[pred],
        "probabilities": {
            name: round(float(p), 3) for name, p in zip(target_names, proba)
        },
    }