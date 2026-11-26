from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
target_names = artifact["target_names"]
model_meta = {
    "model_name": artifact["model_name"],
    "model_version": artifact["model_version"],
    "algorithm": artifact["algorithm"],
    "run_id": artifact["run_id"],
    "metrics": artifact["metrics"],
}

app = FastAPI(
    title="Iris Classifier API",
    description="Serves the champion model from the MLflow Model Registry.",
    version="2.0.0",
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
    model_version: int


class ModelInfo(BaseModel):
    model_name: str
    model_version: int
    algorithm: str
    run_id: str
    metrics: dict[str, float]


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    return model_meta


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures):
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
        "model_version": model_meta["model_version"],
    }