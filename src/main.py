import os
from datetime import datetime
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from src.features import transform_features

MODEL_PATH = "models/occupancy_xgb.json"
ENCODER_PATH = "models/encoder.joblib"

artifacts = {"model": None, "encoder": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Hydrates model and encoder into memory on startup."""
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        artifacts["model"] = model
        artifacts["encoder"] = joblib.load(ENCODER_PATH)
    yield
    artifacts.clear()


app = FastAPI(
    title="Room Occupancy Event-Driven Inference Service",
    version="1.0.0",
    lifespan=lifespan,
)


class SensorEvent(BaseModel):
    timestamp: datetime = Field(..., examples=["2015-02-11 14:48:00"])
    temperature: float = Field(..., examples=[21.76])
    humidity: float = Field(..., examples=[31.13])
    light: float = Field(..., examples=[437.33])
    co2: float = Field(..., examples=[1029.66])
    humidity_ratio: float = Field(..., examples=[0.005021])


class PredictionResponse(BaseModel):
    occupancy_prediction: int
    confidence_score: float
    actuator_directive: str


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    if artifacts.get("model") is None or artifacts.get("encoder") is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded in memory.",
        )
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_occupancy(event: SensorEvent):
    model = artifacts.get("model")
    encoder = artifacts.get("encoder")

    if model is None or encoder is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine unavailable.",
        )

    raw_df = pd.DataFrame(
        [
            {
                "date": event.timestamp,
                "Temperature": event.temperature,
                "Humidity": event.humidity,
                "Light": event.light,
                "CO2": event.co2,
                "HumidityRatio": event.humidity_ratio,
            }
        ]
    )

    features_df = transform_features(raw_df, encoder, fit=False)
    prob = float(model.predict_proba(features_df)[0][1])
    prediction = 1 if prob >= 0.5 else 0
    directive = "ENABLE_HVAC_AND_LIGHTS" if prediction == 1 else "STANDBY_MODE"

    return PredictionResponse(
        occupancy_prediction=prediction,
        confidence_score=round(prob, 6),
        actuator_directive=directive,
    )
