"""
main_segmentation.py
----------------------
FastAPI backend that serves the trained customer segmentation model.

Run with:
    uvicorn main_segmentation:app --reload --port 8000

Endpoints:
    GET  /                -> health check
    GET  /personas         -> list of possible persona labels
    POST /predict          -> predict the persona for one customer
    POST /predict_batch    -> predict personas for a list of customers
"""

from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

FEATURES = [
    "age",
    "annual_income_k",
    "spending_score",
    "purchase_frequency",
    "avg_transaction_value",
]

app = FastAPI(
    title="Customer Persona Segmentation API",
    description="Predicts a customer persona/segment using a trained KMeans model.",
    version="1.0.0",
)

# ---- Load model artifacts at startup ----
try:
    model = joblib.load("segmentation_model.pkl")
    scaler = joblib.load("preprocessor.pkl")
    persona_labels = joblib.load("persona_labels.pkl")  # {cluster_id: name}
except FileNotFoundError:
    model = scaler = persona_labels = None


class CustomerInput(BaseModel):
    age: int = Field(..., ge=16, le=100, example=35)
    annual_income_k: float = Field(..., ge=0, le=1000, example=65.0,
                                    description="Annual income in $ thousands")
    spending_score: float = Field(..., ge=0, le=100, example=55.0,
                                   description="Spending score, 0-100")
    purchase_frequency: float = Field(..., ge=0, le=100, example=12.0,
                                       description="Purchases per year")
    avg_transaction_value: float = Field(..., ge=0, le=10000, example=95.0,
                                          description="Average $ per transaction")


class PredictionOutput(BaseModel):
    cluster: int
    persona: str
    persona_description: str


PERSONA_DESCRIPTIONS = {
    "Budget Shopper": "Price-sensitive customers who buy infrequently and spend little per visit.",
    "Occasional Buyer": "Moderate-income customers who shop now and then without a strong pattern.",
    "Loyal Regular": "Consistent, engaged customers with solid income and frequent purchases.",
    "High Roller": "High-income, high-frequency customers with large transaction values — your VIPs.",
}


def _predict_persona(customer: CustomerInput) -> PredictionOutput:
    if model is None or scaler is None or persona_labels is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Run train_segmentation.py first.",
        )

    x = np.array([[getattr(customer, f) for f in FEATURES]])
    x_scaled = scaler.transform(x)
    cluster_id = int(model.predict(x_scaled)[0])
    persona = persona_labels.get(cluster_id, f"Cluster {cluster_id}")
    description = PERSONA_DESCRIPTIONS.get(persona, "")

    return PredictionOutput(cluster=cluster_id, persona=persona, persona_description=description)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "message": "Customer Persona Segmentation API is running.",
    }


@app.get("/personas")
def list_personas():
    if persona_labels is None:
        raise HTTPException(status_code=503, detail="Model artifacts not found.")
    return {
        "personas": [
            {"cluster": cid, "name": name, "description": PERSONA_DESCRIPTIONS.get(name, "")}
            for cid, name in sorted(persona_labels.items())
        ]
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(customer: CustomerInput):
    return _predict_persona(customer)


@app.post("/predict_batch", response_model=List[PredictionOutput])
def predict_batch(customers: List[CustomerInput]):
    return [_predict_persona(c) for c in customers]
