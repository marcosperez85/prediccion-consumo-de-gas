from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Gas Consumption Prediction API")

# Cargamos el modelo (el que prefieras de tus comparaciones)
model = joblib.load("models/model.pkl")
#model = joblib.load("models/randomforest_model.pkl")

class PredictRequest(BaseModel):
    temp: float
    humidity: float
    wind: float
    hour: int
    weekday: int
    is_weekend: int
    lag1: float
    lag24: float

@app.post("/predict")
def predict(req: PredictRequest):
    data = pd.DataFrame([req.dict()])
    pred = model.predict(data)[0]
    return {"prediction_m3": pred}

@app.get("/")
def root():
    return {"message": "API para predecir consumo de gas"}
