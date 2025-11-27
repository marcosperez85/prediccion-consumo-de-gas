from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import os

from config_loader import load_config
from predict import predict_single, load_model_info, get_required_features

# Cargar configuración al inicio
try:
    config = load_config()
    dataset_config = config.get('dataset', {})
    raw_features = dataset_config.get('raw_feature_columns', [])
    target_col = dataset_config.get('target_col', 'value')
    print(f"✅ Configuración cargada: {len(raw_features)} features raw")
except Exception as e:
    print(f"⚠️  Error cargando configuración: {e}")
    raw_features = []
    target_col = 'value'

# Cargar modelo al inicio
model = None
model_info = None

try:
    model = joblib.load("models/best_model.pkl")
    model_info = load_model_info()
    print(f"✅ Modelo cargado: {model_info['model_type']}")
    print(f"   Features requeridas: {len(model_info['features'])}")
except FileNotFoundError:
    print("⚠️  Modelo no encontrado. Ejecuta: python model_compare.py --save")
except Exception as e:
    print(f"⚠️  Error cargando modelo: {e}")

# Crear aplicación FastAPI
app = FastAPI(
    title="Industrial Time Series Forecasting API",
    description="API configurable para predicción de series temporales industriales",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Modelo Pydantic dinámico basado en configuración
def create_predict_request_model():
    """Crea el modelo de request dinámicamente basado en la configuración."""
    
    # Campos básicos según configuración
    field_definitions = {}
    
    # Features raw desde configuración
    for feature in raw_features:
        if 'temperature' in feature.lower():
            field_definitions[feature] = (float, Field(..., description="Temperatura en °C", example=22.5))
        elif 'demand' in feature.lower():
            field_definitions[feature] = (float, Field(..., description="Factor de demanda (0-1)", ge=0, le=1, example=0.75))
        elif 'efficiency' in feature.lower():
            field_definitions[feature] = (float, Field(..., description="Eficiencia operacional (0-1)", ge=0, le=1, example=0.85))
        elif 'price' in feature.lower():
            field_definitions[feature] = (float, Field(..., description="Precio de energía", gt=0, example=85.0))
        else:
            field_definitions[feature] = (float, Field(..., description=f"Feature: {feature}", example=1.0))
    
    # Features temporales opcionales
    field_definitions.update({
        'hour': (Optional[int], Field(None, description="Hora del día (0-23)", ge=0, le=23, example=14)),
        'day_of_week': (Optional[int], Field(None, description="Día de la semana (0=Lunes)", ge=0, le=6, example=2)),
        'month': (Optional[int], Field(None, description="Mes (1-12)", ge=1, le=12, example=6)),
        'is_weekend': (Optional[int], Field(None, description="Es fin de semana (0/1)", ge=0, le=1, example=0)),
    })
    
    # Features avanzadas opcionales (para casos donde el usuario las conozca)
    if model_info:
        for feature in model_info['features']:
            if 'lag_' in feature and feature not in field_definitions:
                field_definitions[feature] = (Optional[float], Field(None, description=f"Valor lag: {feature}", example=1000.0))
            elif 'rolling_' in feature and feature not in field_definitions:
                field_definitions[feature] = (Optional[float], Field(None, description=f"Estadística móvil: {feature}", example=1000.0))
    
    # Crear modelo dinámicamente
    PredictRequest = type('PredictRequest', (BaseModel,), {
        '__annotations__': field_definitions,
        'Config': type('Config', (), {
            'schema_extra': {
                "example": {
                    **{feat: 22.5 if 'temp' in feat.lower() else 
                              0.75 if 'demand' in feat.lower() else
                              0.85 if 'efficiency' in feat.lower() else
                              85.0 if 'price' in feat.lower() else 1.0 
                       for feat in raw_features},
                    "hour": 14,
                    "day_of_week": 2,
                    "month": 6,
                    "is_weekend": 0
                }
            }
        })
    })
    
    return PredictRequest

# Crear modelo de request
PredictRequest = create_predict_request_model()

# Modelos de respuesta
class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Valor predicho")
    model_type: str = Field(..., description="Tipo de modelo usado")
    model_mae: float = Field(..., description="MAE del modelo en test")
    model_r2: float = Field(..., description="R² del modelo en test")
    features_count: int = Field(..., description="Número de features utilizadas")
    
class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado del servicio")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    config_loaded: bool = Field(..., description="Si la configuración está cargada")
    features_available: int = Field(..., description="Número de features configuradas")

class ModelInfoResponse(BaseModel):
    model_type: str = Field(..., description="Tipo de modelo")
    test_mae: float = Field(..., description="MAE en test")
    test_r2: float = Field(..., description="R² en test")
    features_count: int = Field(..., description="Número de features")
    target_column: str = Field(..., description="Columna objetivo")
    raw_features: List[str] = Field(..., description="Features raw configuradas")

# Dependencias
def get_model_dependency():
    """Dependencia para verificar que el modelo esté cargado."""
    if model is None or model_info is None:
        raise HTTPException(
            status_code=503, 
            detail="Modelo no disponible. Ejecuta: python model_compare.py --save"
        )
    return model, model_info

# Endpoints
@app.get("/", response_model=Dict[str, Any])
def root():
    """Endpoint raíz con información general."""
    return {
        "message": "API para predicción de series temporales industriales",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "config_loaded": len(raw_features) > 0,
        "target_column": target_col,
        "raw_features_count": len(raw_features),
        "endpoints": {
            "predict": "POST /predict - Realizar predicción",
            "health": "GET /health - Estado del servicio",  
            "model-info": "GET /model-info - Información del modelo",
            "features": "GET /features - Lista de features requeridas",
            "docs": "GET /docs - Documentación interactiva"
        }
    }

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, model_data=Depends(get_model_dependency)):
    """
    Realiza una predicción basada en las features proporcionadas.
    
    Las features se toman automáticamente de la configuración en config.yaml.
    Features opcionales se rellenan con valores por defecto si no se proporcionan.
    """
    try:
        # Convertir request a diccionario
        input_data = request.dict(exclude_unset=True)
        
        # Realizar predicción usando el sistema configurable
        result = predict_single(input_data)
        
        return PredictResponse(
            prediction=result['prediction'],
            model_type=result['model_type'],
            model_mae=result['model_mae'],
            model_r2=result['model_r2'],
            features_count=result['features_count']
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {str(e)}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Verificación de salud del servicio."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        config_loaded=len(raw_features) > 0,
        features_available=len(raw_features)
    )

@app.get("/model-info", response_model=ModelInfoResponse)
def model_info_endpoint(model_data=Depends(get_model_dependency)):
    """Información detallada del modelo cargado."""
    _, info = model_data
    
    return ModelInfoResponse(
        model_type=info['model_type'],
        test_mae=info['test_mae'],
        test_r2=info['test_r2'],
        features_count=len(info['features']),
        target_column=target_col,
        raw_features=raw_features
    )

@app.get("/features")
def get_features_endpoint(model_data=Depends(get_model_dependency)):
    """Lista de features requeridas por el modelo."""
    _, info = model_data
    
    return {
        "raw_features": raw_features,
        "model_features": info['features'],
        "raw_features_count": len(raw_features),
        "total_features_count": len(info['features']),
        "feature_engineering_applied": len(info['features']) > len(raw_features)
    }

@app.get("/config")
def get_config_endpoint():
    """Configuración actual del sistema."""
    return {
        "dataset": dataset_config,
        "feature_engineering": config.get('feature_engineering', {}),
        "training": config.get('training', {}),
        "raw_features": raw_features,
        "target_column": target_col
    }

# Endpoint para predicciones de ejemplo
@app.get("/predict/example")
def predict_example(model_data=Depends(get_model_dependency)):
    """Realiza una predicción con datos de ejemplo."""
    try:
        from predict import predict_from_config_example
        result = predict_from_config_example()
        
        return {
            "example_input": result['input_features'],
            "prediction": result['prediction'],
            "model_info": {
                "type": result['model_type'],
                "mae": result['model_mae'],
                "r2": result['model_r2']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción de ejemplo: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 INICIANDO API CONFIGURABLE")
    print("=" * 40)
    print(f"Modelo cargado: {'✅' if model else '❌'}")
    print(f"Configuración: {'✅' if raw_features else '❌'}")
    print(f"Features raw: {len(raw_features)}")
    print(f"Target: {target_col}")
    print("=" * 40)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)