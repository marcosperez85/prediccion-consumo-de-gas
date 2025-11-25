# 🏭 Industrial Time Series Forecasting

Pipeline completo y modular para predecir valores futuros en series temporales industriales. Está diseñado como un template reutilizable para experimentación, benchmarking de modelos y despliegue rápido de prototipos.

## 🎯 Objetivo

Desarrollar un sistema de predicción genérico para series temporales industriales que permita:
- Experimentación rápida con diferentes algoritmos
- Feature engineering avanzado
- Comparación objetiva de modelos
- Despliegue a través de API REST
- Visualización y análisis de resultados

## ✨ Características

- 🔧 **Feature Engineering Avanzado**: Creación automática de features temporales, lags, rolling statistics e interacciones
- 🤖 **Múltiples Modelos**: Linear Regression, Random Forest, Gradient Boosting con comparación automática
- 🚀 **API REST**: FastAPI para servir predicciones en producción
- 📊 **Métricas Completas**: MSE, RMSE, MAE, MAPE, R² para evaluación exhaustiva
- 📈 **Visualización**: Notebooks con análisis exploratorio y comparación de modelos
- 🔄 **Reproducibilidad**: Scripts automatizados para entrenamiento y evaluación
- 🏗️ **Organización MLOps**: Estructura limpia para proyectos de ML en producción

## 🏗️ Arquitectura del Proyecto

```
industrial-time-series-forecasting/
│
├── data/                                    # Datos del proyecto
│   ├── raw/                                # Datos sin procesar
│   │   └── industrial_timeseries.csv      # Dataset sintético generado
│   └── processed/                          # Datos procesados
│       ├── industrial_timeseries_featured.csv  # Con feature engineering
│       └── future_predictions.csv         # Predicciones futuras
│
├── models/                                 # Modelos entrenados
│   ├── best_model.pkl                     # Mejor modelo según métricas
│   ├── model_info.pkl                     # Metadatos del modelo
│   ├── linear_regression_model.pkl        # Modelo de regresión lineal
│   ├── random_forest_model.pkl           # Modelo Random Forest
│   └── gradient_boosting_model.pkl       # Modelo Gradient Boosting
│
├── notebooks/                             # Análisis y experimentación
│   ├── 01_load.ipynb                     # Carga y validación de datos
│   ├── 02_eda.ipynb                      # Análisis exploratorio
│   ├── 03_feature_engineering.ipynb      # Creación de características
│   ├── 04_model.ipynb                    # Entrenamiento y evaluación
│   └── 05_forecast.ipynb                 # Predicciones futuras
│
└── src/                                   # Código fuente
    ├── create_dataset.py                  # Generación de datos sintéticos
    ├── load.py                           # Funciones de carga de datos
    ├── train_model.py                    # Entrenamiento de modelos
    ├── model_compare.py                  # Comparación de modelos
    ├── predict.py                        # Funciones de predicción
    └── main_api.py                       # API FastAPI
```

## 📊 Dataset

El dataset sintético incluye 6 features principales que representan variables comunes en entornos industriales:

### Features Principales
- **`timestamp`**: Marca temporal horaria
- **`value`**: Variable objetivo (producción/consumo industrial)
- **`temperature`**: Temperatura ambiente (°C) con estacionalidad
- **`demand_factor`**: Factor de demanda del mercado (0-1)
- **`operational_efficiency`**: Eficiencia operacional (0-1)
- **`energy_price`**: Precio de energía ($/MWh)

### Features Derivadas (Feature Engineering)
- Variables temporales: `hour`, `day_of_week`, `month`, `is_weekend`
- Lags: `lag_1h`, `lag_24h`, `lag_168h`
- Rolling statistics: `rolling_mean_24h`, `rolling_std_24h`
- Interacciones: `demand_efficiency_interaction`, `temp_demand_interaction`
- Transformaciones: `temp_squared`

## 🚀 Inicio Rápido

### 1. Generar Dataset
```bash
cd src
python create_dataset.py
```

### 2. Ejecutar Notebooks (Orden recomendado)
1. `01_load.ipynb` - Cargar y explorar datos iniciales
2. `02_eda.ipynb` - Análisis exploratorio detallado
3. `03_feature_engineering.ipynb` - Crear features avanzadas
4. `04_model.ipynb` - Entrenar y comparar modelos
5. `05_forecast.ipynb` - Generar predicciones futuras

### 3. Entrenar Modelos por CLI

#### Entrenar modelo individual:
```bash
cd src

# Random Forest (recomendado)
python train_model.py --model random_forest

# Linear Regression
python train_model.py --model linear_regression

# Gradient Boosting
python train_model.py --model gradient_boosting
```

#### Comparar múltiples modelos:
```bash
# Solo comparar (no guardar)
python model_compare.py

# Comparar y guardar mejor modelo
python model_compare.py --save
```

### 4. Usar API REST

#### Iniciar servidor:
```bash
cd src
python main_api.py
```

#### Realizar predicción:
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "temperature": 22.5,
  "demand_factor": 0.75,
  "operational_efficiency": 0.85,
  "energy_price": 85.0,
  "hour": 14,
  "day_of_week": 2,
  "month": 6,
  "is_weekend": 0,
  "lag_1h": 1150.0,
  "lag_24h": 1180.0,
  "rolling_mean_24h": 1165.0,
  "rolling_std_24h": 25.0
}'
```

#### Endpoints disponibles:
- `GET /` - Información general
- `POST /predict` - Realizar predicción
- `GET /health` - Estado del servicio
- `GET /model-info` - Información del modelo cargado

## 🔧 Uso del Código

### Predicción Individual
```python
from src.predict import predict_single

result = predict_single(
    temperature=22.5,
    demand_factor=0.75,
    operational_efficiency=0.85,
    energy_price=85.0,
    hour=14
)
print(f"Predicción: {result['prediction']}")
```

### Predicción en Lote
```python
from src.predict import predict_batch
import pandas as pd

df = pd.read_csv("your_data.csv")
predictions = predict_batch(df)
```

### Cargar Datos
```python
from src.load import load_data, load_processed_data

# Datos originales
df_raw = load_data()

# Datos con feature engineering
df_processed = load_processed_data()
```

## 📈 Métricas de Evaluación

El sistema evalúa modelos usando múltiples métricas:

- **MAE** (Mean Absolute Error): Error promedio absoluto
- **RMSE** (Root Mean Square Error): Penaliza errores grandes
- **R²** (R-squared): Proporción de varianza explicada
- **MAPE** (Mean Absolute Percentage Error): Error porcentual promedio

## 🛠️ Personalización

### Agregar Nuevos Modelos
Edita `model_compare.py` o `train_model.py`:

```python
from sklearn.svm import SVR

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "SVM": SVR(kernel='rbf')  # Nuevo modelo
}
```

### Modificar Features
Edita la lista `feature_columns` en los scripts:

```python
feature_columns = [
    'temperature', 'demand_factor', 'operational_efficiency',
    'your_new_feature'  # Nueva feature
]
```

### Personalizar Dataset
Modifica `create_dataset.py` para generar datos específicos para tu dominio.

## 📋 Dependencias

```txt
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
fastapi>=0.85.0
uvicorn>=0.18.0
joblib>=1.1.0
pydantic>=1.10.0
```

## 🤝 Contribuciones

1. Fork el proyecto
2. Crea una branch para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la branch (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Notas de Desarrollo

- Los modelos se guardan automáticamente en `models/`
- El split de datos es temporal (80% entrenamiento, 20% prueba)
- Las features de lag requieren datos históricos
- La API incluye validación automática de inputs
- Todos los scripts incluyen manejo de errores

## 🔮 Próximas Características

- [ ] Modelos de deep learning (LSTM, GRU)
- [ ] Detección automática de anomalías
- [ ] Dashboard interactivo con Streamlit
- [ ] Containerización con Docker
- [ ] Pipeline de CI/CD
- [ ] Monitoreo de deriva de datos
- [ ] Explicabilidad de modelos (SHAP)

---