# Predicción de Consumo de Gas

## 🎯 Objetivo  
Desarrollar un modelo de machine learning para predecir el consumo de gas horario a partir de variables como temperatura, humedad, viento, historial (lags), día de la semana, etc.

## 📁 Estructura del proyecto

ml-gas-lab/
│
├── data/
│ ├── raw/
│ │ └── gas_consumption.csv
│ └── processed/
│
├── notebooks/
│ ├── 01_load.ipynb
│ ├── 02_eda.ipynb
│ ├── FE_feature_engineering.ipynb
│ ├── 03_model.ipynb
│ └── 04_forecast.ipynb
│
├── src/
│ ├── load.py
│ ├── eda.py
│ ├── train_linear_regression.py
│ ├── model_compare.py
│ └── main_api.py
│
├── models/
│ ├── linear_regression_model.pkl
│ └── randomforest_model.pkl
│
└── requirements.txt


## 🧪 Cómo usar

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt


2. Entrenar el modelo
   ```bash
    python src/train.py

3. Comparar modelos
    ```bash
    python src/model_compare.py

4. Ejecutar la API para hacer predicciones desde el root del proyecto:
    ```bash
    uvicorn src.main_api:app --reload

5. Documentación interactiva en:
    ```bash
    http://localhost:8000/docs