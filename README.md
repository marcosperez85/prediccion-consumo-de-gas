# 🔮 Predicción de Consumo de Gas

Sistema de predicción de consumo de gas horario basado en machine learning, utilizando variables climáticas y patrones temporales.

## 🎯 Objetivo
Desarrollar un modelo predictivo de alta precisión para anticipar el consumo de gas horario, permitiendo optimizar la distribución y gestión de recursos energéticos.

## ✨ Características

- Predicción horaria de consumo de gas
- Análisis de factores climáticos (temperatura, humedad, viento)
- Incorporación de patrones temporales (día de semana, estacionalidad)
- API REST para integración con otros sistemas
- Comparativa de rendimiento entre diversos algoritmos

## 🏗️ Arquitectura del Proyecto

```
ml-gas-lab/
│
├── data/                      # Todos los datos del proyecto
│   ├── raw/                   # Datos sin procesar
│   │   └── gas_consumption.csv
│   └── processed/             # Datos preprocesados para modelado
│
├── notebooks/                 # Jupyter notebooks para análisis y experimentación
│   ├── 01_load.ipynb          # Carga y validación inicial de datos
│   ├── 02_eda.ipynb           # Análisis exploratorio de datos
│   ├── 03_feature_engineering.ipynb  # Creación de características
│   ├── 04_model.ipynb         # Entrenamiento y evaluación de modelos
│   └── 05_forecast.ipynb      # Generación y análisis de pronósticos
│
├── src/                       # Código fuente modularizado
│   ├── load.py                # Funciones para carga de datos
│   ├── eda.py                 # Funciones para análisis exploratorio
│   ├── feature_engineering.py # Transformación y creación de características
│   ├── train_model.py         # Entrenamiento de modelos
│   ├── model_compare.py       # Comparativa de modelos
│   └── main_api.py            # API REST para predicciones
│
├── models/                    # Modelos entrenados y serializados
│   ├── linear_regression_model.pkl
│   └── random_forest_model.pkl
│
├── tests/                     # Pruebas unitarias y de integración
│   ├── test_load.py
│   ├── test_model.py
│   └── test_api.py
│
├── docs/                      # Documentación adicional
│   ├── data_dictionary.md     # Descripción de variables
│   └── model_performance.md   # Resultados de evaluación de modelos
│
├── .env.example               # Plantilla para variables de entorno
├── requirements.txt           # Dependencias del proyecto
├── setup.py                   # Configuración para instalación como paquete
└── README.md                  # Documentación principal
```

## 🚀 Instalación y Uso

### Prerrequisitos
- Python 3.8+
- pip

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/marcosperez85/prediccion-consumo-de-gas.git
   cd ml-gas-lab
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Uso

#### Entrenamiento de modelos

```bash
python src/train_model.py --model random_forest --output models/rf_model.pkl
```

#### Comparación de modelos

```bash
python src/model_compare.py --models linear,random_forest,xgboost
```

#### API de predicción

1. Iniciar la API:
   ```bash
   uvicorn src.main_api:app --reload
   ```

2. Acceder a la documentación interactiva:
   ```
   http://localhost:8000/docs
   ```

## 📊 Resultados

| Modelo | RMSE | MAE | R² |
|--------|------|-----|---|
| Linear Regression | 12.45 | 10.21 | 0.75 |
| Random Forest | 8.32 | 6.78 | 0.86 |
| XGBoost | 7.14 | 5.92 | 0.89 |

## 🤝 Contribución

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Fork el repositorio
2. Crea una rama para tu característica (`git checkout -b feature/amazing-feature`)
3. Haz commit de tus cambios (`git commit -m 'Add some amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request