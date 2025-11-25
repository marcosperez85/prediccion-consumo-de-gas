import pandas as pd
import argparse
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Para el modelo de regresión lineal (valor por defecto)
# python src/train_model.py

# Para el modelo de Random Forest
# python src/train_model.py --model random_forest

def load_and_prepare_data():
    """Carga y prepara los datos para el entrenamiento."""
    df = pd.read_csv("../data/raw/gas_consumption.csv", parse_dates=["date"])
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["lag1"] = df["consumption_m3"].shift(1)
    # df["lag24"] = df["consumption_m3"].shift(24)
    df = df.dropna()

    # X = df[["temp", "humidity", "wind", "hour"]]
    X = df[["temp", "humidity", "wind", "hour", "weekday", "is_weekend", "lag1"]]
    y = df["consumption_m3"]
    
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def train(model_type="linear_regression"):
    """
    Entrena un modelo de predicción de consumo de gas.
    
    Args:
        model_type (str): Tipo de modelo a entrenar ('linear_regression' o 'random_forest')
    """
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Seleccionar el tipo de modelo
    if model_type.lower() == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model_filename = "../models/random_forest_model.pkl"
    else:  # default to linear regression
        model = LinearRegression()
        model_filename = "../models/linear_regression_model.pkl"
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Resultados del modelo {model_type}:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Guardar el modelo
    joblib.dump(model, model_filename)
    print(f"Modelo entrenado y guardado en {model_filename}")
    
    return model, mse, r2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrena un modelo para predecir consumo de gas")
    parser.add_argument("--model", type=str, default="linear_regression",
                      choices=["linear_regression", "random_forest"],
                      help="Tipo de modelo a entrenar (linear_regression o random_forest)")
    
    args = parser.parse_args()
    train(model_type=args.model)