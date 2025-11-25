import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Solo para comparar modelos sin guardarlos
# python src/model_compare.py

# Para comparar y también guardar los modelos
# python src/model_compare.py --save

def load_and_prepare(path="../data/raw/gas_consumption.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["lag1"] = df["consumption_m3"].shift(1)
    # df["lag24"] = df["consumption_m3"].shift(24)
    df = df.dropna()
    X = df[["temp", "humidity", "wind", "hour", "weekday", "is_weekend", "lag1"]]
    y = df["consumption_m3"]
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def compare(save_models=False):
    """
    Compara diferentes modelos de regresión y muestra sus métricas.
    
    Args:
        save_models (bool): Si es True, guarda los modelos entrenados.
    """
    X_train, X_test, y_train, y_test = load_and_prepare()

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42)
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "model": name,
            "MAE": mae,
            "MSE": mse,
            "R2": r2
        })
        
        # Guardar modelo solo si se solicita explícitamente
        if save_models:
            model_path = f"../models/{name.lower()}_model.pkl"
            
            # Verificar si el modelo ya existe y mostrar advertencia
            if os.path.exists(model_path):
                print(f"Advertencia: Sobreescribiendo modelo existente en {model_path}")
                
            joblib.dump(model, model_path)
            print(f"Modelo {name} guardado en {model_path}")

    # Crear y mostrar tabla de resultados
    df_res = pd.DataFrame(results)
    print("\nComparación de modelos:")
    print(df_res)
    
    # Identificar el mejor modelo según R2
    best_model = df_res.loc[df_res["R2"].idxmax()]
    print(f"\nMejor modelo: {best_model['model']} con R² = {best_model['R2']:.4f}")
    
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compara diferentes modelos de regresión")
    parser.add_argument("--save", action="store_true", 
                       help="Guardar los modelos entrenados (por defecto: no guardar)")
    
    args = parser.parse_args()
    compare(save_models=args.save)