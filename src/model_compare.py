
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def load_and_prepare(path="../data/raw/gas_consumption.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df["hour"] = df["date"].dt.hour
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["lag1"] = df["consumption_m3"].shift(1)
    df["lag24"] = df["consumption_m3"].shift(24)
    df = df.dropna()
    X = df[["temp", "humidity", "wind", "hour", "weekday", "is_weekend", "lag1", "lag24"]]
    y = df["consumption_m3"]
    return train_test_split(X, y, test_size=0.2, shuffle=False)

def compare():
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
        # Guardar modelo
        joblib.dump(model, f"../models/{name.lower()}_model.pkl")

    df_res = pd.DataFrame(results)
    print(df_res)

if __name__ == "__main__":
    compare()
