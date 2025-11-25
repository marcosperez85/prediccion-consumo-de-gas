import pandas as pd
import numpy as np

np.random.seed(42)

# 1 año horario
rng = pd.date_range("2023-01-01", periods=24*365, freq="H")

# Tendencia
trend = np.linspace(100, 150, len(rng))

# Estacionalidad diaria
daily = 20 * np.sin(np.linspace(0, 24 * np.pi * 365 / 24, len(rng)))

# Estacionalidad semanal
weekly = 15 * np.sin(np.linspace(0, 2 * np.pi * 52, len(rng)))

# Ruido
noise = np.random.normal(0, 5, len(rng))

# Eventos aleatorios
events = np.zeros(len(rng))
indices = np.random.choice(len(rng), size=20, replace=False)
events[indices] += np.random.normal(40, 10, size=20)

value = trend + daily + weekly + noise + events

df = pd.DataFrame({"timestamp": rng, "value": value})
df.to_csv("../data/raw/dataset_sintetico.csv", index=False)
