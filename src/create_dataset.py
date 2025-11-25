import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

def create_industrial_timeseries_dataset(
    start_date="2022-01-01", 
    periods=24*365*2,  # 2 años de datos horarios
    freq="H"
):
    """
    Genera un dataset sintético para series temporales industriales
    con patrones realistas y múltiples features correlacionadas.
    """
    
    # Generar timestamps
    rng = pd.date_range(start_date, periods=periods, freq=freq)
    
    # Feature 1: temperature (°C)
    # Estacionalidad anual + variación diaria + ruido
    annual_temp = 15 * np.sin(2 * np.pi * np.arange(len(rng)) / (24 * 365)) + 20
    daily_temp_var = 8 * np.sin(2 * np.pi * np.arange(len(rng)) / 24)
    temp_noise = np.random.normal(0, 2, len(rng))
    temperature = annual_temp + daily_temp_var + temp_noise
    
    # Feature 2: demand_factor (0-1)
    # Representa la demanda del mercado/producto
    weekly_demand = 0.3 * np.sin(2 * np.pi * np.arange(len(rng)) / (24 * 7))
    seasonal_demand = 0.2 * np.sin(2 * np.pi * np.arange(len(rng)) / (24 * 90))  # trimestral
    demand_trend = np.linspace(0.5, 0.8, len(rng))  # tendencia creciente
    demand_noise = np.random.normal(0, 0.05, len(rng))
    demand_factor = np.clip(demand_trend + weekly_demand + seasonal_demand + demand_noise, 0, 1)
    
    # Feature 3: operational_efficiency (0-1)
    # Eficiencia operacional con degradación por mantenimiento
    base_efficiency = 0.85
    # Degradación gradual cada 3 meses con mantenimientos
    maintenance_cycles = np.floor(np.arange(len(rng)) / (24 * 90))  # cada 90 días
    efficiency_degradation = -0.15 * (np.arange(len(rng)) % (24 * 90)) / (24 * 90)
    efficiency_recovery = 0.15 * (maintenance_cycles % 2)  # recuperación post-mantenimiento
    efficiency_noise = np.random.normal(0, 0.02, len(rng))
    operational_efficiency = np.clip(
        base_efficiency + efficiency_degradation + efficiency_recovery + efficiency_noise, 
        0.6, 1.0
    )
    
    # Feature 4: energy_price ($/MWh)
    # Precio de energía con volatilidad y patrones estacionales
    base_price = 80
    seasonal_price = 15 * np.sin(2 * np.pi * np.arange(len(rng)) / (24 * 365 / 4))  # trimestral
    daily_price_var = 10 * np.sin(2 * np.pi * np.arange(len(rng)) / 24)  # picos diarios
    price_volatility = np.random.normal(0, 5, len(rng))
    # Eventos de alta volatilidad (crisis, mantenimientos de red, etc.)
    volatility_events = np.zeros(len(rng))
    event_indices = np.random.choice(len(rng), size=50, replace=False)
    volatility_events[event_indices] = np.random.normal(0, 20, size=50)
    
    energy_price = np.clip(
        base_price + seasonal_price + daily_price_var + price_volatility + volatility_events,
        30, 200
    )
    
    # TARGET: value (producción/consumo industrial)
    # Combinación compleja de todas las features con patrones realistas
    
    # Componente base con tendencia
    base_production = np.linspace(1000, 1200, len(rng))
    
    # Impacto de temperatura (relación no lineal)
    temp_impact = -2 * (temperature - 20)**2 / 100 + 50
    
    # Impacto de demanda (relación directa)
    demand_impact = 300 * demand_factor
    
    # Impacto de eficiencia operacional
    efficiency_impact = 200 * operational_efficiency
    
    # Impacto de precio de energía (relación inversa)
    price_impact = -0.5 * (energy_price - 80)
    
    # Patrones temporales adicionales
    # Menor producción en fines de semana
    weekend_effect = -100 * np.isin(pd.to_datetime(rng).dayofweek, [5, 6])
    
    # Menor producción en horarios nocturnos
    night_effect = -50 * np.isin(pd.to_datetime(rng).hour, range(22, 24)) + \
                   -80 * np.isin(pd.to_datetime(rng).hour, range(0, 6))
    
    # Eventos anómalos (paradas, mantenimientos no programados)
    anomaly_events = np.zeros(len(rng))
    anomaly_indices = np.random.choice(len(rng), size=30, replace=False)
    anomaly_events[anomaly_indices] = np.random.normal(-200, 50, size=30)
    
    # Ruido final
    final_noise = np.random.normal(0, 15, len(rng))
    
    # Combinación final del target
    value = (base_production + temp_impact + demand_impact + efficiency_impact + 
             price_impact + weekend_effect + night_effect + anomaly_events + final_noise)
    
    # Asegurar valores positivos
    value = np.clip(value, 50, None)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'timestamp': rng,
        'value': value.round(2),
        'temperature': temperature.round(2),
        'demand_factor': demand_factor.round(4),
        'operational_efficiency': operational_efficiency.round(4),
        'energy_price': energy_price.round(2)
    })
    
    # Agregar features temporales derivadas (útiles para ML)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    return df

# Generar y guardar dataset
if __name__ == "__main__":
    print("Generando dataset industrial sintético...")
    
    # Crear dataset
    df = create_industrial_timeseries_dataset()
    
    # Estadísticas del dataset
    print(f"\nDataset generado:")
    print(f"- Períodos: {len(df):,} registros")
    print(f"- Rango temporal: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print(f"- Features principales: {list(df.columns[:6])}")
    
    # Guardar archivo
    output_path = "../data/raw/industrial_timeseries.csv"
    df.to_csv(output_path, index=False)
    print(f"- Guardado en: {output_path}")
    
    # Mostrar estadísticas descriptivas
    print(f"\nEstadísticas descriptivas:")
    print(df[['value', 'temperature', 'demand_factor', 
              'operational_efficiency', 'energy_price']].describe().round(2))
    
    print("\n✅ Dataset generado exitosamente!")