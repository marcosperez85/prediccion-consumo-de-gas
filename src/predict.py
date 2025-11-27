import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Any, Union

from config_loader import load_config
from features.feature_engineering import FeatureEngineeringEngine

def predict_single(
    input_data: Dict[str, Any],
    model_path: str = "models/best_model.pkl",
    model_info_path: str = "models/model_info.pkl",
    config_path: str = "config.yaml"
) -> Dict[str, Any]:
    """
    Realiza una predicción individual usando el modelo entrenado.
    
    Args:
        input_data: Diccionario con las features de entrada
        model_path: Ruta al modelo guardado
        model_info_path: Ruta a la información del modelo
        config_path: Ruta al archivo de configuración
        
    Returns:
        Dict con la predicción y metadatos
    """
    try:
        # Cargar configuración
        config = load_config(config_path)
        fe_engine = FeatureEngineeringEngine(config_path)
        
        # Cargar modelo y configuración
        model = joblib.load(model_path)
        model_info = joblib.load(model_info_path)
        
        # Verificar features requeridas vs proporcionadas
        raw_features = fe_engine.raw_features
        missing_raw = set(raw_features) - set(input_data.keys())
        
        if missing_raw:
            print(f"⚠️  Features básicas faltantes: {missing_raw}")
            print(f"   Se usarán valores por defecto")
        
        # Crear DataFrame con los datos de entrada
        input_df = pd.DataFrame([input_data])
        
        # Aplicar feature engineering si es necesario
        if any(col not in input_df.columns for col in model_info['features']):
            print("🔧 Aplicando feature engineering...")
            
            # Agregar columna temporal dummy si no existe
            datetime_col = config['dataset']['datetime_col']
            if datetime_col not in input_df.columns:
                input_df[datetime_col] = pd.Timestamp.now()
            
            # Agregar target dummy
            target_col = fe_engine.target_col
            if target_col not in input_df.columns:
                input_df[target_col] = 0
            
            # Aplicar transformaciones
            input_df_transformed = fe_engine.transform(input_df)
        else:
            input_df_transformed = input_df
        
        # Verificar que todas las features del modelo están presentes
        missing_features = set(model_info['features']) - set(input_df_transformed.columns)
        if missing_features:
            print(f"⚠️  Features del modelo faltantes: {missing_features}")
            # Rellenar con valores por defecto
            for feature in missing_features:
                if 'lag_' in feature:
                    input_df_transformed[feature] = input_data.get(fe_engine.target_col, 1000)
                elif 'rolling_' in feature:
                    if 'mean' in feature:
                        input_df_transformed[feature] = input_data.get(fe_engine.target_col, 1000)
                    elif 'std' in feature:
                        input_df_transformed[feature] = 20.0
                elif '_squared' in feature:
                    base_col = feature.replace('_squared', '')
                    input_df_transformed[feature] = input_data.get(base_col, 0) ** 2
                elif '_interaction' in feature:
                    input_df_transformed[feature] = 0.5  # Valor neutral
                else:
                    input_df_transformed[feature] = 0
        
        # Preparar features en el orden correcto
        X_pred = input_df_transformed[model_info['features']]
        
        # Realizar predicción
        prediction = model.predict(X_pred)[0]
        
        return {
            'prediction': round(prediction, 2),
            'model_type': model_info['model_type'],
            'model_mae': round(model_info['test_mae'], 2),
            'model_r2': round(model_info['test_r2'], 4),
            'features_used': model_info['features'],
            'input_features': input_data,
            'features_count': len(model_info['features'])
        }
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Modelo no encontrado. Ejecuta primero el entrenamiento. Detalles: {e}")
    except Exception as e:
        raise Exception(f"Error en la predicción: {e}")

def predict_batch(
    df: pd.DataFrame,
    model_path: str = "models/best_model.pkl",
    model_info_path: str = "models/model_info.pkl",
    config_path: str = "config.yaml"
) -> pd.DataFrame:
    """
    Realiza predicciones en lote para un DataFrame.
    
    Args:
        df: DataFrame con las features necesarias
        model_path: Ruta al modelo guardado
        model_info_path: Ruta a la información del modelo
        config_path: Ruta al archivo de configuración
        
    Returns:
        DataFrame original con columna 'prediction' añadida
    """
    try:
        # Cargar modelo y configuración
        model = joblib.load(model_path)
        model_info = joblib.load(model_info_path)
        
        # Verificar que todas las features necesarias están presentes
        missing_features = set(model_info['features']) - set(df.columns)
        if missing_features:
            raise ValueError(f"Features faltantes en el DataFrame: {missing_features}")
        
        # Seleccionar features en el orden correcto
        X = df[model_info['features']]
        
        # Realizar predicciones
        predictions = model.predict(X)
        
        # Añadir predicciones al DataFrame
        result_df = df.copy()
        result_df['prediction'] = predictions.round(2)
        
        return result_df
        
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Modelo no encontrado. Ejecuta primero el entrenamiento. Detalles: {e}")
    except Exception as e:
        raise Exception(f"Error en predicción batch: {e}")

def predict_from_config_example(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Realiza una predicción de ejemplo usando valores por defecto basados en la configuración.
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Dict con la predicción de ejemplo
    """
    try:
        # Cargar configuración
        config = load_config(config_path)
        raw_features = config['dataset']['raw_feature_columns']
        
        # Crear datos de ejemplo (valores típicos)
        example_data = {}
        
        for feature in raw_features:
            if 'temperature' in feature.lower():
                example_data[feature] = 22.5
            elif 'demand' in feature.lower():
                example_data[feature] = 0.75
            elif 'efficiency' in feature.lower():
                example_data[feature] = 0.85
            elif 'price' in feature.lower():
                example_data[feature] = 85.0
            else:
                example_data[feature] = 1.0  # Valor por defecto
        
        # Añadir features temporales por defecto
        example_data.update({
            'hour': 14,
            'day_of_week': 2,
            'month': 6,
            'is_weekend': 0
        })
        
        print("📋 Datos de ejemplo generados automáticamente:")
        for key, value in example_data.items():
            print(f"   {key}: {value}")
        
        # Realizar predicción
        result = predict_single(example_data, config_path=config_path)
        
        return result
        
    except Exception as e:
        raise Exception(f"Error en predicción de ejemplo: {e}")

# Funciones de utilidad
def load_model_info(
    model_info_path: str = "models/model_info.pkl"
) -> Dict[str, Any]:
    """
    Carga información del modelo.
    
    Args:
        model_info_path: Ruta al archivo de información del modelo
        
    Returns:
        Dict con información del modelo
    """
    return joblib.load(model_info_path)

def get_required_features(
    model_info_path: str = "models/model_info.pkl"
) -> list:
    """
    Obtiene la lista de features requeridas por el modelo.
    
    Args:
        model_info_path: Ruta al archivo de información del modelo
        
    Returns:
        Lista de features requeridas
    """
    model_info = load_model_info(model_info_path)
    return model_info['features']

# Ejemplo de uso y testing
if __name__ == "__main__":
    print("🔮 SISTEMA DE PREDICCIONES CONFIGURABLE")
    print("=" * 50)
    
    try:
        # Predicción de ejemplo
        print("\n1️⃣  PREDICCIÓN DE EJEMPLO:")
        result = predict_from_config_example()
        
        print(f"\n📊 RESULTADO:")
        print(f"   Predicción: {result['prediction']}")
        print(f"   Modelo: {result['model_type']}")
        print(f"   MAE del modelo: {result['model_mae']}")
        print(f"   R² del modelo: {result['model_r2']}")
        
        # Mostrar features requeridas
        print(f"\n2️⃣  FEATURES REQUERIDAS:")
        try:
            required_features = get_required_features()
            print(f"   Total: {len(required_features)} features")
            print(f"   Primeras 10: {required_features[:10]}")
        except:
            print("   No se pudo cargar información del modelo")
        
        print(f"\n✅ Test completado exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en el test: {e}")
        print(f"   Ejecuta primero: python model_compare.py --save")