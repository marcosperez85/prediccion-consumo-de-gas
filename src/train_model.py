import pandas as pd
import numpy as np
import argparse
import joblib
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config_loader import load_config, get_training_config, get_model_configs
from data_loader import load_processed
from features.feature_engineering import FeatureEngineeringEngine

def calculate_mape(y_true, y_pred):
    """Calcula Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_model_from_config(model_name, model_config):
    """
    Crea un modelo basado en la configuración.
    
    Args:
        model_name (str): Nombre del modelo
        model_config (dict): Configuración del modelo
        
    Returns:
        sklearn model: Modelo configurado
    """
    if model_name == 'random_forest':
        return RandomForestRegressor(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', None),
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'gradient_boosting':
        return GradientBoostingRegressor(
            n_estimators=model_config.get('n_estimators', 100),
            learning_rate=model_config.get('learning_rate', 0.1),
            random_state=42
        )
    elif model_name == 'linear_regression':
        return LinearRegression()
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

def load_and_prepare_data(config_path="config.yaml"):
    """
    Carga y prepara los datos para el entrenamiento.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_columns, target_col
    """
    print(f"🔄 Cargando configuración desde: {config_path}")
    config = load_config(config_path)
    training_config = get_training_config(config)
    
    print(f"📊 Cargando datos procesados...")
    df = load_processed(config_path)
    
    # Usar feature engineering engine para obtener columnas
    fe_engine = FeatureEngineeringEngine(config_path)
    feature_columns = fe_engine.get_feature_columns(df)
    target_col = fe_engine.target_col
    
    print(f"🎯 Configuración de modelado:")
    print(f"   Features: {len(feature_columns)} columnas")
    print(f"   Target: {target_col}")
    print(f"   Dataset shape: {df.shape}")
    
    X = df[feature_columns]
    y = df[target_col]
    
    # Split temporal según configuración
    test_ratio = training_config.get('test_ratio', 0.2)
    split_idx = int(len(df) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📈 División de datos:")
    print(f"   Entrenamiento: {len(X_train):,} registros ({(1-test_ratio)*100:.0f}%)")
    print(f"   Prueba: {len(X_test):,} registros ({test_ratio*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test, feature_columns, target_col

def train(model_type, config_path="config.yaml", save_model=True):
    """
    Entrena un modelo específico según configuración.
    
    Args:
        model_type (str): Tipo de modelo a entrenar
        config_path (str): Ruta al archivo de configuración  
        save_model (bool): Si guardar el modelo entrenado
        
    Returns:
        tuple: (model, metrics_dict)
    """
    try:
        # Cargar configuración
        config = load_config(config_path)
        model_configs = get_model_configs(config)
        
        print(f"🚀 ENTRENAMIENTO DE MODELO - {model_type.upper()}")
        print("=" * 60)
        
        # Cargar y preparar datos
        X_train, X_test, y_train, y_test, feature_columns, target_col = load_and_prepare_data(config_path)
        
        # Crear modelo según configuración
        if model_type in model_configs:
            model_config = model_configs[model_type]
            print(f"🤖 Configuración del modelo {model_type}:")
            for key, value in model_config.items():
                print(f"   {key}: {value}")
        else:
            model_config = {}
            print(f"⚠️  Modelo {model_type} no está en config.yaml, usando configuración por defecto")
        
        model = create_model_from_config(model_type, model_config)
        
        # Entrenar el modelo
        print(f"\n🔄 Entrenando {model_type}...")
        model.fit(X_train, y_train)
        
        # Realizar predicciones
        print("📊 Evaluando modelo...")
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calcular métricas
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mape = calculate_mape(y_test, y_pred_test)
        
        # Mostrar resultados
        print(f"\n📈 RESULTADOS DEL MODELO:")
        print("=" * 40)
        print(f"  ENTRENAMIENTO:")
        print(f"    MAE:  {train_mae:.2f}")
        print(f"    RMSE: {train_rmse:.2f}")
        print(f"    R²:   {train_r2:.4f}")
        print(f"  ")
        print(f"  PRUEBA:")
        print(f"    MAE:  {test_mae:.2f}")
        print(f"    RMSE: {test_rmse:.2f}")
        print(f"    R²:   {test_r2:.4f}")
        print(f"    MAPE: {test_mape:.2f}%")
        
        # Análisis de overfitting
        overfitting_mae = ((train_mae - test_mae) / test_mae) * 100
        overfitting_r2 = ((train_r2 - test_r2) / test_r2) * 100
        
        print(f"\n🔍 ANÁLISIS DE OVERFITTING:")
        print(f"    Diferencia MAE: {overfitting_mae:+.1f}%")
        print(f"    Diferencia R²: {overfitting_r2:+.1f}%")
        
        if abs(overfitting_mae) > 15 or abs(overfitting_r2) > 15:
            print("    ⚠️  Posible overfitting detectado")
        else:
            print("    ✅ Modelo generaliza bien")
        
        # Guardar modelo si se solicita
        if save_model:
            os.makedirs("models", exist_ok=True)
            
            # Nombres de archivo
            model_filename = f"models/{model_type}_model.pkl"
            info_filename = f"models/{model_type}_info.pkl"
            
            # Guardar modelo
            joblib.dump(model, model_filename)
            
            # Información del modelo
            model_info = {
                'features': feature_columns,
                'model_type': model_type,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'test_mape': test_mape,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'feature_count': len(feature_columns),
                'config_used': model_config
            }
            
            joblib.dump(model_info, info_filename)
            
            print(f"\n💾 MODELO GUARDADO:")
            print(f"    Modelo: {model_filename}")
            print(f"    Info: {info_filename}")
        
        metrics = {
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mape': test_mape,
            'train_mae': train_mae,
            'train_r2': train_r2
        }
        
        return model, metrics
        
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado.")
        print(f"   Asegúrate de haber ejecutado primero el notebook 03_feature_engineering.ipynb")
        print(f"   Detalles: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrena un modelo para predicción de series temporales según config.yaml"
    )
    parser.add_argument("--model", type=str, default="random_forest",
                      choices=["linear_regression", "random_forest", "gradient_boosting"],
                      help="Tipo de modelo a entrenar")
    parser.add_argument("--config", type=str, default="config.yaml",
                      help="Ruta al archivo de configuración")
    parser.add_argument("--no-save", action="store_true",
                      help="No guardar el modelo entrenado")
    
    args = parser.parse_args()
    
    print("🚀 ENTRENAMIENTO DE MODELO - SERIE TEMPORAL CONFIGURABLE")
    print("=" * 70)
    
    model, metrics = train(
        model_type=args.model, 
        config_path=args.config,
        save_model=not args.no_save
    )
    
    if model is not None:
        print(f"\n✅ Entrenamiento completado exitosamente!")
        print(f"   Modelo: {args.model}")
        print(f"   Test MAE: {metrics['test_mae']:.2f}")
        print(f"   Test R²: {metrics['test_r2']:.4f}")
    else:
        print(f"\n❌ Entrenamiento falló.")