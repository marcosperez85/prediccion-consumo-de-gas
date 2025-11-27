import pandas as pd
import numpy as np
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

from config_loader import load_config, get_training_config, get_model_configs
from data_loader import load_processed
from features.feature_engineering import FeatureEngineeringEngine

def load_and_prepare(config_path="config.yaml"):
    """
    Carga y prepara los datos para comparación de modelos según configuración.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_columns, target_col
    """
    print(f"🔄 Cargando configuración desde: {config_path}")
    config = load_config(config_path)
    training_config = get_training_config(config)
    
    # Cargar datos procesados
    df = load_processed(config_path)
    
    # Usar feature engineering engine para obtener columnas
    fe_engine = FeatureEngineeringEngine(config_path)
    feature_columns = fe_engine.get_feature_columns(df)
    target_col = fe_engine.target_col
    
    print(f"📊 Dataset: {df.shape}")
    print(f"🎯 Features: {len(feature_columns)} columnas")
    print(f"🎯 Target: {target_col}")
    
    X = df[feature_columns]
    y = df[target_col]
    
    # Split temporal según configuración
    test_ratio = training_config.get('test_ratio', 0.2)
    split_idx = int(len(df) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"📈 Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    return X_train, X_test, y_train, y_test, feature_columns, target_col

def calculate_mape(y_true, y_pred):
    """Calcula Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_models_from_config(config_path="config.yaml"):
    """
    Crea modelos basados en la configuración.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        dict: Diccionario con modelos configurados
    """
    config = load_config(config_path)
    model_configs = get_model_configs(config)
    
    models = {}
    
    # Crear modelos según configuración
    for model_name, model_config in model_configs.items():
        if model_name == 'random_forest':
            models['Random Forest'] = RandomForestRegressor(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', None),
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'gradient_boosting':
            models['Gradient Boosting'] = GradientBoostingRegressor(
                n_estimators=model_config.get('n_estimators', 100),
                learning_rate=model_config.get('learning_rate', 0.1),
                random_state=42
            )
    
    # Siempre incluir Linear Regression como baseline
    models['Linear Regression'] = LinearRegression()
    
    return models

def compare(config_path="config.yaml", save_models=False):
    """
    Compara diferentes modelos según configuración.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        save_models (bool): Si es True, guarda los modelos entrenados.
        
    Returns:
        pd.DataFrame: Resultados de la comparación
    """
    print("🤖 COMPARACIÓN AUTOMÁTICA DE MODELOS")
    print("=" * 50)
    
    # Cargar configuración
    config = load_config(config_path)
    model_configs = get_model_configs(config)
    
    print("📋 Modelos configurados:")
    for model_name, model_config in model_configs.items():
        print(f"   • {model_name}: {model_config}")
    
    # Preparar datos
    X_train, X_test, y_train, y_test, feature_columns, target_col = load_and_prepare(config_path)
    
    # Crear modelos según configuración
    models = create_models_from_config(config_path)
    
    print(f"\n🔄 Entrenando {len(models)} modelos...")
    
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\n   🔄 Entrenando {name}...")
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Métricas
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mape = calculate_mape(y_test, y_pred_test)

        results.append({
            "Model": name,
            "Train_MAE": train_mae,
            "Test_MAE": test_mae,
            "Train_RMSE": train_rmse,
            "Test_RMSE": test_rmse,
            "Train_R2": train_r2,
            "Test_R2": test_r2,
            "Test_MAPE": test_mape
        })
        
        trained_models[name] = model
        
        print(f"      ✅ MAE: {test_mae:.2f} | R²: {test_r2:.4f} | MAPE: {test_mape:.1f}%")
        
        # Guardar modelo individual si se solicita
        if save_models:
            os.makedirs("models", exist_ok=True)
            model_filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
            
            if os.path.exists(model_filename):
                print(f"      ⚠️  Sobreescribiendo: {model_filename}")
                
            joblib.dump(model, model_filename)
            print(f"      💾 Guardado: {model_filename}")

    # Crear DataFrame de resultados
    df_results = pd.DataFrame(results)
    
    print("\n📊 COMPARACIÓN DE MODELOS:")
    print("=" * 80)
    
    # Mostrar tabla formateada
    display_df = df_results.copy()
    numeric_cols = ['Train_MAE', 'Test_MAE', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2', 'Test_MAPE']
    for col in numeric_cols:
        if col in ['Train_R2', 'Test_R2']:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
        elif col == 'Test_MAPE':
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}%")
        else:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
    
    print(display_df.to_string(index=False))
    
    # Identificar mejores modelos
    best_mae = df_results.loc[df_results["Test_MAE"].idxmin()]
    best_r2 = df_results.loc[df_results["Test_R2"].idxmax()]
    best_mape = df_results.loc[df_results["Test_MAPE"].idxmin()]
    
    print(f"\n🏆 MEJORES MODELOS:")
    print(f"  • Menor MAE: {best_mae['Model']} (MAE: {best_mae['Test_MAE']:.2f})")
    print(f"  • Mayor R²: {best_r2['Model']} (R²: {best_r2['Test_R2']:.4f})")
    print(f"  • Menor MAPE: {best_mape['Model']} (MAPE: {best_mape['Test_MAPE']:.2f}%)")
    
    # Análisis de overfitting
    print(f"\n🔍 ANÁLISIS DE OVERFITTING:")
    for _, row in df_results.iterrows():
        mae_diff = ((row['Train_MAE'] - row['Test_MAE']) / row['Test_MAE']) * 100
        r2_diff = ((row['Train_R2'] - row['Test_R2']) / row['Test_R2']) * 100
        
        status = "✅ OK" if abs(mae_diff) <= 15 and abs(r2_diff) <= 15 else "⚠️  Overfitting"
        print(f"   {row['Model']}: MAE {mae_diff:+.1f}%, R² {r2_diff:+.1f}% - {status}")
    
    # Guardar mejor modelo automáticamente si se solicita
    if save_models:
        # Usar MAE como criterio principal
        best_model_name = best_mae['Model']
        best_model = trained_models[best_model_name]
        
        # Información del modelo
        model_info = {
            'features': feature_columns,
            'model_type': best_model_name,
            'test_mae': best_mae['Test_MAE'],
            'test_r2': best_mae['Test_R2'],
            'test_rmse': best_mae['Test_RMSE'],
            'test_mape': best_mae['Test_MAPE'],
            'train_mae': best_mae['Train_MAE'],
            'train_r2': best_mae['Train_R2'],
            'feature_count': len(feature_columns)
        }
        
        joblib.dump(best_model, "models/best_model.pkl")
        joblib.dump(model_info, "models/model_info.pkl")
        
        print(f"\n✅ MEJOR MODELO GUARDADO:")
        print(f"   • Modelo: {best_model_name}")
        print(f"   • Archivos: models/best_model.pkl, models/model_info.pkl")
        print(f"   • Test MAE: {best_mae['Test_MAE']:.2f}")
        print(f"   • Test R²: {best_mae['Test_R2']:.4f}")
    
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compara modelos según configuración en config.yaml"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                      help="Ruta al archivo de configuración")
    parser.add_argument("--save", action="store_true", 
                       help="Guardar los modelos entrenados")
    
    args = parser.parse_args()
    
    try:
        results = compare(config_path=args.config, save_models=args.save)
        print(f"\n✅ Comparación completada exitosamente")
        print(f"   Configuración usada: {args.config}")
        print(f"   Modelos evaluados: {len(results)}")
    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado.")
        print(f"   Verifica que existe: {args.config}")
        print(f"   Ejecuta primero: notebook 03_feature_engineering.ipynb")
        print(f"   Detalles: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")