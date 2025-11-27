import pandas as pd
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any

from config_loader import load_config, get_dataset_config

def load_raw(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Carga el dataset crudo según configuración.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        pd.DataFrame: Dataset cargado
    """
    config = load_config(config_path)
    dataset_config = get_dataset_config(config)
    
    # Construir ruta del archivo
    data_path = dataset_config.get('path')
    datetime_col = dataset_config.get('datetime_col')
    
    if not data_path:
        raise ValueError("Ruta del dataset no especificada en config.yaml")
    
    if not datetime_col:
        raise ValueError("Columna datetime no especificada en config.yaml")
    
    # Buscar archivo desde diferentes ubicaciones
    possible_paths = [
        data_path,
        f"../{data_path}",
        f"../../{data_path}",
        os.path.join(Path(__file__).parent.parent, data_path)
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError(f"Dataset no encontrado en ninguna de estas ubicaciones: {possible_paths}")
    
    # Cargar dataset
    try:
        df = pd.read_csv(file_path, parse_dates=[datetime_col])
    except Exception as e:
        raise Exception(f"Error cargando dataset desde {file_path}: {e}")
    
    print(f"✅ Dataset raw cargado desde: {file_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columnas: {list(df.columns)}")
    print(f"   Rango temporal: {df[datetime_col].min()} a {df[datetime_col].max()}")
    
    return df

def load_processed(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Carga el dataset procesado con features engineered.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        pd.DataFrame: Dataset procesado
    """
    config = load_config(config_path)
    dataset_config = get_dataset_config(config)
    
    # Construir ruta procesada
    raw_path = dataset_config.get('path')
    processed_path = raw_path.replace('/raw/', '/processed/').replace('.csv', '_featured.csv')
    datetime_col = dataset_config.get('datetime_col')
    
    # Buscar archivo desde diferentes ubicaciones
    possible_paths = [
        processed_path,
        f"../{processed_path}",
        f"../../{processed_path}",
        os.path.join(Path(__file__).parent.parent, processed_path)
    ]
    
    file_path = None
    for path in possible_paths:
        if os.path.exists(path):
            file_path = path
            break
    
    if file_path is None:
        raise FileNotFoundError(
            f"Dataset procesado no encontrado. "
            f"Ejecuta primero el notebook 03_feature_engineering.ipynb. "
            f"Buscado en: {possible_paths}"
        )
    
    try:
        df = pd.read_csv(file_path, parse_dates=[datetime_col])
    except Exception as e:
        raise Exception(f"Error cargando dataset procesado desde {file_path}: {e}")
    
    print(f"✅ Dataset procesado cargado desde: {file_path}")
    print(f"   Shape: {df.shape}")
    
    return df

def get_feature_target_columns(config_path: str = "config.yaml") -> Tuple[List[str], str]:
    """
    Obtiene las columnas de features y target según configuración.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        Tuple[List[str], str]: (feature_columns, target_column)
    """
    config = load_config(config_path)
    dataset_config = get_dataset_config(config)
    
    feature_columns = dataset_config.get('raw_feature_columns', [])
    target_column = dataset_config.get('target_col')
    
    if not feature_columns:
        raise ValueError("Features no especificadas en config.yaml -> dataset -> raw_feature_columns")
    
    if not target_column:
        raise ValueError("Target no especificado en config.yaml -> dataset -> target_col")
    
    return feature_columns, target_column

# Funciones de utilidad
def validate_dataset_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Valida la configuración del dataset.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        Dict con resultados de validación
    """
    try:
        config = load_config(config_path)
        dataset_config = get_dataset_config(config)
        
        validation = {
            'config_loaded': True,
            'path_specified': bool(dataset_config.get('path')),
            'datetime_col_specified': bool(dataset_config.get('datetime_col')),
            'target_col_specified': bool(dataset_config.get('target_col')),
            'raw_features_specified': bool(dataset_config.get('raw_feature_columns')),
            'raw_features_count': len(dataset_config.get('raw_feature_columns', [])),
            'file_exists': False,
            'processed_file_exists': False
        }
        
        # Verificar si archivos existen
        if validation['path_specified']:
            raw_path = dataset_config.get('path')
            processed_path = raw_path.replace('/raw/', '/processed/').replace('.csv', '_featured.csv')
            
            # Buscar archivo raw
            for prefix in ["", "../", "../../"]:
                if os.path.exists(f"{prefix}{raw_path}"):
                    validation['file_exists'] = True
                    break
            
            # Buscar archivo procesado
            for prefix in ["", "../", "../../"]:
                if os.path.exists(f"{prefix}{processed_path}"):
                    validation['processed_file_exists'] = True
                    break
        
        validation['is_valid'] = all([
            validation['config_loaded'],
            validation['path_specified'],
            validation['datetime_col_specified'],
            validation['target_col_specified'],
            validation['raw_features_specified'],
            validation['file_exists']
        ])
        
        return validation
        
    except Exception as e:
        return {
            'config_loaded': False,
            'error': str(e),
            'is_valid': False
        }

if __name__ == "__main__":
    print("🔍 VALIDACIÓN DE CONFIGURACIÓN DE DATOS")
    print("=" * 50)
    
    # Validar configuración
    validation = validate_dataset_config()
    
    print("📋 Resultados de validación:")
    for key, value in validation.items():
        if key != 'error':
            icon = "✅" if value else "❌"
            print(f"   {key}: {icon} {value}")
    
    if 'error' in validation:
        print(f"❌ Error: {validation['error']}")
    
    if validation.get('is_valid'):
        print(f"\n✅ Configuración válida - probando carga...")
        try:
            df = load_raw()
            print(f"   Dataset cargado exitosamente: {df.shape}")
        except Exception as e:
            print(f"   ❌ Error cargando dataset: {e}")
    else:
        print(f"\n❌ Configuración inválida")