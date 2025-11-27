import yaml
import os
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Carga la configuración desde el archivo YAML.
    
    Args:
        config_path (str): Ruta al archivo de configuración
        
    Returns:
        Dict[str, Any]: Configuración cargada
    """
    # Buscar el archivo de config desde diferentes ubicaciones
    possible_paths = [
        config_path,
        f"../{config_path}",
        f"../../{config_path}",
        os.path.join(Path(__file__).parent.parent, config_path)
    ]
    
    config_file = None
    for path in possible_paths:
        if os.path.exists(path):
            config_file = path
            break
    
    if config_file is None:
        raise FileNotFoundError(f"Archivo de configuración no encontrado en: {possible_paths}")
    
    with open(config_file, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    return config

def get_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae configuración del dataset"""
    return config.get('dataset', {})

def get_feature_engineering_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae configuración de feature engineering"""
    return config.get('feature_engineering', {})

def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae configuración de entrenamiento"""
    return config.get('training', {})

def get_model_configs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extrae configuraciones de modelos"""
    training_config = get_training_config(config)
    return training_config.get('models', {})