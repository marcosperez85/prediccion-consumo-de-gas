import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import os
from pathlib import Path

# Cambiar imports relativos a absolutos
from config_loader import load_config, get_dataset_config, get_feature_engineering_config

class FeatureEngineeringEngine:
    """
    Motor de feature engineering configurable para series temporales.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa el motor de feature engineering.
        
        Args:
            config_path (str): Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.dataset_config = get_dataset_config(self.config)
        self.fe_config = get_feature_engineering_config(self.config)
        
        self.datetime_col = self.dataset_config.get('datetime_col')
        self.target_col = self.dataset_config.get('target_col')
        self.raw_features = self.dataset_config.get('raw_feature_columns', [])
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features temporales."""
        if not self.fe_config.get('time_features', False):
            return df
        
        print("🕐 Creando features temporales...")
        df_copy = df.copy()
        
        # Asegurar que datetime_col es datetime
        df_copy[self.datetime_col] = pd.to_datetime(df_copy[self.datetime_col])
        
        # Features temporales estándar
        df_copy['hour'] = df_copy[self.datetime_col].dt.hour
        df_copy['day_of_week'] = df_copy[self.datetime_col].dt.dayofweek
        df_copy['month'] = df_copy[self.datetime_col].dt.month
        df_copy['quarter'] = df_copy[self.datetime_col].dt.quarter
        df_copy['is_weekend'] = (df_copy['day_of_week'] >= 5).astype(int)
        
        print(f"   ✅ Creadas: hour, day_of_week, month, quarter, is_weekend")
        return df_copy
    
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de lag."""
        lags = self.fe_config.get('lags', [])
        if not lags:
            return df
        
        print(f"⏰ Creando features de lag: {lags}...")
        df_copy = df.copy()
        
        for lag in lags:
            col_name = f'lag_{lag}h'
            df_copy[col_name] = df_copy[self.target_col].shift(lag)
            print(f"   ✅ Creado: {col_name}")
        
        return df_copy
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de rolling statistics."""
        rolling_config = self.fe_config.get('rolling', {})
        if not rolling_config:
            return df
        
        windows = rolling_config.get('windows', [])
        functions = rolling_config.get('functions', [])
        
        if not windows or not functions:
            return df
        
        print(f"📊 Creando rolling features: windows={windows}, functions={functions}...")
        df_copy = df.copy()
        
        for window in windows:
            for func in functions:
                col_name = f'rolling_{func}_{window}h'
                if func == 'mean':
                    df_copy[col_name] = df_copy[self.target_col].rolling(window=window).mean()
                elif func == 'std':
                    df_copy[col_name] = df_copy[self.target_col].rolling(window=window).std()
                elif func == 'min':
                    df_copy[col_name] = df_copy[self.target_col].rolling(window=window).min()
                elif func == 'max':
                    df_copy[col_name] = df_copy[self.target_col].rolling(window=window).max()
                
                print(f"   ✅ Creado: {col_name}")
        
        return df_copy
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea features de interacción."""
        interactions = self.fe_config.get('interactions', [])
        if not interactions:
            return df
        
        print(f"🔗 Creando features de interacción...")
        df_copy = df.copy()
        
        for interaction in interactions:
            if len(interaction) == 2:
                col1, col2 = interaction
                col_name = f'{col1}_{col2}_interaction'
                df_copy[col_name] = df_copy[col1] * df_copy[col2]
                print(f"   ✅ Creado: {col_name}")
        
        return df_copy
    
    def create_squared_terms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea términos cuadráticos."""
        squared_terms = self.fe_config.get('squared_terms', [])
        if not squared_terms:
            return df
        
        print(f"² Creando términos cuadráticos: {squared_terms}...")
        df_copy = df.copy()
        
        for col in squared_terms:
            if col in df_copy.columns:
                col_name = f'{col}_squared'
                df_copy[col_name] = df_copy[col] ** 2
                print(f"   ✅ Creado: {col_name}")
        
        return df_copy
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas las transformaciones de feature engineering.
        
        Args:
            df (pd.DataFrame): Dataset original
            
        Returns:
            pd.DataFrame: Dataset con features engineered
        """
        print("🚀 INICIANDO FEATURE ENGINEERING")
        print("=" * 50)
        
        # Aplicar transformaciones en orden
        df_transformed = df.copy()
        
        # 1. Features temporales
        df_transformed = self.create_time_features(df_transformed)
        
        # 2. Lag features
        df_transformed = self.create_lag_features(df_transformed)
        
        # 3. Rolling features
        df_transformed = self.create_rolling_features(df_transformed)
        
        # 4. Interaction features
        df_transformed = self.create_interaction_features(df_transformed)
        
        # 5. Squared terms
        df_transformed = self.create_squared_terms(df_transformed)
        
        # Limpiar NaN
        initial_rows = len(df_transformed)
        df_transformed = df_transformed.dropna().reset_index(drop=True)
        final_rows = len(df_transformed)
        
        print(f"\n🧹 Limpieza de datos:")
        print(f"   Filas iniciales: {initial_rows:,}")
        print(f"   Filas finales: {final_rows:,}")
        print(f"   Filas removidas: {initial_rows - final_rows:,}")
        
        print(f"\n✅ FEATURE ENGINEERING COMPLETADO")
        print(f"   Features totales: {len(df_transformed.columns)}")
        print(f"   Features nuevas: {len(df_transformed.columns) - len(df.columns)}")
        
        return df_transformed
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Obtiene las columnas de features para modelado (excluye datetime y target).
        
        Args:
            df (pd.DataFrame): Dataset con features
            
        Returns:
            List[str]: Lista de columnas de features
        """
        exclude_cols = {self.datetime_col, self.target_col}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def save_processed_data(self, df: pd.DataFrame) -> str:
        """
        Guarda el dataset procesado.
        
        Args:
            df (pd.DataFrame): Dataset procesado
            
        Returns:
            str: Ruta donde se guardó el archivo
        """
        # Construir ruta de salida
        raw_path = self.dataset_config.get('path')
        processed_path = raw_path.replace('/raw/', '/processed/').replace('.csv', '_featured.csv')
        
        # Crear directorio si no existe
        processed_dir = os.path.dirname(processed_path)
        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)
        
        # Buscar la ubicación correcta para guardar
        possible_base_paths = [
            "",
            "../",
            "../../",
            str(Path(__file__).parent.parent.parent)
        ]
        
        save_path = None
        for base_path in possible_base_paths:
            full_path = os.path.join(base_path, processed_path)
            directory = os.path.dirname(full_path)
            
            if os.path.exists(directory) or base_path == "":
                if not os.path.exists(directory):
                    os.makedirs(directory)
                save_path = full_path
                break
        
        if save_path is None:
            save_path = processed_path
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Guardar archivo
        df.to_csv(save_path, index=False)
        
        print(f"\n💾 Dataset procesado guardado en: {save_path}")
        print(f"   Forma: {df.shape}")
        
        return save_path