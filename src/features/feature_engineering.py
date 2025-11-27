import pandas as pd

class FeatureEngineeringEngine:
    def __init__(self, config, target_col):
        self.cfg = config
        self.target = target_col

    # ------------------------------------------
    # Time-based features
    # ------------------------------------------
    def add_time_features(self, df):
        if not self.cfg.get("time_features", False):
            return df
        
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["month"] = df.index.month
        df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
        return df

    # ------------------------------------------
    # Lags
    # ------------------------------------------
    def add_lags(self, df):
        for lag in self.cfg["lags"]:
            df[f"{self.target}_lag_{lag}"] = df[self.target].shift(lag)
        return df

    # ------------------------------------------
    # Rolling statistics
    # ------------------------------------------
    def add_rolling(self, df):
        windows = self.cfg["rolling"]["windows"]
        funcs = self.cfg["rolling"]["functions"]

        for win in windows:
            for func in funcs:
                if func == "mean":
                    df[f"{self.target}_roll_mean_{win}"] = df[self.target].rolling(win).mean()
                elif func == "std":
                    df[f"{self.target}_roll_std_{win}"] = df[self.target].rolling(win).std()
        return df

    # ------------------------------------------
    # Interaction terms
    # ------------------------------------------
    def add_interactions(self, df):
        for a, b in self.cfg["interactions"]:
            df[f"{a}_x_{b}"] = df[a] * df[b]
        return df

    # ------------------------------------------
    # Squared terms
    # ------------------------------------------
    def add_squared(self, df):
        for col in self.cfg["squared_terms"]:
            df[f"{col}_sq"] = df[col] ** 2
        return df

    # ------------------------------------------
    # Master pipeline
    # ------------------------------------------
    def run(self, df):
        df = df.copy()
        df = self.add_time_features(df)
        df = self.add_lags(df)
        df = self.add_rolling(df)
        df = self.add_interactions(df)
        df = self.add_squared(df)
        return df
