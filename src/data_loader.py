import pandas as pd

def load_raw(cfg):
    df = pd.read_csv(cfg["dataset"]["path"])
    df[cfg["dataset"]["datetime_col"]] = pd.to_datetime(df[cfg["dataset"]["datetime_col"]])
    df = df.set_index(cfg["dataset"]["datetime_col"])
    df = df.asfreq(cfg["dataset"]["freq"])
    return df
