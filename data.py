import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def generate_sine_dataset(n_series=1000, seq_len=50, noise=0.1):
    X = []
    y = []
    for _ in range(n_series):
        freq = random.uniform(0.05, 0.5)
        phase = random.uniform(0, 2 * math.pi)
        amp = random.uniform(0.5, 2.0)
        t = np.arange(seq_len + 1)
        series = amp * np.sin(2 * math.pi * freq * t + phase)
        series += np.random.normal(0, noise, size=series.shape)
        X.append(series[:-1].astype(np.float32).reshape(-1, 1))
        y.append(series[-1].astype(np.float32))

    X = np.stack(X)  # (n_series, seq_len, 1)
    y = np.stack(y)
    return X, y


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_sequences_from_dataframe(df, feature_cols, target_col, seq_len=50, dropna=True):
    """Prepare sliding-window sequences from a pandas DataFrame.

    Args:
        df: pandas.DataFrame indexed by time or ordered rows.
        feature_cols: list of column names to use as inputs.
        target_col: name of the column to predict (scalar per time step).
        seq_len: length of input sequences.
        dropna: whether to drop rows with NaNs.

    Returns:
        X: numpy array shape (n_samples, seq_len, n_features)
        y: numpy array shape (n_samples,)
    """
    import pandas as pd

    if dropna:
        df = df.dropna()

    values = df[feature_cols + [target_col]].values.astype('float32')
    n_rows, n_cols = values.shape
    n_features = len(feature_cols)

    X = []
    y = []
    for i in range(n_rows - seq_len):
        window = values[i:i + seq_len]
        X.append(window[:, :n_features])
        y.append(values[i + seq_len, -1])

    X = np.stack(X) if len(X) > 0 else np.zeros((0, seq_len, n_features), dtype=np.float32)
    y = np.stack(y) if len(y) > 0 else np.zeros((0,), dtype=np.float32)
    return X, y


def load_csv_time_series(path_or_url, feature_cols, target_col, seq_len=50, parse_dates=None, **pd_read_csv_kwargs):
    """Load a CSV time-series from a path or URL and convert to sliding-window sequences.

    This is a lightweight helper intended to work with research CSVs (e.g., station timeseries).
    For large gridded datasets (NetCDF/ERA5), prefer using xarray or the provider's API.

    Args:
        path_or_url: local path or HTTP(S) URL to CSV file.
        feature_cols: list of input columns.
        target_col: output column name.
        seq_len: sequence length for model inputs.
        parse_dates: column name(s) to parse as dates (optional).

    Returns:
        X, y as numpy arrays suitable for `TimeSeriesDataset`.
    """
    import pandas as pd

    df = pd.read_csv(path_or_url, parse_dates=parse_dates, **pd_read_csv_kwargs)
    X, y = prepare_sequences_from_dataframe(df, feature_cols, target_col, seq_len=seq_len)
    return X, y
