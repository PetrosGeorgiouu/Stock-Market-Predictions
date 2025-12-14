# data_prep.py

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple

FEATURE_COLS = [
    "log_return_1d",
    "pct_return_1d",
    "pct_return_5d",
    "pct_return_10d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "close_over_ma5",
    "close_over_ma10",
    "close_over_ma20",
    "volume_zscore_20",
    "volume_change",
    # "rsi_14",  
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_width",
    "lag_ret_1",
    "lag_ret_2",
    "lag_vol_1",
]

def download_price_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,  
    )

    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker} between {start} and {end}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found. Columns: {list(df.columns)}")
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols].dropna()
    return df

def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    
    # Stationary, relative technical features to the price DataFrame
    df = df.copy()
    price = df["Close"]
    volume = df["Volume"]
    # ===== Returns (stationary) =====
    df["log_return_1d"] = np.log(price).diff()
    df["pct_return_1d"] = price.pct_change()

    df["pct_return_5d"] = price.pct_change(5)
    df["pct_return_10d"] = price.pct_change(10)
    # ===== Volatility of returns =====
    df["vol_5d"] = df["pct_return_1d"].rolling(5).std()
    df["vol_10d"] = df["pct_return_1d"].rolling(10).std()
    df["vol_20d"] = df["pct_return_1d"].rolling(20).std()
    # ===== Moving averages (for relative trend only) =====
    ma_5 = price.rolling(5).mean()
    ma_10 = price.rolling(10).mean()
    ma_20 = price.rolling(20).mean()
    df["close_over_ma5"] = price / ma_5 - 1.0
    df["close_over_ma10"] = price / ma_10 - 1.0
    df["close_over_ma20"] = price / ma_20 - 1.0
    # ===== Volume (normalized) =====
    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std()
    df["volume_zscore_20"] = (volume - vol_mean_20) / (vol_std_20 + 1e-9)
    df["volume_change"] = volume.pct_change()
    # ===== RSI (14) =====
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    # ===== MACD (12, 26, 9) =====
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    # ===== Bollinger Band width (relative volatility) =====
    bb_mid = price.rolling(20).mean()
    bb_std = price.rolling(20).std()
    df["bb_width"] = (4 * bb_std) / bb_mid   # normalized width
    # ===== Lagged features =====
    df["lag_ret_1"] = df["pct_return_1d"].shift(1)
    df["lag_ret_2"] = df["pct_return_1d"].shift(2)
    df["lag_vol_1"] = df["volume_change"].shift(1)
    return df

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    price_col = "Close"
    next_close = df[price_col].shift(-1)
    df["target"] = (next_close > df[price_col]).astype(int)
    # Drop last row (no label for last day)
    df = df.iloc[:-1]
    return df


def build_feature_matrix_and_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df_model = df.dropna(subset=FEATURE_COLS + ["target"]).copy()
    X = df_model[FEATURE_COLS].values
    y = df_model["target"].values.astype(int)
    return X, y

def split_train_test_by_date(
    df: pd.DataFrame,
    test_size_years: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    last_date = df.index.max()
    cutoff_year = last_date.year - test_size_years + 1
    cutoff_date = pd.Timestamp(year=cutoff_year, month=1, day=1)
    df_train = df[df.index < cutoff_date]
    df_test = df[df.index >= cutoff_date]
    return df_train, df_test

def create_dataset(
    ticker: str,
    start: str,
    end: str,
    test_size_years: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    df = download_price_data(ticker, start, end)
    df = add_technical_features(df)
    df = add_labels(df)
    df_train, df_test = split_train_test_by_date(df, test_size_years=test_size_years)
    X_train, y_train = build_feature_matrix_and_labels(df_train)
    X_test, y_test = build_feature_matrix_and_labels(df_test)
    return X_train, X_test, y_train, y_test

def add_regression_labels(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    price_col = "Close"

    next_close = df[price_col].shift(-1)
    df["target_reg"] = (next_close - df[price_col]) / df[price_col]

    df = df.iloc[:-1]
    return df


def build_feature_matrix_and_regression_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df_model = df.dropna(subset=FEATURE_COLS + ["target_reg"]).copy()
    X = df_model[FEATURE_COLS].values
    y = df_model["target_reg"].values.astype(float)
    return X, y

def create_regression_dataset(
    ticker: str,
    start: str,
    end: str,
    test_size_years: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = download_price_data(ticker, start, end)
    df = add_technical_features(df)
    df = add_regression_labels(df)

    df_train, df_test = split_train_test_by_date(df, test_size_years=test_size_years)

    X_train, y_train = build_feature_matrix_and_regression_labels(df_train)
    X_test, y_test = build_feature_matrix_and_regression_labels(df_test)

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Simple sanity check when running this file directly
    X_train, X_test, y_train, y_test = create_dataset(
        ticker="AAPL",
        start="2015-01-01",
        end="2024-12-31",
        test_size_years=2,
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
