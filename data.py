import yfinance as yf
import pandas as pd
import numpy as np


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()

    # RSI 14 (Wilder smoothing)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # MACD (12/26/9)
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    return df


def _safe(val):
    return float(val) if not pd.isna(val) else None


def get_stock_data(ticker: str = "AAPL") -> list[dict]:
    data = yf.download(ticker, period="60d", auto_adjust=True, progress=False)

    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    data = _add_indicators(data)

    result = []
    for index, row in data.iterrows():
        result.append({
            "date": index.strftime("%b %d"),
            "close": float(row["Close"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "volume": int(row["Volume"]),
            "sma20": _safe(row["SMA20"]),
            "ema12": _safe(row["EMA12"]),
            "rsi14": _safe(row["RSI14"]),
            "macd": _safe(row["MACD"]),
        })

    return result


def get_features(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Build feature DataFrame for ML model training."""
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    df = _add_indicators(df)

    feat = pd.DataFrame(index=df.index)
    feat["close"] = df["Close"]
    feat["volume"] = df["Volume"]
    feat["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
    feat["oc_spread"] = (df["Close"] - df["Open"]) / df["Open"]

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        feat[f"close_lag{lag}"] = df["Close"].shift(lag)

    # Returns
    feat["ret_1d"] = df["Close"].pct_change(1)
    feat["ret_5d"] = df["Close"].pct_change(5)
    feat["ret_20d"] = df["Close"].pct_change(20)

    # Technical indicators
    feat["sma20"] = df["SMA20"]
    feat["ema12"] = df["EMA12"]
    feat["sma_ratio"] = df["Close"] / df["SMA20"]
    feat["rsi14"] = df["RSI14"]
    feat["macd"] = df["MACD"]
    feat["macd_signal"] = df["MACD_Signal"]
    feat["macd_hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Band position (0 = lower band, 1 = upper band)
    bb_std = df["Close"].rolling(20).std()
    bb_upper = df["SMA20"] + 2 * bb_std
    bb_lower = df["SMA20"] - 2 * bb_std
    band_width = (bb_upper - bb_lower).replace(0, np.nan)
    feat["bb_position"] = (df["Close"] - bb_lower) / band_width

    # Volume ratio vs 20-day average
    feat["vol_sma20"] = df["Volume"].rolling(20).mean()
    feat["vol_ratio"] = df["Volume"] / feat["vol_sma20"]

    # Day of week (0 = Monday)
    feat["day_of_week"] = df.index.dayofweek

    return feat.dropna()
