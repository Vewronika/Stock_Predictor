import yfinance as yf
import pandas as pd

def get_stock_data(ticker="AAPL"):
    data = yf.download(ticker, period="60d")

    # flatten columns if needed
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)


    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()
    # RSI (14)
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26

    # Bollinger Bands %B
    sma20 = data["Close"].rolling(window=20).mean()
    std20 = data["Close"].rolling(window=20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    data["BBP"] = (data["Close"] - lower) / (upper - lower)

    # Avg Volume (20)
    data["AVG_VOL"] = data["Volume"].rolling(window=20).mean()

    result = []
    for index, row in data.iterrows():
        result.append({
            "date": index.strftime("%b %d"),
            "close": float(row["Close"]),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "volume": int(row["Volume"]),
            "sma20": float(row["SMA20"]) if not pd.isna(row["SMA20"]) else None,
            "ema12": float(row["EMA12"]) if not pd.isna(row["EMA12"]) else None,
            "rsi14": float(row["RSI14"]) if not pd.isna(row["RSI14"]) else None,
            "macd": float(row["MACD"]) if not pd.isna(row["MACD"]) else None,
            "bbp": float(row["BBP"]) if not pd.isna(row["BBP"]) else None,
            "avg_vol": int(row["AVG_VOL"]) if not pd.isna(row["AVG_VOL"]) else None,
        })

    return result