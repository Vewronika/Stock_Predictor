import yfinance as yf
import pandas as pd

def get_stock_data(ticker="AAPL"):
    data = yf.download(ticker, period="60d")

    # flatten columns if needed
    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    # 👉 ADD THIS
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["EMA12"] = data["Close"].ewm(span=12, adjust=False).mean()

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
            "ema12": float(row["EMA12"]) if not pd.isna(row["EMA12"]) else None
        })

    return result