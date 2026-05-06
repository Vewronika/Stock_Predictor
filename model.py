import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data import get_features

SUPPORTED_STOCKS = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "JPM"]
MODEL_DIR = Path(__file__).parent / "models"
MODEL_TTL = timedelta(hours=1)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _model_path(ticker: str, horizon: int) -> Path:
    return MODEL_DIR / f"{ticker}_{horizon}d.pkl"


def _meta_path(ticker: str, horizon: int) -> Path:
    return MODEL_DIR / f"{ticker}_{horizon}d_meta.json"


def _is_fresh(ticker: str, horizon: int) -> bool:
    """True if a saved model exists and was trained less than MODEL_TTL ago."""
    mp, metap = _model_path(ticker, horizon), _meta_path(ticker, horizon)
    if not mp.exists() or not metap.exists():
        return False
    try:
        trained_at = datetime.fromisoformat(
            json.loads(metap.read_text())["trained_at"]
        )
        return datetime.now() - trained_at < MODEL_TTL
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train(
    ticker: str, horizon: int
) -> tuple[dict, RandomForestRegressor, pd.DataFrame]:
    """Train on 2 years of data, persist to models/, return (meta, model, feat)."""
    MODEL_DIR.mkdir(exist_ok=True)
    feat = get_features(ticker)

    if len(feat) < 80:
        raise ValueError(f"Insufficient historical data for {ticker}")

    target = feat["close"].shift(-horizon).dropna()
    feat_aligned = feat.loc[target.index]

    X = feat_aligned.values
    y = target.values
    current_prices_all = feat_aligned["close"].values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    current_prices_test = current_prices_all[split:]

    model = RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    preds_test = model.predict(X_test)
    dir_pred = (preds_test > current_prices_test).astype(int)
    dir_true = (y_test > current_prices_test).astype(int)
    confidence = float((dir_pred == dir_true).mean())

    joblib.dump(model, _model_path(ticker, horizon))

    meta = {
        "ticker": ticker,
        "horizon": horizon,
        "trained_at": datetime.now().isoformat(),
        "confidence": round(confidence, 4),
        "train_samples": split,
        "feature_count": X.shape[1],
    }
    _meta_path(ticker, horizon).write_text(json.dumps(meta, indent=2))

    return meta, model, feat


def train(ticker: str, horizon: int) -> dict:
    """Public entry point for pre-training. Saves to models/ and returns meta."""
    meta, _, _ = _train(ticker, horizon)
    return meta


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(ticker: str, horizon: int = 1) -> dict:
    """
    Return a price prediction for `ticker` `horizon` trading days ahead.

    Loads the saved model when it is less than MODEL_TTL (1 hour) old.
    Otherwise retrains from scratch, saves, then predicts.
    """
    retrained = False
    if _is_fresh(ticker, horizon):
        model = joblib.load(_model_path(ticker, horizon))
        meta = json.loads(_meta_path(ticker, horizon).read_text())
        feat = get_features(ticker, period="60d")
    else:
        meta, model, feat = _train(ticker, horizon)
        retrained = True

    current_price = float(feat["close"].iloc[-1])
    target_price = float(model.predict(feat.values[-1].reshape(1, -1))[0])
    trend = "up" if target_price > current_price else "down"

    last_date = feat.index[-1]
    pred_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)
    predictions = []
    for i, d in enumerate(pred_dates, 1):
        frac = i / horizon
        price = current_price + frac * (target_price - current_price)
        predictions.append({"date": d.strftime("%b %d"), "price": round(price, 2)})

    return {
        "ticker": ticker,
        "horizon": horizon,
        "current_price": round(current_price, 2),
        "target_price": round(target_price, 2),
        "trend": trend,
        "confidence": round(meta["confidence"], 2),
        "predictions": predictions,
        "model": "Random Forest",
        "trained_at": meta["trained_at"],
        "retrained": retrained,
    }
