import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from data import get_features

SUPPORTED_STOCKS = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "JPM"]
MODEL_DIR = Path(__file__).parent / "models"
MODEL_TTL = timedelta(hours=1)

MODELS = {
    "rf": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "gb": GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42),
}
MODEL_NAMES = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
}

# Sentiment weight per horizon: news impact fades over time
_SENTIMENT_CACHE = Path(__file__).parent / "sentiment_cache.json"
_SENTIMENT_ALPHA = {1: 0.020, 3: 0.015, 7: 0.010}


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def _get_cached_sentiment(ticker: str) -> float | None:
    if not _SENTIMENT_CACHE.exists():
        return None
    try:
        return json.loads(_SENTIMENT_CACHE.read_text()).get(ticker, {}).get("avg_sentiment")
    except Exception:
        return None


def _apply_sentiment(
    raw_price: float, current_price: float, ticker: str, horizon: int
) -> tuple[float, float | None]:
    sentiment = _get_cached_sentiment(ticker)
    if sentiment is None:
        return raw_price, None
    alpha = _SENTIMENT_ALPHA.get(horizon, 0.015)
    return raw_price + sentiment * alpha * current_price, round(sentiment, 3)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _model_path(ticker: str, horizon: int, model_key: str) -> Path:
    return MODEL_DIR / f"{ticker}_{horizon}d_{model_key}.pkl"


def _meta_path(ticker: str, horizon: int, model_key: str) -> Path:
    return MODEL_DIR / f"{ticker}_{horizon}d_{model_key}_meta.json"


def _is_fresh(ticker: str, horizon: int, model_key: str) -> bool:
    mp = _model_path(ticker, horizon, model_key)
    metap = _meta_path(ticker, horizon, model_key)
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

def _train(ticker: str, horizon: int, model_key: str) -> tuple[dict, object, pd.DataFrame]:
    """Train on 2 years of data, persist to models/, return (meta, model, feat)."""
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODELS)}")

    MODEL_DIR.mkdir(exist_ok=True)
    feat = get_features(ticker, horizon=horizon)

    if len(feat) < 80:
        raise ValueError(f"Insufficient historical data for {ticker}")

    # Predict future *return* (pct change) rather than absolute price.
    # This normalises the target across price levels and makes directional
    # accuracy a direct outcome of the sign of the prediction.
    target = (feat["close"].shift(-horizon) / feat["close"] - 1).dropna()
    feat_aligned = feat.loc[target.index]

    X = feat_aligned.values
    y = target.values

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Clone a fresh instance so repeated calls don't accumulate state
    import sklearn.base
    model = sklearn.base.clone(MODELS[model_key])
    n = len(X_train)
    sample_weights = np.linspace(1.0, 2.0, n)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    preds_test = model.predict(X_test)
    dir_pred = (preds_test > 0).astype(int)
    dir_true = (y_test > 0).astype(int)
    confidence = float((dir_pred == dir_true).mean())

    joblib.dump(model, _model_path(ticker, horizon, model_key))

    meta = {
        "ticker": ticker,
        "horizon": horizon,
        "model_key": model_key,
        "trained_at": datetime.now().isoformat(),
        "confidence": round(confidence, 4),
        "train_samples": split,
        "feature_count": X.shape[1],
    }
    _meta_path(ticker, horizon, model_key).write_text(json.dumps(meta, indent=2))

    return meta, model, feat


def train(ticker: str, horizon: int, model_key: str = "rf") -> dict:
    """Public entry point for pre-training. Saves to models/ and returns meta."""
    meta, _, _ = _train(ticker, horizon, model_key)
    return meta


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(ticker: str, horizon: int = 1, model_key: str = "rf") -> dict:
    """
    Return a price prediction for `ticker` `horizon` trading days ahead.

    Loads the saved model when it is less than MODEL_TTL (1 hour) old.
    Otherwise retrains from scratch, saves, then predicts.
    """
    if model_key not in MODELS:
        model_key = "rf"

    retrained = False
    if _is_fresh(ticker, horizon, model_key):
        model = joblib.load(_model_path(ticker, horizon, model_key))
        meta = json.loads(_meta_path(ticker, horizon, model_key).read_text())
        feat = get_features(ticker, period="3y", horizon=horizon)
    else:
        meta, model, feat = _train(ticker, horizon, model_key)
        retrained = True

    current_price = float(feat["close"].iloc[-1])
    raw_return = float(model.predict(feat.values[-1].reshape(1, -1))[0])
    raw_price = current_price * (1 + raw_return)
    target_price, sentiment_score = _apply_sentiment(raw_price, current_price, ticker, horizon)
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
        "model_key": model_key,
        "model": MODEL_NAMES[model_key],
        "current_price": round(current_price, 2),
        "target_price": round(target_price, 2),
        "raw_rf_price": round(raw_price, 2),
        "sentiment_score": sentiment_score,
        "sentiment_adjusted": sentiment_score is not None,
        "trend": trend,
        "confidence": round(meta["confidence"], 2),
        "predictions": predictions,
        "trained_at": meta["trained_at"],
        "retrained": retrained,
    }
