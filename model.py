# ML prediction: classifier for direction + regressor for price target
# Models cached to disk, auto-expire after 1 hour

import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn.base
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data import get_features

MODEL_DIR = Path(__file__).parent / "models"
MODEL_TTL = timedelta(hours=1)

CLASSIFIERS = {
    "rf": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=8,
        max_features="sqrt", random_state=42, n_jobs=-1,
    ),
    "gb": GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05,
        subsample=0.85, min_samples_leaf=10, random_state=42,
    ),
}

REGRESSORS = {
    "rf": RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    ),
    "gb": GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.8, min_samples_leaf=10, random_state=42,
    ),
}

ALGO_NAMES = {"rf": "Random Forest", "gb": "Gradient Boosting"}
SENTIMENT_WEIGHT = {1: 0.020, 3: 0.015, 7: 0.010}
SENTIMENT_FILE = Path(__file__).parent / "sentiment_cache.json"


# -- Sentiment cache (written by news.py / social.py, read by predict) --

def read_sentiment(ticker):
    if not SENTIMENT_FILE.exists():
        return None
    try:
        return json.loads(SENTIMENT_FILE.read_text()).get(ticker, {}).get("avg_sentiment")
    except Exception:
        return None


def save_sentiment(ticker, score):
    cache = {}
    if SENTIMENT_FILE.exists():
        try:
            cache = json.loads(SENTIMENT_FILE.read_text())
        except Exception:
            pass
    cache[ticker] = {"avg_sentiment": score, "updated": datetime.now().isoformat()}
    SENTIMENT_FILE.write_text(json.dumps(cache, indent=2))


# -- File paths --

def _path(ticker, horizon, algo, suffix):
    return MODEL_DIR / f"{ticker}_{horizon}d_{algo}_{suffix}"


def _is_fresh(ticker, horizon, algo):
    meta = _path(ticker, horizon, algo, "meta.json")
    if not meta.exists():
        return False
    try:
        trained = datetime.fromisoformat(json.loads(meta.read_text())["trained_at"])
        return datetime.now() - trained < MODEL_TTL
    except Exception:
        return False


# -- Training --

def _train_model(ticker, horizon, algo):
    if algo not in CLASSIFIERS:
        raise ValueError(f"Unknown algorithm '{algo}'")

    MODEL_DIR.mkdir(exist_ok=True)
    features = get_features(ticker, period="3y", horizon=horizon)

    if len(features) < 100:
        raise ValueError(f"Not enough data for {ticker}")

    # Target: future return over the horizon
    future_return = (features["close"].shift(-horizon) / features["close"] - 1).dropna()
    direction = (future_return > 0).astype(int)

    X = features.loc[future_return.index].values
    y_cls = direction.values
    y_reg = future_return.values

    # 80/20 time-based split (no look-ahead bias)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_cls_train, y_cls_test = y_cls[:split], y_cls[split:]
    y_reg_train, y_reg_test = y_reg[:split], y_reg[split:]

    # Recent data weighted higher (markets change over time)
    weights = np.linspace(1.0, 2.0, len(X_train))

    # Train classifier (direction) and regressor (magnitude)
    clf = sklearn.base.clone(CLASSIFIERS[algo])
    clf.fit(X_train, y_cls_train, sample_weight=weights)

    reg = sklearn.base.clone(REGRESSORS[algo])
    reg.fit(X_train, y_reg_train, sample_weight=weights)

    # Evaluate
    cls_preds = clf.predict(X_test)
    reg_preds = reg.predict(X_test)
    accuracy = float((cls_preds == y_cls_test).mean())

    tp = int(((cls_preds == 1) & (y_cls_test == 1)).sum())
    fp = int(((cls_preds == 1) & (y_cls_test == 0)).sum())
    fn = int(((cls_preds == 0) & (y_cls_test == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    names = list(features.loc[future_return.index].columns)
    importances = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])[:10]

    # Save model bundle
    joblib.dump({"clf": clf, "reg": reg}, _path(ticker, horizon, algo, "model.pkl"))

    meta = {
        "ticker": ticker, "horizon": horizon, "algo": algo,
        "trained_at": datetime.now().isoformat(),
        "accuracy": round(accuracy, 4),
        "mae": round(float(mean_absolute_error(y_reg_test, reg_preds)), 6),
        "rmse": round(float(np.sqrt(mean_squared_error(y_reg_test, reg_preds))), 6),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "train_size": split, "test_size": len(X_test), "features": X.shape[1],
        "top_features": [{"name": n, "weight": round(float(w), 4)} for n, w in importances],
    }
    _path(ticker, horizon, algo, "meta.json").write_text(json.dumps(meta, indent=2))
    return meta, {"clf": clf, "reg": reg}, features


def train(ticker, horizon, algo="rf"):
    meta, _, _ = _train_model(ticker, horizon, algo)
    return meta


# -- Prediction --

def predict(ticker, horizon=1, algo="rf"):
    if algo not in CLASSIFIERS:
        algo = "rf"

    retrained = False
    if _is_fresh(ticker, horizon, algo):
        bundle = joblib.load(_path(ticker, horizon, algo, "model.pkl"))
        meta = json.loads(_path(ticker, horizon, algo, "meta.json").read_text())
        features = get_features(ticker, period="3y", horizon=horizon)
    else:
        meta, bundle, features = _train_model(ticker, horizon, algo)
        retrained = True

    clf, reg = bundle["clf"], bundle["reg"]
    current_price = float(features["close"].iloc[-1])
    latest = features.values[-1:].reshape(1, -1)

    # Direction from classifier, magnitude from regressor
    direction = int(clf.predict(latest)[0])
    predicted_return = float(reg.predict(latest)[0])

    # Align regressor with classifier direction
    if direction == 0 and predicted_return > 0:
        predicted_return = -abs(predicted_return)
    elif direction == 1 and predicted_return < 0:
        predicted_return = abs(predicted_return)

    raw_price = current_price * (1 + predicted_return)

    # Sentiment adjustment
    sentiment = read_sentiment(ticker)
    if sentiment is not None:
        alpha = SENTIMENT_WEIGHT.get(horizon, 0.015)
        target_price = raw_price + sentiment * alpha * current_price
    else:
        target_price = raw_price

    # Day-by-day prediction path
    last_date = features.index[-1]
    pred_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=horizon)
    daily = []
    for i, d in enumerate(pred_dates, 1):
        price = current_price + (i / horizon) * (target_price - current_price)
        daily.append({"date": d.strftime("%b %d"), "price": round(price, 2)})

    return {
        "ticker": ticker, "horizon": horizon, "algo": algo,
        "model": ALGO_NAMES[algo],
        "current_price": round(current_price, 2),
        "target_price": round(target_price, 2),
        "raw_price": round(raw_price, 2),
        "sentiment_score": round(sentiment, 3) if sentiment is not None else None,
        "trend": "up" if direction == 1 else "down",
        "accuracy": round(meta["accuracy"], 2),
        "mae": meta.get("mae"), "rmse": meta.get("rmse"),
        "precision": meta.get("precision"), "recall": meta.get("recall"),
        "predictions": daily,
        "trained_at": meta["trained_at"], "retrained": retrained,
    }


def get_evaluation(ticker, horizon=3, algo="rf"):
    path = _path(ticker, horizon, algo, "meta.json")
    if not path.exists():
        meta, _, _ = _train_model(ticker, horizon, algo)
    else:
        meta = json.loads(path.read_text())
    return meta
