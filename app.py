from flask import Flask, jsonify, render_template, request

import yfinance as yf
from data import get_stock_data
from model import predict as ml_predict, train as ml_train
from news import fetch_news, get_remaining

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/data")
def data():
    ticker = request.args.get("stock", "AAPL").upper()
    try:
        return jsonify(get_stock_data(ticker))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict")
def predict():
    ticker = request.args.get("stock", "AAPL").upper()
    days = request.args.get("days", 3, type=int)
    model_key = request.args.get("model", "rf").lower()
    if days not in (1, 3, 7):
        days = 3
    if model_key not in ("rf", "gb"):
        model_key = "rf"
    try:
        return jsonify(ml_predict(ticker, days, model_key))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/news")
def news():
    ticker = request.args.get("stock", "AAPL").upper()
    try:
        return jsonify(fetch_news(ticker))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/econ")
def econ():
    try:
        vix_df = yf.download("^VIX", period="2d", auto_adjust=True, progress=False)
        if hasattr(vix_df.columns, "levels"):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix = round(float(vix_df["Close"].iloc[-1]), 2) if not vix_df.empty else None
    except Exception:
        vix = None
    return jsonify({
        "fed_rate": 4.33,   # last known value — live data requires FRED API key
        "cpi": 2.4,          # last known value — live data requires FRED API key
        "vix": vix,
    })


@app.route("/retrain")
def retrain():
    ticker = request.args.get("stock", "AAPL").upper()
    try:
        results = {}
        for model_key in ("rf", "gb"):
            for horizon in [1, 3, 7]:
                meta = ml_train(ticker, horizon, model_key)
                results[f"{model_key}_{horizon}d"] = {
                    "confidence": meta["confidence"],
                    "trained_at": meta["trained_at"],
                }
        return jsonify({"ticker": ticker, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/news/remaining")
def news_remaining():
    ticker = request.args.get("stock", "AAPL").upper()
    return jsonify({"remaining": get_remaining(ticker)})


if __name__ == "__main__":
    app.run(debug=True)
