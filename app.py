# Stock Trend Predictor — Flask Application
# AI Fundamentals Project, Warsaw University of Technology

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import Flask, jsonify, render_template, request
import yfinance as yf

from data import get_stock_data
from model import predict as run_prediction, train as train_model, get_evaluation
from news import fetch_news
from social import fetch_social
from sentiment import finbert_available

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
    algo = request.args.get("model", "rf").lower()
    if days not in (1, 3, 7):
        days = 3
    if algo not in ("rf", "gb"):
        algo = "rf"
    try:
        return jsonify(run_prediction(ticker, days, algo))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/news")
def news():
    ticker = request.args.get("stock", "AAPL").upper()
    try:
        return jsonify(fetch_news(ticker))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/social")
def social():
    ticker = request.args.get("stock", "AAPL").upper()
    try:
        return jsonify(fetch_social(ticker))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/econ")
def econ():
    # VIX — real, no key needed
    try:
        vix_df = yf.download("^VIX", period="2d", auto_adjust=True, progress=False)
        if hasattr(vix_df.columns, "levels"):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix = round(float(vix_df["Close"].iloc[-1]), 2) if not vix_df.empty else None
    except Exception:
        vix = None

    # Fed rate + CPI — real if FRED key set, hardcoded otherwise
    fed_rate, cpi, fred_live = 4.33, 2.4, False
    fred_key = os.getenv("FRED_API_KEY", "")
    if fred_key:
        try:
            from fredapi import Fred
            fred = Fred(api_key=fred_key)
            fed_rate = round(float(fred.get_series("FEDFUNDS").iloc[-1]), 2)
            cpi_series = fred.get_series("CPIAUCSL")
            cpi = round(float(cpi_series.pct_change(12, fill_method=None).dropna().iloc[-1] * 100), 1)
            fred_live = True
        except Exception as e:
            print(f"[FRED] {e}, using fallback values")

    return jsonify({"fed_rate": fed_rate, "cpi": cpi, "vix": vix, "fred_live": fred_live})


@app.route("/retrain")
def retrain():
    ticker = request.args.get("stock", "AAPL").upper()
    try:
        results = {}
        for algo in ("rf", "gb"):
            for horizon in (1, 3, 7):
                meta = train_model(ticker, horizon, algo)
                results[f"{algo}_{horizon}d"] = {"accuracy": meta["accuracy"], "trained_at": meta["trained_at"]}
        return jsonify({"ticker": ticker, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/evaluate")
def evaluate():
    ticker = request.args.get("stock", "AAPL").upper()
    algo = request.args.get("model", "rf").lower()
    horizon = request.args.get("days", 3, type=int)
    try:
        return jsonify(get_evaluation(ticker, horizon, algo))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
