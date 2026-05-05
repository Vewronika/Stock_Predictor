from flask import Flask, render_template, jsonify, request
import random
from data import get_stock_data


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict")
def predict():
    # TEMP: fake prediction (replace later)
    return jsonify({
        "trend": "bullish",
        "price": round(200 + random.random()*10, 2),
        "confidence": round(70 + random.random()*20, 2)
    })


@app.route("/data")
def data():
    ticker = request.args.get("stock", "AAPL")
    stock_data = get_stock_data(ticker)
    return jsonify(stock_data)


if __name__ == "__main__":
    app.run(debug=True)