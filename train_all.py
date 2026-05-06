"""
Pre-train models for all supported stocks, horizons, and algorithms.

Run once before launching the app:

    python train_all.py

Models are saved to models/<TICKER>_<N>d_<key>.pkl and expire after 1 hour,
after which the app retrains automatically on the next prediction request.
"""
from model import MODELS, SUPPORTED_STOCKS, train

HORIZONS = [1, 3, 7]

if __name__ == "__main__":
    model_keys = list(MODELS.keys())
    total = len(SUPPORTED_STOCKS) * len(HORIZONS) * len(model_keys)
    done = 0

    for ticker in SUPPORTED_STOCKS:
        for horizon in HORIZONS:
            for model_key in model_keys:
                done += 1
                print(f"[{done}/{total}]  {ticker}  {horizon}D  {model_key.upper()} ... ", end="", flush=True)
                try:
                    meta = train(ticker, horizon, model_key)
                    print(
                        f"OK  "
                        f"confidence={meta['confidence']:.1%}  "
                        f"samples={meta['train_samples']}"
                    )
                except Exception as e:
                    print(f"FAILED — {e}")

    print(f"\nDone. Models saved to models/")
