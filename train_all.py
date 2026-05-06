"""
Pre-train Random Forest models for all supported stocks and horizons.

Run once before launching the app:

    python train_all.py

Models are saved to models/<TICKER>_<N>d.pkl and expire after 1 hour,
after which the app retrains automatically on the next prediction request.
"""
from model import SUPPORTED_STOCKS, train

HORIZONS = [1, 3, 7]

if __name__ == "__main__":
    total = len(SUPPORTED_STOCKS) * len(HORIZONS)
    done = 0

    for ticker in SUPPORTED_STOCKS:
        for horizon in HORIZONS:
            done += 1
            print(f"[{done}/{total}]  {ticker}  {horizon}D ... ", end="", flush=True)
            try:
                meta = train(ticker, horizon)
                print(
                    f"OK  "
                    f"confidence={meta['confidence']:.1%}  "
                    f"samples={meta['train_samples']}  "
                    f"features={meta['feature_count']}"
                )
            except Exception as e:
                print(f"FAILED — {e}")

    print(f"\nDone. Models saved to models/")
