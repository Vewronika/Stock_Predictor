# Pre-train models for common stocks so predictions are instant

from model import train

STOCKS = ["AAPL", "MSFT", "TSLA", "AMZN", "GOOGL", "JPM"]

total = len(STOCKS) * 3 * 2  # stocks × horizons × algorithms
i = 0

for ticker in STOCKS:
    for horizon in [1, 3, 7]:
        for algo in ["rf", "gb"]:
            i += 1
            print(f"[{i}/{total}]  {ticker}  {horizon}D  {algo.upper()} ... ", end="", flush=True)
            try:
                meta = train(ticker, horizon, algo)
                print(f"OK  accuracy={meta['accuracy']*100:.1f}%  samples={meta['train_size']}")
            except Exception as e:
                print(f"FAIL  {e}")

print(f"\nDone. Models saved to models/")
