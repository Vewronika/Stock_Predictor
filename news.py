import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RATE_LIMIT = 3
_RATE_FILE = Path(__file__).parent / "news_rate_limit.json"
SENTIMENT_CACHE = Path(__file__).parent / "sentiment_cache.json"

COMPANY_NAMES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet Google",
    "JPM": "JPMorgan",
}

_vader = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Rate limiting helpers
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    if _RATE_FILE.exists():
        try:
            return json.loads(_RATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_state(state: dict) -> None:
    _RATE_FILE.write_text(json.dumps(state, indent=2))


def get_remaining(ticker: str) -> int:
    today = str(date.today())
    return RATE_LIMIT - _load_state().get(today, {}).get(ticker, 0)


def _increment(ticker: str) -> int:
    today = str(date.today())
    state = _load_state()
    state.setdefault(today, {})
    state[today][ticker] = state[today].get(ticker, 0) + 1
    _save_state(state)
    return RATE_LIMIT - state[today][ticker]


# ---------------------------------------------------------------------------
# Sentiment helpers
# ---------------------------------------------------------------------------

def _score(text: str) -> float:
    return round(_vader.polarity_scores(text)["compound"], 3)


def _label(score: float) -> str:
    if score >= 0.05:
        return "Bullish"
    if score <= -0.05:
        return "Bearish"
    return "Neutral"


# ---------------------------------------------------------------------------
# News sources
# ---------------------------------------------------------------------------

def _fetch_finnhub(ticker: str) -> list[dict]:
    token = os.getenv("FINNHUB_KEY", "")
    if not token:
        return []
    from_date = (date.today() - timedelta(days=7)).isoformat()
    url = (
        f"https://finnhub.io/api/v1/company-news"
        f"?symbol={ticker}&from={from_date}&to={date.today().isoformat()}&token={token}"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        articles = []
        for item in r.json()[:8]:
            if not item.get("headline"):
                continue
            text = item["headline"] + " " + item.get("summary", "")
            s = _score(text)
            articles.append({
                "title": item["headline"],
                "source": item.get("source", "FinnHub"),
                "url": item.get("url", ""),
                "published_at": datetime.fromtimestamp(item["datetime"]).strftime("%Y-%m-%d %H:%M"),
                "sentiment": s,
                "sentiment_label": _label(s),
            })
        return articles
    except Exception:
        return []


def _fetch_newsapi(ticker: str) -> list[dict]:
    token = os.getenv("NEWSAPI_KEY", "")
    if not token:
        return []
    company = COMPANY_NAMES.get(ticker, ticker)
    q = requests.utils.quote(f"{ticker} OR {company} stock")
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={q}&language=en&sortBy=publishedAt&pageSize=8&apiKey={token}"
    )
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        articles = []
        for item in r.json().get("articles", [])[:8]:
            title = item.get("title", "")
            if not title or title == "[Removed]":
                continue
            text = title + " " + (item.get("description") or "")
            s = _score(text)
            articles.append({
                "title": title,
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "url": item.get("url", ""),
                "published_at": item.get("publishedAt", "")[:16].replace("T", " "),
                "sentiment": s,
                "sentiment_label": _label(s),
            })
        return articles
    except Exception:
        return []


_MOCK = [
    ("Fed signals potential rate pause amid cooling inflation data", "Reuters", 0.32),
    ("Tech earnings beat expectations across the board in Q1 report", "Bloomberg", 0.78),
    ("Supply chain disruptions ease significantly in Q2 outlook", "CNBC", 0.41),
    ("New tariff concerns continue to weigh on global equity markets", "WSJ", -0.55),
    ("Company announces major $10B stock buyback program", "MarketWatch", 0.62),
]


def _mock_articles(ticker: str) -> list[dict]:
    company = COMPANY_NAMES.get(ticker, ticker)
    return [
        {
            "title": title.replace("Company", company),
            "source": src,
            "url": "",
            "published_at": str(date.today()),
            "sentiment": score,
            "sentiment_label": _label(score),
        }
        for title, src, score in _MOCK
    ]


# ---------------------------------------------------------------------------
# Sentiment cache (shared with model.py for sentiment-adjusted predictions)
# ---------------------------------------------------------------------------

def _save_sentiment(ticker: str, avg_sentiment: float) -> None:
    cache: dict = {}
    if SENTIMENT_CACHE.exists():
        try:
            cache = json.loads(SENTIMENT_CACHE.read_text())
        except Exception:
            pass
    cache[ticker] = {
        "avg_sentiment": avg_sentiment,
        "fetched_at": datetime.now().isoformat(),
    }
    SENTIMENT_CACHE.write_text(json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news(ticker: str) -> dict:
    remaining = get_remaining(ticker)

    if remaining <= 0:
        return {
            "error": "rate_limit",
            "message": f"Daily news limit reached for {ticker}. Try again tomorrow.",
            "remaining_fetches": 0,
        }

    articles = _fetch_finnhub(ticker)
    source = "FinnHub"

    if not articles:
        articles = _fetch_newsapi(ticker)
        source = "NewsAPI"

    demo_mode = False
    if not articles:
        articles = _mock_articles(ticker)
        source = "Demo"
        demo_mode = True

    new_remaining = _increment(ticker)
    scores = [a["sentiment"] for a in articles]
    avg = round(sum(scores) / len(scores), 3) if scores else 0.0

    # Persist sentiment so the prediction model can use it
    _save_sentiment(ticker, avg)

    return {
        "ticker": ticker,
        "articles": articles,
        "avg_sentiment": avg,
        "mood": _label(avg),
        "remaining_fetches": new_remaining,
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "demo_mode": demo_mode,
    }
