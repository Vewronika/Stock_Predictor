# News fetching: FinnHub → NewsAPI → demo headlines (fallback chain)
# Each article scored with both VADER and FinBERT

import os
from datetime import date, datetime, timedelta

import requests
import yfinance as yf

from sentiment import score_both, label as sent_label, pick_best
from model import save_sentiment

_name_cache = {}


def _company_name(ticker):
    if ticker in _name_cache:
        return _name_cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        for suffix in [", Inc.", " Inc.", " Corp.", " Corporation", " Co.", " Ltd.", " plc", " Group"]:
            name = name.replace(suffix, "")
        _name_cache[ticker] = name.strip()
    except Exception:
        _name_cache[ticker] = ticker
    return _name_cache[ticker]


def _score_article(text):
    v, f = score_both(text)
    combined = pick_best(v, f)
    return {
        "sentiment_vader": v,
        "sentiment_finbert": f,
        "sentiment": combined,
        "sentiment_label": sent_label(combined),
    }


def _fetch_finnhub(ticker):
    token = os.getenv("FINNHUB_KEY", "")
    if not token:
        return []
    from_date = (date.today() - timedelta(days=7)).isoformat()
    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={from_date}&to={date.today().isoformat()}&token={token}"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        articles = []
        for item in resp.json()[:8]:
            headline = item.get("headline", "")
            if not headline:
                continue
            text = headline + " " + item.get("summary", "")
            articles.append({
                "title": headline,
                "source": item.get("source", "FinnHub"),
                "url": item.get("url", ""),
                "published_at": datetime.fromtimestamp(item["datetime"]).strftime("%Y-%m-%d %H:%M"),
                **_score_article(text),
            })
        return articles
    except Exception:
        return []


def _fetch_newsapi(ticker):
    token = os.getenv("NEWSAPI_KEY", "")
    if not token:
        return []
    company = _company_name(ticker)
    query = requests.utils.quote(f"{ticker} OR {company} stock")
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=8&apiKey={token}"
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        articles = []
        for item in resp.json().get("articles", [])[:8]:
            title = item.get("title", "")
            if not title or title == "[Removed]":
                continue
            text = title + " " + (item.get("description") or "")
            articles.append({
                "title": title,
                "source": item.get("source", {}).get("name", "NewsAPI"),
                "url": item.get("url", ""),
                "published_at": item.get("publishedAt", "")[:16].replace("T", " "),
                **_score_article(text),
            })
        return articles
    except Exception:
        return []


_DEMO = [
    ("Fed signals potential rate pause amid cooling inflation data", "Reuters"),
    ("Tech earnings beat expectations across the board in Q1 report", "Bloomberg"),
    ("Supply chain disruptions ease significantly in Q2 outlook", "CNBC"),
    ("New tariff concerns continue to weigh on global equity markets", "WSJ"),
    ("Company announces major $10B stock buyback program", "MarketWatch"),
]


def _demo_articles(ticker):
    company = _company_name(ticker)
    return [
        {"title": t.replace("Company", company), "source": s,
         "url": "", "published_at": str(date.today()), **_score_article(t)}
        for t, s in _DEMO
    ]


def fetch_news(ticker):
    articles = _fetch_finnhub(ticker)
    source = "FinnHub"

    if not articles:
        articles = _fetch_newsapi(ticker)
        source = "NewsAPI"

    demo = False
    if not articles:
        articles = _demo_articles(ticker)
        source = "Demo"
        demo = True

    vader_scores = [a["sentiment_vader"] for a in articles]
    avg_vader = round(sum(vader_scores) / len(vader_scores), 3) if vader_scores else 0.0

    fb_scores = [a["sentiment_finbert"] for a in articles if a["sentiment_finbert"] is not None]
    avg_finbert = round(sum(fb_scores) / len(fb_scores), 3) if fb_scores else None

    avg = avg_finbert if avg_finbert is not None else avg_vader
    save_sentiment(ticker, avg)

    return {
        "ticker": ticker, "articles": articles,
        "avg_sentiment": avg, "avg_vader": avg_vader, "avg_finbert": avg_finbert,
        "finbert_available": avg_finbert is not None,
        "mood": sent_label(avg),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": source, "demo_mode": demo,
    }
