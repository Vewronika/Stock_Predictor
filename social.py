# Social media sentiment via Bluesky (free public API, no key needed)

import re
import requests
import yfinance as yf
from datetime import datetime, timedelta

from sentiment import score_both, label as sent_label, pick_best

BLUESKY_URL = "https://api.bsky.app/xrpc/app.bsky.feed.searchPosts"
MAX_AGE_DAYS = 30

_name_cache = {}


def _company_name(ticker):
    if ticker in _name_cache:
        return _name_cache[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        for suffix in [", Inc.", " Inc.", " Corp.", " Corporation", " Co.", " Ltd."]:
            name = name.replace(suffix, "")
        _name_cache[ticker] = name.strip()
    except Exception:
        _name_cache[ticker] = ticker
    return _name_cache[ticker]


def _build_query(ticker, company):
    if len(ticker) <= 2:
        return f'"{company}" stock'
    return f'${ticker} OR "{company}" stock'


def _clean_text(text):
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'#\S+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _is_relevant(text, ticker, company):
    text_lower = text.lower()
    company_lower = company.lower()

    finance_words = [
        "stock", "share", "buy", "sell", "bull", "bear", "earnings",
        "price", "market", "trading", "invest", "profit", "revenue",
        "dividend", "rally", "dip", "calls", "puts", "options",
        "portfolio", "nasdaq", "nyse", "forecast", "target", "analyst",
    ]
    has_finance = any(w in text_lower for w in finance_words)

    # $TICKER is always relevant
    if f"${ticker.lower()}" in text_lower or f"${ticker}" in text:
        return True

    # For long tickers (3+ chars): company name alone is enough
    if len(ticker) >= 3 and company_lower in text_lower:
        return True

    # For short tickers: company name + financial context required
    if company_lower in text_lower and has_finance:
        return True

    # Ticker as standalone word + financial context
    if has_finance and f" {ticker.lower()} " in f" {text_lower} ":
        return True

    return False


def _is_spam(text):
    cleaned = _clean_text(text)
    if len(cleaned) < 20:
        return True
    alpha_ratio = sum(c.isalpha() for c in cleaned) / max(len(cleaned), 1)
    return alpha_ratio < 0.4


def _is_recent(created_at, max_days=MAX_AGE_DAYS):
    try:
        post_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        cutoff = datetime.now(post_date.tzinfo) - timedelta(days=max_days)
        return post_date >= cutoff
    except Exception:
        return True  # keep if we can't parse the date


def _post_url(post):
    handle = post.get("author", {}).get("handle", "")
    uri = post.get("uri", "")
    parts = uri.split("/")
    if len(parts) >= 5 and handle:
        return f"https://bsky.app/profile/{handle}/post/{parts[-1]}"
    return ""


def fetch_social(ticker):
    company = _company_name(ticker)
    query = _build_query(ticker, company)
    messages = []
    seen_authors = set()
    error = None

    try:
        resp = requests.get(BLUESKY_URL, params={"q": query, "limit": 25, "sort": "latest"}, timeout=10)
        resp.raise_for_status()

        for post in resp.json().get("posts", []):
            text = post.get("record", {}).get("text", "")
            created_at = post.get("record", {}).get("createdAt", "")

            if _is_spam(text):
                continue
            if not _is_relevant(_clean_text(text), ticker, company):
                continue
            if not _is_recent(created_at):
                continue

            author = post.get("author", {})
            handle = author.get("handle", "")
            if handle in seen_authors:
                continue
            seen_authors.add(handle)

            v, f = score_both(text)
            combined = pick_best(v, f)

            messages.append({
                "body": _clean_text(text)[:250],
                "username": author.get("displayName", handle),
                "handle": f"@{handle}",
                "url": _post_url(post),
                "created_at": created_at[:16].replace("T", " "),
                "sentiment_vader": v,
                "sentiment_finbert": f,
                "sentiment": combined,
                "sentiment_label": sent_label(combined),
                "likes": post.get("likeCount", 0),
                "reposts": post.get("repostCount", 0),
            })

        messages.sort(key=lambda m: m["likes"] + m["reposts"], reverse=True)
        messages = messages[:5]

    except requests.exceptions.HTTPError as e:
        error = f"Bluesky API error: {e.response.status_code}"
    except requests.exceptions.ConnectionError:
        error = "Could not connect to Bluesky."
    except Exception as e:
        error = str(e)

    if error:
        return {"ticker": ticker, "messages": [], "error": error, "source": "Bluesky"}

    if not messages:
        return {
            "ticker": ticker, "messages": [], "avg_sentiment": 0, "avg_vader": 0,
            "avg_finbert": None, "finbert_available": False, "mood": "Neutral",
            "message_count": 0, "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source": "Bluesky",
        }

    avg_vader = round(sum(m["sentiment_vader"] for m in messages) / len(messages), 3)
    fb_scores = [m["sentiment_finbert"] for m in messages if m["sentiment_finbert"] is not None]
    avg_finbert = round(sum(fb_scores) / len(fb_scores), 3) if fb_scores else None
    avg = avg_finbert if avg_finbert is not None else avg_vader

    try:
        from model import save_sentiment, read_sentiment
        news_sent = read_sentiment(ticker)
        blended = round((news_sent + avg) / 2, 3) if news_sent is not None else avg
        save_sentiment(ticker, blended)
    except Exception:
        pass

    return {
        "ticker": ticker, "messages": messages,
        "avg_sentiment": avg, "avg_vader": avg_vader, "avg_finbert": avg_finbert,
        "finbert_available": avg_finbert is not None,
        "mood": sent_label(avg), "message_count": len(messages),
        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": "Bluesky",
    }
