# Dual sentiment scoring: VADER (rule-based) + FinBERT (neural, financial text)

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
_finbert = None
_finbert_tried = False


def _load_finbert():
    global _finbert, _finbert_tried
    if _finbert_tried:
        return _finbert
    _finbert_tried = True
    try:
        from transformers import pipeline
        print("[FinBERT] Loading model...")
        _finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        print("[FinBERT] Ready.")
    except ImportError:
        print("[FinBERT] transformers/torch not installed, using VADER only.")
    except Exception as e:
        print(f"[FinBERT] Failed: {e}, using VADER only.")
    return _finbert


def score_vader(text):
    return round(_vader.polarity_scores(text)["compound"], 3)


def score_finbert(text):
    pipe = _load_finbert()
    if pipe is None:
        return None
    try:
        result = pipe(text[:512])[0]
        label = result["label"].lower()
        score = result["score"]
        if label == "positive":
            return round(score, 3)
        elif label == "negative":
            return round(-score, 3)
        return 0.0
    except Exception:
        return None


def score_both(text):
    return score_vader(text), score_finbert(text)


def label(score):
    if score >= 0.05:
        return "Bullish"
    elif score <= -0.05:
        return "Bearish"
    return "Neutral"


def pick_best(vader, finbert):
    return finbert if finbert is not None else vader


def finbert_available():
    return _load_finbert() is not None
