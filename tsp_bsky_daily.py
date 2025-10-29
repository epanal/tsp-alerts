"""
TSP Alerts — Daily fund prices + % change posted to Bluesky.
Data source: Official TSP CSV (tsp.gov) over a rolling window (default 30 days).

Setup:
  pip install pandas python-dotenv requests atproto tweepy

Run:
  python tsp_bsky_daily.py   # DRY_RUN=true by default (set in .env)
"""

from datetime import date, timedelta
import io
import os
import time
from typing import Dict, List, Tuple

import pandas as pd
import re
import requests
from dotenv import load_dotenv
from atproto import Client, models
import tweepy

# -------------------- Weekend guard --------------------
# Skip Sat(5)/Sun(6). Keep this after datetime imports.
#if date.today().weekday() in (5, 6):
#    print("[info] Weekend: skipping post.")
#    raise SystemExit(0)

# -------------------- Config helpers --------------------

DEFAULT_FUNDS = ["G Fund", "F Fund", "C Fund", "S Fund", "I Fund"]

def get_funds_from_env() -> List[str]:
    raw = os.getenv("FUNDS", "").strip()
    if not raw:
        return DEFAULT_FUNDS
    return [x.strip() for x in raw.split(",") if x.strip()]


# -------------------- HTTP headers for tsp.gov --------------------

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,application/octet-stream,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.tsp.gov/share-price-history/",
    "Connection": "keep-alive",
}

def build_tsp_csv_url(window_days: int = 30, include_L: bool = True, include_inv: bool = True) -> str:
    end = date.today()
    start = end - timedelta(days=window_days)
    return (
        "https://www.tsp.gov/data/fund-price-history.csv"
        f"?startdate={start:%Y-%m-%d}"
        f"&enddate={end:%Y-%m-%d}"
        f"&Lfunds={'1' if include_L else '0'}"
        f"&InvFunds={'1' if include_inv else '0'}"
        "&download=1"  # attachment-type often behaves better
    )

# -------------------- Core logic --------------------

def http_get_with_retry(url, session, tries=3, backoff=2.0):
    last_err = None
    for i in range(tries):
        try:
            r = session.get(url, timeout=30, allow_redirects=True)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i+1))
    raise last_err

CORE = {"G Fund","F Fund","C Fund","S Fund","I Fund"}
def top_mover_label(changes: dict) -> str:
    pool = {k:v for k,v in changes.items() if k in CORE} or changes
    if not pool:
        return ""
    k = max(pool, key=lambda f: abs(pool[f]))
    v = pool[k]
    sign = "+" if v >= 0 else ""
    return f" — Top: {k.split()[0]} {sign}{v:.2f}%"

def fetch_prices_and_changes_from_csv_dynamic(
    funds: List[str],
    window_days: int = 30
) -> Tuple[Dict[str, float], Dict[str, float], date]:
    """
    Pull official TSP CSV for the last `window_days`, compute:
      - prices:  latest share price per fund
      - changes: % change vs previous trading day per fund
    Returns (prices_dict, changes_dict, as_of_date).
    """
    url = build_tsp_csv_url(window_days=window_days, include_L=True, include_inv=True)

    with requests.Session() as s:
        s.headers.update(BROWSER_HEADERS)
        r = http_get_with_retry(url, s)

        # Tolerate csv as text/csv or octet-stream; reject obvious HTML.
        raw = r.content
        head = raw[:200].lstrip().lower()
        if head.startswith(b"<html") or b"<html" in head:
            # try toggling the download flag (some deployments differ)
            alt = url.replace("&download=1", "&download=0") if "&download=1" in url else url.replace("&download=0", "&download=1")
            r = http_get_with_retry(alt, s)
            raw = r.content
            head = raw[:200].lstrip().lower()
            if head.startswith(b"<html") or b"<html" in head:
                raise RuntimeError("Got HTML instead of CSV (blocked).")

        text = raw.decode("utf-8", errors="ignore")
        df = pd.read_csv(io.StringIO(text))
        if "Date" not in df.columns:
            raise RuntimeError(f"Unexpected CSV columns: {list(df.columns)[:8]}")

    if df.empty:
        return {}, {}, date.today()

    # Normalize
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns:
        raise RuntimeError(f"Unexpected CSV columns: {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    for f in funds:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")

    # Keep only rows with at least one price present
    present_cols = [f for f in funds if f in df.columns]
    df = df.dropna(subset=present_cols, how="all").sort_values("Date")

    if len(df) == 0:
        return {}, {}, date.today()

    last_row = df.iloc[-1]
    as_of = last_row["Date"]

    # Prices at last trading day
    prices: Dict[str, float] = {}
    for f in funds:
        if f in df.columns and pd.notna(last_row.get(f)):
            prices[f] = float(last_row[f])

    # % change vs prior day (only if we have at least 2 rows)
    changes: Dict[str, float] = {}
    if len(df) >= 2:
        prev_row = df.iloc[-2]
        for f in funds:
            if f in df.columns and pd.notna(last_row.get(f)) and pd.notna(prev_row.get(f)):
                prev = float(prev_row[f])
                if prev != 0.0:
                    changes[f] = round((float(last_row[f]) - prev) / prev * 100.0, 2)

    return prices, changes, as_of

def format_post(prices, changes, as_of, prefix="TSP Returns", multiline=False):
    if not prices:
        return f"📊 {prefix} — no new prices yet (weekend/holiday or upstream delay)."

    order = ["G Fund","F Fund","C Fund","S Fund","I Fund"]
    lines = []
    for f in order:
        if f in prices:
            p, c = prices[f], changes.get(f)
            if c is None:
                lines.append(f"{f.split()[0]}: ${p:,.2f}")
            else:
                sign = "+" if c >= 0 else ""
                lines.append(f"{f.split()[0]}: ${p:,.2f} ({sign}{c:.2f}%)")

    top = top_mover_label(changes).replace(" — ", "")
    if multiline:
        body = "\n".join(lines + ([top] if top else []))
        return f"📊 {prefix} {as_of.isoformat()}\n{body}"
    else:
        body = " | ".join(lines)
        return f"📊 {prefix} {as_of.isoformat()} — {body}{top}"
    
MAX_CHARS = 300
HASHTAGS = "#TSP #ThriftSavingsPlan"

def assemble_post(prices, changes, as_of, prefix="TSP Returns"):
    # Build all variants
    multi_top    = format_post(prices, changes, as_of, prefix=prefix, multiline=True)
    single_top   = format_post(prices, changes, as_of, prefix=prefix, multiline=False)

    # Versions without the Top mover (strip the trailing part we added)
    def strip_top(m: str) -> str:
        return m.replace("\nTop:", "\n").replace(" — Top:", "").rstrip()

    multi_no_top  = strip_top(multi_top)
    single_no_top = strip_top(single_top)

    # Try to append hashtags (with a preceding newline for multiline; space for single)
    candidates = [
        (multi_top + f"\n\n{HASHTAGS}"),
        (single_top + f"\n\n{HASHTAGS}"),
        (multi_no_top + f"\n\n{HASHTAGS}"),
        (single_no_top + f"\n\n{HASHTAGS}"),

        multi_top, single_top, multi_no_top, single_no_top
    ]

    for msg in candidates:
        if len(msg) <= MAX_CHARS:
            return msg

    # As a final fallback, hard trim 
    return candidates[-1][:MAX_CHARS-1] + "…"

# -------------------- Bluesky --------------------
HASHTAG_RE = re.compile(r'(?<!\w)#([A-Za-z0-9_]+)')

def build_hashtag_facets(text: str):
    """
    Produce app.bsky.richtext.facet entries for hashtags in `text`.
    Uses UTF-8 byte offsets as required by Bluesky.
    """
    facets = []
    for m in HASHTAG_RE.finditer(text):
        tag = m.group(1)
        # byte offsets (not codepoints!)
        byte_start = len(text[:m.start()].encode('utf-8'))
        byte_end   = byte_start + len(m.group(0).encode('utf-8'))

        facets.append(
            models.AppBskyRichtextFacet.Main(
                features=[models.AppBskyRichtextFacet.Tag(tag=tag)],
                index=models.AppBskyRichtextFacet.ByteSlice(
                    byte_start=byte_start,
                    byte_end=byte_end,
                ),
            )
        )
    return facets

def post_bsky(text: str, handle: str, app_pw: str, dry_run: bool = True):
    if dry_run:
        print("[dry-run]", text); return
    try:
        client = Client()
        client.login(handle, app_pw)

        facets = build_hashtag_facets(text)
        if facets:
            client.send_post(text=text, facets=facets)
        else:
            client.send_post(text=text)

        print("[info] Posted to Bluesky with facets:", bool(facets))
    except Exception as e:
        raise SystemExit(f"Bluesky post failed: {e}")

# -------------------- X/Twitter --------------------
def _init_twitter_client():
    k  = os.getenv("TWITTER_API_KEY")
    ks = os.getenv("TWITTER_API_SECRET")
    t  = os.getenv("TWITTER_ACCESS_TOKEN")
    ts = os.getenv("TWITTER_ACCESS_SECRET")
    if not all([k, ks, t, ts]):
        return None
    try:
        return tweepy.Client(
            consumer_key=k,
            consumer_secret=ks,
            access_token=t,
            access_token_secret=ts,
            wait_on_rate_limit=True,
        )
    except Exception as e:
        print(f"[twitter] init failed: {e}")
        return None

def _split_for_twitter(text: str, limit: int = 280):
    if len(text) <= limit:
        return [text]
    words, chunks, cur = text.split(), [], []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        # reserve ~6 chars for " (1/10)" suffix if needed
        if len(trial) + 6 <= limit:
            cur.append(w)
        else:
            chunks.append(" ".join(cur))
            cur = [w]
    if cur:
        chunks.append(" ".join(cur))
    if len(chunks) > 1:
        total = len(chunks)
        chunks = [f"{c} ({i}/{total})" for i, c in enumerate(chunks, 1)]
    return chunks

def post_twitter(text: str, dry_run: bool = True):
    client = _init_twitter_client()
    if client is None:
        print("[twitter] Skipping: missing API credentials")
        return None
    if dry_run:
        print(f"[dry-run][twitter] {text}")
        return None
    try:
        chunks = _split_for_twitter(text, limit=280)
        root_id = None
        for i, chunk in enumerate(chunks):
            if i == 0:
                resp = client.create_tweet(text=chunk)
                root_id = resp.data.get("id")
            else:
                resp = client.create_tweet(text=chunk, in_reply_to_tweet_id=root_id)
            time.sleep(0.3)
        print("[info] Posted to X/Twitter")
        return root_id
    except tweepy.TweepyException as e:
        print(f"[twitter] post failed: {e}")
        return None

# -------------------- Main --------------------

if __name__ == "__main__":
    load_dotenv(override=False)
    ALLOW_WEEKEND = os.getenv("ALLOW_WEEKEND", "false").lower() in {"1","true","yes","on"}

    # Skip Sat/Sun unless explicitly allowed
    if (date.today().weekday() in (5, 6)) and not ALLOW_WEEKEND:
        print("[info] Weekend: skipping post.")
        raise SystemExit(0)

    FUNDS = get_funds_from_env()
    WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "30"))
    PREFIX = os.getenv("POST_PREFIX", "TSP Returns")
    DRY_RUN = os.getenv("DRY_RUN", "true").lower() in {"1", "true", "yes", "on"}

    BLSKY_HANDLE = os.getenv("BLSKY_HANDLE", "").strip()
    BLSKY_APP_PW = os.getenv("BLSKY_APP_PW", "").strip()

    if not BLSKY_HANDLE or not BLSKY_APP_PW:
        if not DRY_RUN:
            raise SystemExit("Set BLSKY_HANDLE and BLSKY_APP_PW in .env or run with DRY_RUN=true")

    try:
        prices, changes, as_of = fetch_prices_and_changes_from_csv_dynamic(FUNDS, window_days=WINDOW_DAYS)
        msg_multi  = format_post(prices, changes, as_of, multiline=True)
        msg_single = format_post(prices, changes, as_of, multiline=False)
        msg = assemble_post(prices, changes, as_of, prefix=PREFIX)
        post_bsky(msg, BLSKY_HANDLE, BLSKY_APP_PW, dry_run=DRY_RUN)
        # Mirror to X/Twitter
        post_twitter(msg, dry_run=DRY_RUN)
    except Exception as e:
        print(f"[error] {e}")
        # raise  # uncomment to fail CI on error
