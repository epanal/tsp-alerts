"""
TSP Alerts — Daily fund % changes posted to Bluesky.
Data source: Official TSP CSV (tsp.gov) over a rolling window (default 30 days).

Setup:
  pip install pandas python-dotenv requests atproto

Run:
  python tspalerts.py   # DRY_RUN=true by default (set in .env)
"""

from datetime import date, timedelta
import io
import os
from typing import Dict, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from atproto import Client

# Skip Sat(5)/Sun(6)
if date.today().weekday() in (5, 6):
    print("[info] Weekend: skipping post.")
    raise SystemExit(0)

# -------------------- Config helpers --------------------

DEFAULT_FUNDS = ["G Fund", "F Fund", "C Fund", "S Fund", "I Fund"]

def get_funds_from_env() -> List[str]:
    raw = os.getenv("FUNDS", "").strip()
    if not raw:
        return DEFAULT_FUNDS
    return [x.strip() for x in raw.split(",") if x.strip()]

# -------------------- Core logic --------------------


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

def build_tsp_csv_url(window_days=30, include_L=True, include_inv=True):
    end = date.today()
    start = end - timedelta(days=window_days)
    return (
        "https://www.tsp.gov/data/fund-price-history.csv"
        f"?startdate={start:%Y-%m-%d}"
        f"&enddate={end:%Y-%m-%d}"
        f"&Lfunds={'1' if include_L else '0'}"
        f"&InvFunds={'1' if include_inv else '0'}"
        "&download=0"
    )

def fetch_changes_from_csv_dynamic(funds=("G Fund","F Fund","C Fund","S Fund","I Fund"), window_days=30):
    url = build_tsp_csv_url(window_days)
    with requests.Session() as s:
        s.headers.update(BROWSER_HEADERS)
        r = s.get(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        # If we were blocked, servers often return HTML, not CSV
        if r.headers.get("Content-Type","").lower().startswith("text/html"):
            raise RuntimeError("Got HTML instead of CSV (blocked).")
        df = pd.read_csv(io.StringIO(r.text))
    if df.empty or len(df) < 2:
        return {}, date.today()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date")
    last, prev = df.iloc[-1], df.iloc[-2]
    as_of = last["Date"]
    changes = {}
    for f in funds:
        if f in df.columns and pd.notna(last[f]) and pd.notna(prev[f]) and float(prev[f]) != 0.0:
            changes[f] = round((float(last[f]) - float(prev[f])) / float(prev[f]) * 100.0, 2)
    return changes, as_of


def format_post(changes: Dict[str, float], as_of: date, prefix: str = "TSP Returns") -> str:
    """Produce a compact, Bluesky-friendly line."""
    if not changes:
        return f"📊 {prefix} — no new prices yet (weekend/holiday or upstream delay)."

    # G/F/C/S/I first, then any L funds if included in FUNDS
    head = ["G Fund", "F Fund", "C Fund", "S Fund", "I Fund"]
    tail = sorted(k for k in changes.keys() if k not in set(head))
    order = [k for k in head if k in changes] + tail

    parts = [f"{k.split()[0]}: {changes[k]:+,.2f}%" for k in order]
    return f"📊 {prefix} {as_of.isoformat()} — " + " | ".join(parts)


# -------------------- Bluesky --------------------

def post_bsky(text: str, handle: str, app_pw: str, dry_run: bool = True):
    if dry_run:
        print("[dry-run]", text)
        return
    client = Client()
    client.login(handle, app_pw)
    client.send_post(text)
    print("[info] Posted to Bluesky.")


# -------------------- Main --------------------

if __name__ == "__main__":
    load_dotenv(override=True)

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
        changes, as_of = fetch_changes_from_csv_dynamic(FUNDS, window_days=WINDOW_DAYS)
        msg = format_post(changes, as_of, prefix=PREFIX)
        post_bsky(msg, BLSKY_HANDLE, BLSKY_APP_PW, dry_run=DRY_RUN)
    except Exception as e:
        # Keep failures visible in logs without crashing silently
        print(f"[error] {e}")
        # You can exit non-zero in CI if desired:
        # raise
