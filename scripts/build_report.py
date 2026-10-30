#!/usr/bin/env python3
"""
Pre-Market Bias Report  ·  builds docs/data/latest.json

POINT-IN-TIME BY DESIGN
  Every read is built only from data that existed at the 9:25 AM ET cutoff of
  its session. Run it at 7 AM and it uses live pre-market data. Run it at
  2 PM or 9 PM and it rewinds every price to 9:25 AM that morning, so the
  session itself never leaks in and the track record is an honest test.
  Pass REPORT_DATE=YYYY-MM-DD to rebuild (backfill) a past morning.

MOSTLY MACRO
  About three quarters of the score is macro and volatility: overnight futures,
  global equities, Treasury futures (10Y and 2Y), the dollar, crude, credit,
  crypto risk appetite, pre-market vol and the VIX curve. Dealer positioning
  (gamma, charm, vanna) is a lighter overlay, used only when the option
  snapshot is valid for that morning.

Free data only: Yahoo Finance (yfinance), CBOE delayed option chains,
ForexFactory's weekly calendar feed, and the Nasdaq earnings calendar.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import traceback
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests
from scipy.stats import norm

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data"
FIXTURES = os.environ.get("REPORT_FIXTURES")      # offline test mode
UA = {"User-Agent": "Mozilla/5.0 (premarket-report; personal research)"}
RISK_FREE = 0.04
CUTOFF = time(9, 25)                              # evidence cutoff, ET
NOTES: list[str] = []
SOURCES: dict[str, dict] = {}

MARKETS = {
    "NQ": {
        "name": "Nasdaq-100 futures", "future": "NQ=F", "chains": ["_NDX", "QQQ"],
        "vol_index": "^VXN", "vol_label": "VXN", "bin": 25, "rates_weight": 1.0,
        "megacaps": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
                     "TSLA", "COST", "NFLX", "AMD"],
    },
    "ES": {
        "name": "S&P 500 futures", "future": "ES=F", "chains": ["_SPX", "SPY"],
        "vol_index": "^VIX", "vol_label": "VIX", "bin": 5, "rates_weight": 0.75,
        "megacaps": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
                     "TSLA", "BRK.B", "JPM", "LLY", "V", "XOM", "UNH"],
    },
}
# Everything here trades (or has a pre-market) before 9:25 ET, except the vol
# indices, which are only used at their prior close (known before the open).
MACRO_TICKERS = ["NQ=F", "ES=F", "ZN=F", "ZT=F", "DX-Y.NYB", "CL=F", "HYG",
                 "BTC-USD", "VXX", "^N225", "^GDAXI", "^HSI",
                 "^VIX", "^VIX3M", "^VXN", "^TNX"]
GLOBAL = {"^N225": "Nikkei", "^GDAXI": "DAX", "^HSI": "Hang Seng"}
PEER_WEIGHT = 0.35   # NQ and ES are ~0.9 correlated day to day


def note(msg: str) -> None:
    print("NOTE:", msg, file=sys.stderr)
    NOTES.append(msg)


def stamp(name, ok, data_time=None, detail=""):
    SOURCES[name] = {"ok": ok, "fetched": datetime.now(ET).isoformat(timespec="seconds"),
                     "data_time": data_time, "detail": detail}


def prev_weekday(d: date) -> date:
    d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def pct(a, b):
    return (a / b - 1) * 100


# ─────────────────────────── session / cutoff ─────────────────────────────
OPEN_MOVE_END = time(10, 0)   # pre-open call is graded on the opening move, 9:30 → 10:00 ET


def current_session(now: datetime):
    """
    Which futures session is trading right now, as (name, start, end, ny_date).
    Asia 6 PM → 2 AM, London 2 AM → 9:30 AM, NY 9:30 AM → 4 PM (ET). Between 4 and
    6 PM it calls the coming Asia session. ny_date is the NY day the session feeds
    into, which sets the "yesterday's 4 PM" baseline. None when futures are shut.
    """
    wd, t, d = now.weekday(), now.time(), now.date()
    at = lambda day, hh, mm=0: datetime.combine(day, time(hh, mm), ET)  # noqa: E731
    if wd == 5 or (wd == 6 and t < time(18)) or (wd == 4 and t >= time(16)):
        return None
    if time(9, 30) <= t < time(16):
        return "NY", at(d, 9, 30), at(d, 16), d
    if t >= time(16):
        return "Asia", at(d, 18), at(d + timedelta(days=1), 2), next_weekday(d)
    if t < time(2):
        return "Asia", at(d - timedelta(days=1), 18), at(d, 2), d
    return "London", at(d, 2), at(d, 9, 30), d

def resolve_session(now: datetime):
    """Which morning are we reading, and is the data live or rewound?"""
    forced = os.environ.get("REPORT_DATE", "").strip()
    if forced:
        target = date.fromisoformat(forced)
    else:
        target = now.date()
        while target.weekday() >= 5:          # weekend → last Friday's morning
            target -= timedelta(days=1)
    cutoff = datetime.combine(target, CUTOFF, ET)
    live = now < cutoff
    as_of = now if live else cutoff
    return target, as_of, live


# ───────────────────────────── data fetchers ──────────────────────────────
def _get_json(url, fixture):
    if FIXTURES:
        p = Path(FIXTURES) / fixture
        return json.loads(p.read_text()) if p.exists() else None
    r = requests.get(url, headers=UA, timeout=45)
    r.raise_for_status()
    return r.json()


def fetch_quotes(tickers, target: date, as_of: datetime, marks: dict | None = None):
    """
    For each ticker: last = last bar at or before `as_of`, prev = last bar at
    or before 4:00 PM ET of the prior weekday. Intraday bars include pre- and
    post-market, so nothing after the cutoff is ever used.
    """
    if FIXTURES:
        p = Path(FIXTURES) / "quotes.json"
        out = json.loads(p.read_text()) if p.exists() else {}
        out = {k: dict(v) for k, v in out.items() if k in tickers}
        for v in out.values():   # fixtures: fake marks halfway between prev and last
            v["marks"] = {m: (v["prev"] + v["last"]) / 2 for m in (marks or {})}
        stamp("Yahoo prices", bool(out), as_of.isoformat(), f"{len(out)} of {len(tickers)} tickers")
        return out
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        note("yfinance missing: macro prices skipped")
        stamp("Yahoo prices", False)
        return {}

    age = (datetime.now(ET).date() - target).days
    interval = "1m" if age <= 6 else "5m" if age <= 55 else "60m"
    start = prev_weekday(prev_weekday(target)) - timedelta(days=1)
    end = target + timedelta(days=1)
    prev_mark = datetime.combine(prev_weekday(target), time(16, 0), ET)
    out, missing = {}, []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start=start, end=end, interval=interval,
                                     prepost=True, auto_adjust=False)
            if h.empty:
                raise ValueError("no bars")
            h.index = h.index.tz_convert(ET)
            cur = h[h.index <= as_of]
            base = h[h.index <= prev_mark]
            if cur.empty or base.empty:
                raise ValueError("no bars near cutoff")
            last_ts = cur.index[-1]
            q = {"last": float(cur["Close"].iloc[-1]), "prev": float(base["Close"].iloc[-1]),
                 "ts": last_ts.isoformat()}
            # a "last" older than the baseline means it hasn't traded since: no change
            q["fresh"] = last_ts > prev_mark
            q["marks"] = {}
            for name, when in (marks or {}).items():
                mk = h[h.index <= when]
                if not mk.empty and mk.index[-1] > prev_mark:
                    q["marks"][name] = float(mk["Close"].iloc[-1])
            out[t] = q
        except Exception as e:  # noqa: BLE001
            missing.append(t)
            print(f"quote {t}: {type(e).__name__} {e}", file=sys.stderr)
    if missing:
        note("No price for " + ", ".join(missing))
    stamp("Yahoo prices", bool(out), as_of.isoformat(),
          f"{len(out)} of {len(tickers)} tickers · {interval} bars, cut at {as_of:%-I:%M %p} ET")
    return out


def fetch_chain(symbol):
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
    name = f"CBOE {symbol.lstrip('_')} options"
    try:
        js = _get_json(url, f"cboe_{symbol}.json")
        if not js:
            raise RuntimeError("empty chain")
    except Exception:
        stamp(name, False)
        raise
    d = js["data"]
    ts = None
    try:
        if js.get("timestamp"):
            ts = datetime.fromisoformat(str(js["timestamp"]))
            # CBOE stamps are UTC. Reading them as ET pushed every evening snapshot
            # into "tomorrow", and the leak guard then threw the whole chain away.
            ts = (ts if ts.tzinfo else ts.replace(tzinfo=ZoneInfo("UTC"))).astimezone(ET)
    except Exception:  # noqa: BLE001
        pass
    stamp(name, True, ts and ts.isoformat(), f"{len(d.get('options', [])):,} contracts")
    spot = float(d.get("current_price") or d.get("close"))
    prev = float(d.get("prev_day_close") or d.get("close") or spot)
    return spot, prev, d["options"], ts


WEEK_EVENTS: list[dict] = []


_CAL: list | None = None
CAL_CACHE = OUT / "calendar_cache.json"


# Indicators from the macro notes that have no free official feed. The calendar lists each
# release's previous value, which is the latest actual, so we keep a running record of them.
CAL_GAUGE_TITLES = {"ISM Manufacturing PMI", "ISM Services PMI", "Chicago PMI", "CB Consumer Confidence"}
READINGS = OUT / "macro_readings.json"


def remember_readings(feed):
    try:
        store = json.loads(READINGS.read_text()) if READINGS.exists() else {}
    except Exception:  # noqa: BLE001
        store = {}
    today = datetime.now(ET).date().isoformat()
    for ev in feed:
        title = ev.get("title", "")
        if ev.get("country") != "USD" or title not in CAL_GAUGE_TITLES:
            continue
        val = _num(ev.get("previous"))
        if val is None:
            continue
        rows = store.setdefault(title, [])
        rel = str(ev.get("date", ""))[:10]
        if not any(r["release"] == rel for r in rows):
            rows.append({"release": rel, "previous": val, "seen": today})
        store[title] = sorted(rows, key=lambda r: r["release"])[-24:]
    try:
        READINGS.write_text(json.dumps(store, indent=1))
    except Exception:  # noqa: BLE001
        pass


def latest_reading(title, as_of):
    """Most recent known actual for a calendar-sourced indicator, public by as_of."""
    try:
        rows = json.loads(READINGS.read_text()).get(title, []) if READINGS.exists() else []
    except Exception:  # noqa: BLE001
        return None
    rows = [r for r in rows if r["seen"] <= as_of.date().isoformat()]
    return rows[-1]["previous"] if rows else None


def _calendar_feed():
    """
    The ForexFactory feed rate-limits repeat hits, and every run used to call it three
    times. Fetch once per run, keep a copy on disk, and fall back to it on an error,
    so the event-day dampener never silently switches off.
    """
    global _CAL
    if _CAL is not None:
        return _CAL, "ok"
    try:
        js = _get_json("https://nfs.faireconomy.media/ff_calendar_thisweek.json", "calendar.json") or []
        if js and not FIXTURES:
            CAL_CACHE.write_text(json.dumps(js))
            remember_readings(js)
        _CAL = js
        return js, "ok"
    except Exception as e:  # noqa: BLE001
        if CAL_CACHE.exists():
            _CAL = json.loads(CAL_CACHE.read_text())
            return _CAL, f"cached copy ({type(e).__name__} live)"
        raise


def fetch_calendar(target):
    try:
        js, how = _calendar_feed()
    except Exception as e:  # noqa: BLE001
        note(f"Economic calendar unavailable ({type(e).__name__})")
        stamp("Economic calendar", False)
        return []
    stamp("Economic calendar", bool(js), detail="" if how == "ok" else how)
    events = []
    for ev in js:
        if ev.get("country") != "USD" or ev.get("impact") not in ("High", "Medium"):
            continue
        try:
            when = datetime.fromisoformat(ev["date"]).astimezone(ET)
        except Exception:  # noqa: BLE001
            continue
        item = {"day": when.strftime("%a %b %-d"), "date": when.date().isoformat(),
                "time": when.strftime("%-I:%M %p") if when.time() != time(0, 0) else "All day",
                "ts": when.isoformat(), "title": ev.get("title", ""), "impact": ev.get("impact"),
                "forecast": ev.get("forecast") or "", "previous": ev.get("previous") or ""}
        if not any(w["ts"] == item["ts"] and w["title"] == item["title"] for w in WEEK_EVENTS):
            WEEK_EVENTS.append(item)
        if when.date() != target:
            continue
        events.append({"time": when.strftime("%-I:%M %p") if when.time() != time(0, 0) else "All day",
                       "ts": when.isoformat(), "title": ev.get("title", ""), "impact": ev.get("impact"),
                       "forecast": ev.get("forecast") or "", "previous": ev.get("previous") or ""})
    return sorted(events, key=lambda e: e["ts"])


def fetch_earnings(day):
    label = f"Earnings {day:%b %-d}"
    try:
        js = _get_json(f"https://api.nasdaq.com/api/calendar/earnings?date={day:%Y-%m-%d}",
                       f"earnings_{day:%Y-%m-%d}.json")
        rows = ((js or {}).get("data") or {}).get("rows") or []
    except Exception as e:  # noqa: BLE001
        note(f"Earnings calendar unavailable for {day} ({type(e).__name__})")
        stamp(label, False)
        return []
    stamp(label, js is not None, detail=f"{len(rows)} companies")
    out = []
    for r in rows:
        t = (r.get("time") or "").lower()
        when = "before open" if "pre" in t else "after close" if "after" in t else "time n/a"
        out.append({"symbol": r.get("symbol", "").upper(), "when": when, "date": day.isoformat(),
                    "eps_est": _num(r.get("epsForecast")), "eps": _num(r.get("eps")),
                    "surprise": _num(r.get("surprise"))})
    return out


def _num(v):
    """'$1.23' / '(0.45)' / '12.5%' / 'N/A' → float or None."""
    if v in (None, "", "N/A", "--"):
        return None
    t = str(v).replace("$", "").replace(",", "").replace("%", "").strip()
    neg = t.startswith("(") and t.endswith(")")
    try:
        x = float(t.strip("()"))
    except ValueError:
        return None
    return -x if neg else x


# ───────────────────── official macro data (Fed reaction function) ────────────────
# Framework: the user's macro notes. The Fed reacts to three things (jobs, inflation,
# activity); hot data → contractionary Fed → bearish for stocks, cool data → expansionary
# → bullish. Only official numbers count, because those are what the Fed watches.
# Free source: FRED's public CSV download (no key). ISM, Chicago PMI and Conference
# Board confidence are licensed and not on FRED, so they can't be scored here.
_FRED: dict = {}
FRED_CACHE = OUT / "fred_cache.json"
FRED_ERR: list[str] = []
DEAD: set[str] = set()   # hosts that failed to connect this run: skip them instead of waiting again
BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                            "(KHTML, like Gecko) Version/17.5 Safari/605.1.15", "Accept": "text/csv,text/plain,*/*"}


def _fred_cache():
    try:
        return json.loads(FRED_CACHE.read_text()) if FRED_CACHE.exists() else {}
    except Exception:  # noqa: BLE001
        return {}


# Same official numbers straight from the Bureau of Labor Statistics (free API, no key),
# used when FRED can't be reached. Jobless claims come from the Labor Dept, not BLS.
BLS_IDS = {"UNRATE": "LNS14000000", "PAYEMS": "CES0000000001", "CPIAUCSL": "CUSR0000SA0", "CPIAUCNS": "CUUR0000SA0",
           "CPILFESL": "CUSR0000SA0L1E", "CPILFENS": "CUUR0000SA0L1E", "PPIFIS": "WPSFD4", "PPIFES": "WPSFD49104"}


def bls(series):
    if "bls" in DEAD:
        return []
    try:
        r = requests.get(f"https://api.bls.gov/publicAPI/v1/timeseries/data/{BLS_IDS[series]}", headers=BROWSER_UA, timeout=(5, 12))
        r.raise_for_status()
        js = r.json()
        rows = []
        for item in (js.get("Results", {}).get("series") or [{}])[0].get("data", []):
            per = item.get("period", "")
            if not per.startswith("M") or per == "M13":
                continue
            try:
                rows.append((date(int(item["year"]), int(per[1:]), 1), float(item["value"])))
            except (ValueError, KeyError):
                continue
        return sorted(rows)
    except Exception as e:  # noqa: BLE001
        FRED_ERR.append(f"BLS {series}: {type(e).__name__}")
        print(f"bls {series}: {type(e).__name__} {e}", file=sys.stderr)
        if isinstance(e, (requests.ConnectionError, requests.Timeout)):
            DEAD.add("bls")
        return []


def fred(series):
    """
    [(date, value)] oldest → newest. Tries FRED's CSV download, then its plain-text file.
    Official data changes once a month, so the last good pull is kept on disk and used
    whenever FRED is unreachable from the runner.
    """
    if series in _FRED:
        return _FRED[series]
    out = []
    key = os.environ.get("FRED_API_KEY", "").strip()
    vintage = os.environ.get("REPORT_DATE", "").strip()     # backfill: ask ALFRED for the data as it stood that day
    if key and not FIXTURES and "fredapi" not in DEAD:     # official FRED API first (free key, GitHub secret)
        try:
            params = {"series_id": series, "api_key": key, "file_type": "json",
                      "observation_start": (datetime.now(ET).date() - timedelta(days=1500)).isoformat()}
            if vintage:
                params.update(realtime_start=vintage, realtime_end=vintage)
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params=params, headers=BROWSER_UA, timeout=(5, 15))
            r.raise_for_status()
            for o in r.json().get("observations", []):
                try:
                    out.append((date.fromisoformat(o["date"]), float(o["value"])))
                except (ValueError, KeyError):
                    continue
            out.sort()
        except Exception as e:  # noqa: BLE001
            FRED_ERR.append(f"FRED API {series}: {type(e).__name__} {str(e)[:60].replace(key, '***')}")
            print(f"fred api {series}: {type(e).__name__}", file=sys.stderr)
            if isinstance(e, (requests.ConnectionError, requests.Timeout)) or getattr(getattr(e, "response", None), "status_code", 0) in (400, 401, 403):
                DEAD.add("fredapi")
            out = []
    api_ok = bool(out)
    if not FIXTURES:
        for url in (() if api_ok else (f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
                    f"https://fred.stlouisfed.org/data/{series}.txt")):
            if "fred" in DEAD:
                break
            try:
                r = requests.get(url, headers=BROWSER_UA, timeout=(5, 12))
                r.raise_for_status()
                lines = r.text.strip().splitlines()
                if url.endswith(".txt"):
                    lines = lines[next(i for i, ln in enumerate(lines) if ln.upper().startswith("DATE")) + 1:]
                else:
                    lines = lines[1:]
                for line in lines:
                    parts = line.replace(",", " ").split()
                    if len(parts) < 2:
                        continue
                    try:
                        out.append((date.fromisoformat(parts[0]), float(parts[1])))
                    except ValueError:
                        continue          # "." marks a missing value
                if len(out) > 12:
                    break
                out = []
            except Exception as e:  # noqa: BLE001
                FRED_ERR.append(f"{series}: {type(e).__name__} {str(e)[:80]}")
                print(f"fred {series} {url}: {type(e).__name__} {e}", file=sys.stderr)
                if isinstance(e, (requests.ConnectionError, requests.Timeout)):
                    DEAD.add("fred")
        if not out and series in BLS_IDS:
            out = bls(series)
        cache = _fred_cache()
        if out and not os.environ.get("REPORT_DATE", "").strip():   # never overwrite live data with a vintage
            cache[series] = [[d.isoformat(), v] for d, v in out[-400:]]
            try:
                FRED_CACHE.write_text(json.dumps(cache))
            except Exception:  # noqa: BLE001
                pass
        elif series in cache:
            out = [(date.fromisoformat(d), v) for d, v in cache[series]]
    _FRED[series] = out
    return out


def _yoy(rows, i=-1):
    return pct(rows[i][1], rows[i - 12][1]) if len(rows) >= 13 - i - 1 else None


def _mom(rows, i=-1):
    return pct(rows[i][1], rows[i - 1][1]) if len(rows) >= 2 - i else None


def _upto(rows, as_of, lag_days):
    """Observations already published by as_of (a reference period is public ~lag_days after it starts)."""
    return [r for r in rows if r[0] + timedelta(days=lag_days) <= as_of.date()]


# name: (series, how to read it, hot test, cool test, unit, publication lag in days)
# hot  → Fed leans contractionary → -1 for stocks; cool → expansionary → +1
FED_GAUGES = {
    "Unemployment rate": ("UNRATE", lambda r: r[-1][1], lambda x: x < 4.0, lambda x: x > 5.5, "%", 36),
    "Initial jobless claims": ("ICSA", lambda r: r[-1][1] / 1000,
                                          lambda x: x < 250, lambda x: x > 350, "K", 5),
    "Payrolls (3-mo avg change)": ("PAYEMS", lambda r: (r[-1][1] - r[-4][1]) / 3,
                                   lambda x: x > 200, lambda x: x < 75, "K", 36),
    "CPI y/y": ("CPIAUCNS", _yoy, lambda x: x > 2.2, lambda x: x < 1.8, "%", 45),
    "Core CPI y/y": ("CPILFENS", _yoy, lambda x: x > 2.2, lambda x: x < 1.8, "%", 45),
    "PPI m/m": ("PPIFIS", _mom, lambda x: x > 0.2, lambda x: x < 0.0, "%", 45),
}


# PMIs: over 55 is strong expansion, under 50 contraction. Confidence: over 120 strong, under 100 weak.
CAL_GAUGES = {"ISM Manufacturing PMI": (55, 50, "ISM manufacturing"), "ISM Services PMI": (55, 50, "ISM services"),
              "Chicago PMI": (55, 50, "Chicago PMI"), "CB Consumer Confidence": (120, 100, "Consumer confidence")}


def _at(rows, d):
    """Latest value dated on or before d."""
    v = None
    for dd, x in rows:
        if dd > d:
            break
        v = x
    return v


def liquidity_signals(cfg, as_of):
    """
    Macro-desk inputs from FRED, each lagged to when it is actually published so nothing
    from after the cutoff leaks in: daily series use yesterday's value, weekly ones their release lag.
    """
    S = []
    r = _upto(fred("DFII10"), as_of, 1)                       # 10-year TIPS: the real rate tech is priced off
    if len(r) > 6:
        d = (r[-1][1] - r[-6][1]) * 100
        sc = -1 if d > 10 else 1 if d < -10 else 0
        S.append(sig("realyield", "Real 10-year yield", sc, 1.0 * cfg["rates_weight"],
                     f"10-year TIPS yield {r[-1][1]:.2f}%, {d:+.0f} bp over the last 5 sessions. " +
                     ("Rising real rates compress long-duration valuations, tech hardest." if sc < 0 else
                      "Falling real rates support valuations." if sc > 0 else "Real rates are steady."), "Macro"))
    r = _upto(fred("BAMLH0A0HYM2"), as_of, 1)                 # ICE BofA high-yield option-adjusted spread
    if len(r) > 6:
        d = (r[-1][1] - r[-6][1]) * 100
        sc = -1 if d > 15 else 1 if d < -15 else 0
        S.append(sig("hyoas", "High-yield spreads", sc, 0.75,
                     f"High-yield spread {r[-1][1]:.2f}%, {d:+.0f} bp over 5 sessions. " +
                     ("Widening spreads mean credit is pricing more risk, which equities usually follow." if sc < 0 else
                      "Tightening spreads mean credit is comfortable taking risk." if sc > 0 else "Credit spreads are steady."), "Macro"))
    r = _upto(fred("NFCI"), as_of, 6)                          # Chicago Fed financial conditions, weekly
    if len(r) > 5:
        lvl, d = r[-1][1], r[-1][1] - r[-5][1]
        sc = -1 if d > .05 else 1 if d < -.05 else 0
        S.append(sig("nfci", "Financial conditions", sc, 0.75,
                     f"Chicago Fed NFCI {lvl:+.2f} ({'tighter' if lvl > 0 else 'looser'} than average), {d:+.2f} over 4 weeks. " +
                     ("Conditions are tightening." if sc < 0 else "Conditions are easing." if sc > 0 else "Conditions are steady."), "Macro"))
    walcl, tga, rrp = _upto(fred("WALCL"), as_of, 2), _upto(fred("WTREGEN"), as_of, 2), _upto(fred("RRPONTSYD"), as_of, 1)
    if len(walcl) > 5 and tga and rrp:                          # Fed balance sheet minus TGA minus reverse repo, $ millions
        def net(i):
            d0, w = walcl[i]
            t, q = _at(tga, d0), _at(rrp, d0)
            return None if t is None or q is None else w - t - q * 1000
        now, before = net(-1), net(-5)
        if now is not None and before is not None:
            chg = (now - before) / 1000
            sc = 1 if chg > 50 else -1 if chg < -50 else 0
            S.append(sig("liquidity", "Fed net liquidity", sc, 0.75,
                         f"Fed balance sheet minus the Treasury account and reverse repo: ${now/1e6:,.2f}T, {chg:+,.0f}B over 4 weeks. " +
                         ("More dollars in the system tends to lift risk assets." if sc > 0 else
                          "Liquidity is draining, a headwind for risk assets." if sc < 0 else "Liquidity is roughly flat."), "Macro"))
    return S


def fed_pressure_signal(as_of):
    """Where the official data sits vs the Fed's comfort zones: net hawkish or dovish."""
    votes, parts = [], []
    for name, (sid, read, hot, cool, unit, lag) in FED_GAUGES.items():
        rows = _upto(fred(sid), as_of, lag)
        try:
            x = read(rows) if rows else None
        except (IndexError, ZeroDivisionError):
            x = None
        if x is None:
            continue
        v = -1 if hot(x) else 1 if cool(x) else 0
        votes.append(v)
        parts.append(f"{name} {x:,.1f}{unit} ({'hot' if v < 0 else 'cool' if v > 0 else 'in range'})")
    for title, (hot_at, cool_at, label) in CAL_GAUGES.items():
        x = latest_reading(title, as_of)
        if x is None:
            continue
        v = -1 if x > hot_at else 1 if x < cool_at else 0
        votes.append(v)
        parts.append(f"{label} {x:,.1f} ({'hot' if v < 0 else 'cool' if v > 0 else 'in range'})")
    if len(votes) < 3:
        note("Official macro data (FRED) unavailable, Fed-pressure gauge skipped" + (f" ({FRED_ERR[0]})" if FRED_ERR else ""))
        return None
    net = sum(votes) / len(votes)
    sc = 1 if net >= 0.34 else -1 if net <= -0.34 else 0
    return sig("fedpressure", "Fed pressure (official data)", sc, 0.75,
               "; ".join(parts) + ". " +
               ("Most of the data is cool, which pushes the Fed toward easing: a tailwind." if sc > 0 else
                "Most of the data runs hot, which keeps the Fed leaning tight: a headwind." if sc < 0 else
                "The data is mixed, so the Fed has no strong push either way."), "Macro")


# ForexFactory release title → how to compute the official actual from FRED, and
# whether a higher-than-forecast print is hawkish (+1) or dovish (-1).
# last field: how far from forecast counts as a real miss (one tick for rates, ~noise for counts)
RELEASES = {
    "CPI m/m": ("CPIAUCSL", _mom, 1, "month", 0.1),
    "CPI y/y": ("CPIAUCNS", _yoy, 1, "month", 0.1),
    "Core CPI m/m": ("CPILFESL", _mom, 1, "month", 0.1),
    "Core CPI y/y": ("CPILFENS", _yoy, 1, "month", 0.1),
    "PPI m/m": ("PPIFIS", _mom, 1, "month", 0.1),
    "Core PPI m/m": ("PPIFES", _mom, 1, "month", 0.1),
    "Unemployment Rate": ("UNRATE", lambda r: r[-1][1], -1, "month", 0.1),
    "Non-Farm Employment Change": ("PAYEMS", lambda r: r[-1][1] - r[-2][1], 1, "month", 25),
    "Unemployment Claims": ("ICSA", lambda r: r[-1][1] / 1000, -1, "week", 7),
    "Core PCE Price Index m/m": ("PCEPILFE", _mom, 1, "month", 0.1),     # the Fed's preferred inflation gauge
    "PCE Price Index m/m": ("PCEPI", _mom, 1, "month", 0.1),
    "Advance GDP q/q": ("A191RL1Q225SBEA", lambda r: r[-1][1], 1, "quarter", 0.3),
    "Prelim GDP q/q": ("A191RL1Q225SBEA", lambda r: r[-1][1], 1, "quarter", 0.3),
    "Final GDP q/q": ("A191RL1Q225SBEA", lambda r: r[-1][1], 1, "quarter", 0.3),
}


def _fresh_for(rows, released: date, kind):
    """The newest observation must be the period this release reports, or FRED hasn't updated yet."""
    if not rows:
        return False
    obs = rows[-1][0]
    if kind == "week":
        return 0 <= (released - obs).days <= 8
    if kind == "quarter":   # quarterly series are dated at the start of the quarter they cover
        return 60 <= (released - obs).days <= 150
    prev_month = (released.replace(day=1) - timedelta(days=1)).replace(day=1)
    return obs == prev_month


def surprise_signal(events, as_of):
    """Releases already out by the cutoff: official actual vs the consensus forecast."""
    hits, parts = [], []
    for e in events:
        spec = RELEASES.get(e["title"])
        if not spec or not e.get("forecast") or e["ts"] > as_of.isoformat():
            continue
        sid, read, hawk, kind, tol = spec
        released = datetime.fromisoformat(e["ts"]).date()
        rows = [r for r in fred(sid) if r[0] <= released]
        if not _fresh_for(rows, released, kind):
            continue
        try:
            actual = read(rows)
        except (IndexError, ZeroDivisionError):
            continue
        fc = _num(e["forecast"].replace("K", "").replace("M", ""))
        if fc is None:
            continue
        diff = round(actual, 1) - fc
        if abs(diff) < tol - 1e-9:
            v = 0
        else:
            v = -1 if diff * hawk > 0 else 1          # hotter than expected → hawkish → bearish
        hits.append(v)
        unit = "%" if "%" in e["forecast"] else "K"
        parts.append(f"{e['title']} {actual:,.1f}{unit} vs {fc:,.1f}{unit} forecast "
                     f"({'hot' if v < 0 else 'cool' if v > 0 else 'in line'})")
    if not hits:
        return None
    net = sum(hits)
    sc = 1 if net > 0 else -1 if net < 0 else 0
    return sig("surprise", "Data surprise", sc, 1.5,
               "; ".join(parts) + ". " +
               ("Cooler than expected: the market prices an easier Fed." if sc > 0 else
                "Hotter than expected: the market prices a tighter Fed." if sc < 0 else
                ("Hot and cool surprises cancel out." if any(hits) else
                 "In line with expectations, so no repricing of the Fed.")), "Macro")



# ─────────────────────────── option math helpers ──────────────────────────
OCC = re.compile(r"^([A-Z]+?)(\d{6})([CP])(\d{8})$")
AM_SETTLED = {"SPX", "NDX"}


def parse_chain(options, as_of, max_days=60):
    rows = []
    for o in options:
        m = OCC.match(o.get("option", ""))
        if not m:
            continue
        root, ymd, cp, k = m.groups()
        exp_d = datetime.strptime(ymd, "%y%m%d").date()
        exp = datetime.combine(exp_d, time(9, 30) if root in AM_SETTLED else time(16, 0), ET)
        days = (exp - as_of).total_seconds() / 86400
        if days <= 0 or days > max_days:
            continue
        oi = float(o.get("open_interest") or 0)
        if oi <= 0:
            continue
        bid, ask = float(o.get("bid") or 0), float(o.get("ask") or 0)
        mid = (bid + ask) / 2 if ask > 0 else float(o.get("last_trade_price") or 0)
        rows.append((float(k) / 1000, days / 365, float(o.get("iv") or 0), oi,
                     float(o.get("volume") or 0), 1 if cp == "C" else -1, mid))
    if not rows:
        raise RuntimeError("no usable contracts")
    K, T, IV, OI, VOL, CP, MID = map(np.array, zip(*rows))
    good = (IV > 0.01) & (IV < 3)
    IV = np.where(good, IV, np.median(IV[good]) if good.any() else 0.2)
    return dict(K=K, T=T, IV=IV, OI=OI, VOL=VOL, CP=CP, MID=MID)


def bs_d1(S, K, T, iv):
    T = np.maximum(T, 1e-6)
    return (np.log(S / K) + (RISK_FREE + 0.5 * iv**2) * T) / (iv * np.sqrt(T))


def bs_gamma(S, K, T, iv):
    T = np.maximum(T, 1e-6)
    return norm.pdf(bs_d1(S, K, T, iv)) / (S * iv * np.sqrt(T))


def bs_delta(S, K, T, iv, cp):
    nd = norm.cdf(bs_d1(S, K, T, iv))
    return np.where(cp > 0, nd, nd - 1)


# Hull & White (2017), "Optimal delta hedging for options" (SSRN 2658343): index implied vol falls
# when the index rises, so the hedge that actually minimises variance is
#   delta_MV = delta_BS + vega/(S*sqrt(T)) * (a + b*delta_BS + c*delta_BS^2)
# and vega/(S*sqrt(T)) is just the normal density at d1. Parameters read off their Figure 1
# (S&P 500, 2013-2015 estimates). Calls get under-hedged, puts over-hedged, vs Black-Scholes.
MV_CALL = (-0.21, 0.33, -0.31)
MV_PUT = (-0.21, -0.43, -0.48)


def mv_delta(S, K, T, iv, cp):
    T = np.maximum(T, 1e-6)
    d1 = bs_d1(S, K, T, iv)
    d = np.where(cp > 0, norm.cdf(d1), norm.cdf(d1) - 1)
    a = np.where(cp > 0, MV_CALL[0], MV_PUT[0]); b = np.where(cp > 0, MV_CALL[1], MV_PUT[1])
    c = np.where(cp > 0, MV_CALL[2], MV_PUT[2])
    return d + norm.pdf(d1) * (a + b * d + c * d * d)


def dealer_gex(S, c, T=None):
    T = c["T"] if T is None else T
    S = np.asarray(S, dtype=float)[..., None]
    return (c["CP"] * bs_gamma(S, c["K"], T, c["IV"]) * c["OI"] * 100 * S**2 * 0.01).sum(-1)


def dealer_delta(S, c, T=None, IV=None):
    T = c["T"] if T is None else T
    IV = c["IV"] if IV is None else IV
    d = mv_delta(S, c["K"], T, IV, c["CP"])   # minimum-variance delta, not plain Black-Scholes
    pos = np.where(c["CP"] > 0, 1.0, -1.0)
    return float((pos * d * c["OI"] * 100 * S).sum()), float((np.abs(d) * c["OI"] * 100 * S).sum())


def find_flip(grid, gex, ref):
    s = np.sign(gex)
    idx = np.where(s[:-1] * s[1:] < 0)[0]
    if not len(idx):
        return None
    pts = [grid[i] - gex[i] * (grid[i + 1] - grid[i]) / (gex[i + 1] - gex[i]) for i in idx]
    return float(min(pts, key=lambda p: abs(p - ref)))


# ───────────────────────────── positioning ────────────────────────────────
def analyse_positioning(cfg, q, target, as_of, live):
    """
    Open interest is published once per morning, so a snapshot taken any time
    on the target day carries that morning's OI. A snapshot from a later day
    would leak future positioning, so it's skipped.
    """
    fut = q.get(cfg["future"])
    if not fut:
        return None, "no futures price"
    chains = []
    for sym in cfg["chains"]:
        try:
            spot, prev, opts, ts = fetch_chain(sym)
            if ts and ts.date() > target and not (ts.date() - target).days == 0:
                return None, f"option snapshot is from {ts:%b %-d}, after this session"
            c = parse_chain(opts, as_of)
            c.update(sym=sym.lstrip("_"), ratio=fut["prev"] / prev)
            chains.append(c)
        except Exception as e:  # noqa: BLE001
            note(f"{sym} option chain unavailable ({type(e).__name__}: {e})")
    if not chains:
        return None, "option chains unavailable"

    px = fut["last"]
    hrs = max((datetime.combine(target, time(16, 0), ET) - as_of).total_seconds() / 3600, 0)
    dT = hrs / (24 * 365)
    grid = np.linspace(px * 0.94, px * 1.06, 361)
    gex_now, gex_close = np.zeros_like(grid), np.zeros_like(grid)
    tot = dict(cg=0.0, pg=0.0, D=0.0, gross=0.0, Dc=0.0, vanna=0.0)
    levels, flow, sflow = {}, dict(call=0.0, put=0.0), {}

    for c in chains:
        r = c["ratio"]
        S = px / r
        gex_now += dealer_gex(grid / r, c)
        alive = c["T"] - dT > 1e-6
        cc = {k: (v[alive] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
        gex_close += dealer_gex(grid / r, cc, T=cc["T"] - dT)
        g = bs_gamma(S, c["K"], c["T"], c["IV"]) * c["OI"] * 100 * S**2 * 0.01
        tot["cg"] += float(g[c["CP"] > 0].sum())
        tot["pg"] += float(g[c["CP"] < 0].sum())
        for k, gg, cp in zip(c["K"] * r, g, c["CP"]):
            b = round(k / cfg["bin"]) * cfg["bin"]
            levels.setdefault(b, [0.0, 0.0])[0 if cp > 0 else 1] += gg
        D, gross = dealer_delta(S, c)
        Dc, _ = dealer_delta(S, c, T=np.maximum(c["T"] - dT, 1e-7))
        Du, _ = dealer_delta(S, c, IV=c["IV"] + 0.005)
        Dd, _ = dealer_delta(S, c, IV=np.maximum(c["IV"] - 0.005, 0.01))
        tot["D"] += D; tot["gross"] += gross; tot["Dc"] += Dc; tot["vanna"] += Du - Dd
        if live:   # volume after the open would leak the session, so live only
            short = c["T"] * 365 <= 7
            prem = c["VOL"] * c["MID"] * 100
            flow["call"] += float(prem[short & (c["CP"] > 0)].sum())
            flow["put"] += float(prem[short & (c["CP"] < 0)].sum())
            for k, p, cp, s in zip(c["K"] * r, prem, c["CP"], short):
                if s and p > 0:
                    key = (round(k / cfg["bin"]) * cfg["bin"], "call" if cp > 0 else "put")
                    sflow[key] = sflow.get(key, 0) + p

    flip = find_flip(grid, gex_now, px)
    near = {k: v for k, v in levels.items() if abs(k / px - 1) < 0.06}
    above = {k: v for k, v in near.items() if k >= px} or near
    below = {k: v for k, v in near.items() if k <= px} or near
    majors = sorted(near, key=lambda k: abs(near[k][0] - near[k][1]), reverse=True)[:5]
    return dict(
        px=px, chains=[c["sym"] for c in chains],
        net_gex=float(np.interp(px, grid, gex_now)), flip=flip,
        flip_close=find_flip(grid, gex_close, px) if hrs > 0.5 else None,
        call_gamma=tot["cg"], put_gamma=tot["pg"],
        call_wall=max(above, key=lambda k: near[k][0]), put_wall=max(below, key=lambda k: near[k][1]),
        majors=[{"level": k, "net_gex": near[k][0] - near[k][1]} for k in sorted(majors)],
        dealer_delta=tot["D"], gross_delta=tot["gross"],
        charm_hedge_flow=-(tot["Dc"] - tot["D"]), vanna_per_volpt=tot["vanna"],
        flow=flow if live else None,
        top_flow=[{"level": k[0], "side": k[1], "premium": v}
                  for k, v in sorted(sflow.items(), key=lambda kv: kv[1], reverse=True)[:5]],
        profile=[{"px": round(float(x), 2), "gex": float(y)} for x, y in zip(grid[::6], gex_now[::6])],
    ), None


# ─────────────────────────────── signals ──────────────────────────────────
def sig(key, label, score, weight, text, group):
    return {"key": key, "label": label, "score": int(np.sign(score)) if score else 0,
            "raw": score, "weight": weight, "text": text, "group": group}


def chg(q, t):
    v = q.get(t)
    if not v or not v.get("fresh", True):
        return None
    return pct(v["last"], v["prev"])


def vol_shift_pts(cfg, q):
    """Pre-market vol change in index points, estimated from VXX's pre-market move."""
    vx, base = chg(q, "VXX"), q.get(cfg["vol_index"])
    if vx is None or not base:
        return None
    return base["prev"] * vx / 100 * 0.5   # VXX moves ~2x less than spot VIX in % terms


def macro_signals(cfg, q):
    S = []
    # global equities
    g = {GLOBAL[t]: chg(q, t) for t in GLOBAL if chg(q, t) is not None}
    if g:
        avg = sum(g.values()) / len(g)
        sc = 1 if avg > 0.4 else -1 if avg < -0.4 else 0
        S.append(sig("global", "Global equities", sc, 1.0,
                     ", ".join(f"{k} {v:+.1f}%" for k, v in g.items()) + ". " +
                     ("Asia and Europe are risk-on, which hands New York a tailwind." if sc > 0 else
                      "Overseas selling is a headwind going into the open." if sc < 0
                      else "Mixed overseas, so no handoff."), "Macro"))
    # 10Y futures (duration ~6.5): price % × −15 ≈ yield bp
    c = chg(q, "ZN=F")
    if c is not None:
        bp = -c * 15.4
        lvl = q.get("^TNX", {}).get("prev")
        sc = -1 if bp > 3.5 else 1 if bp < -3.5 else 0
        S.append(sig("rates", "10-year yield", sc, 1.25 * cfg["rates_weight"],
                     f"10Y futures {c:+.2f}% ≈ {bp:+.1f} bp in yield" + (f" (from {lvl:.2f}%)" if lvl else "") + ". " +
                     ("Higher discount rates compress long-duration multiples, tech hardest." if sc < 0 else
                      "Falling yields lift equity valuations." if sc > 0 else "Rates are not in play."), "Macro"))
    # 2Y futures (duration ~1.9): Fed path
    c = chg(q, "ZT=F")
    if c is not None:
        bp = -c * 52
        sc = -1 if bp > 3 else 1 if bp < -3 else 0
        S.append(sig("fed", "2-year / Fed path", sc, 0.75,
                     f"2Y futures {c:+.3f}% ≈ {bp:+.1f} bp. " +
                     ("The market is pricing a more hawkish Fed." if sc < 0 else
                      "The market is pricing in easier policy." if sc > 0 else "Fed expectations are steady."),
                     "Macro"))
    c = chg(q, "DX-Y.NYB")
    if c is not None:
        sc = -1 if c > 0.25 else 1 if c < -0.25 else 0
        S.append(sig("dollar", "US dollar", sc, 0.75,
                     f"DXY {q['DX-Y.NYB']['last']:.2f} ({c:+.2f}%). " +
                     ("A firmer dollar tightens global financial conditions." if sc < 0 else
                      "A softer dollar eases financial conditions." if sc > 0 else "Flat."), "Macro"))
    c = chg(q, "CL=F")
    if c is not None:
        sc = -1 if c > 2.5 else 0
        S.append(sig("oil", "Crude oil", sc, 0.5,
                     f"WTI {q['CL=F']['last']:.2f} ({c:+.1f}%). " +
                     ("An oil spike acts as an inflation tax and a growth drag." if sc < 0
                      else "No energy shock."), "Macro"))
    c = chg(q, "HYG")
    if c is not None:
        sc = 1 if c > 0.15 else -1 if c < -0.15 else 0
        S.append(sig("credit", "Credit (HYG)", sc, 0.75,
                     f"High-yield bonds {c:+.2f}%. " +
                     ("Credit is bid. Equity rallies backed by credit tend to stick." if sc > 0 else
                      "Credit is leaking. Equity often follows credit lower." if sc < 0
                      else "Credit is calm."), "Macro"))
    # volatility
    vx = chg(q, "VXX")
    if vx is not None:
        sc = 1 if vx < -2.5 else -1 if vx > 2.5 else 0
        S.append(sig("volpre", "Pre-market volatility", sc, 1.25,
                     f"VXX {vx:+.1f}% pre-market. " +
                     ("Fear is bleeding out. Vol sellers are back." if sc > 0 else
                      "Hedging demand is rising into the open." if sc < 0 else "Vol is steady."), "Volatility"))
    v, v3 = q.get("^VIX"), q.get("^VIX3M")
    if v and v3:
        ts = v["prev"] / v3["prev"]
        sc = -1 if ts > 1.0 else 1 if ts < 0.9 else 0
        S.append(sig("term", "VIX term structure", sc, 1.0,
                     f"VIX ÷ VIX3M at yesterday's close = {ts:.2f}. " +
                     ("Backwardation: near-term fear exceeds longer-term, and rallies tend to get sold."
                      if sc < 0 else "Healthy contango, the backdrop where dips get bought." if sc > 0
                      else "A flat-ish curve. Mild caution."), "Volatility"))
    return S


def vol_crush_signal(events, q, as_of):
    """
    After a scheduled high-impact release, implied vol tends to drop as the uncertainty resolves,
    and more so the bigger the uncertainty was (Beber & Brandt, SSRN 906748). Falling vol makes
    dealers buy back hedges (vanna), a tailwind. If vol instead rises after the print, the
    uncertainty didn't resolve. Only on days with a release that is already out.
    """
    done = [e for e in (events or []) if e.get("impact") == "High" and e["ts"] <= as_of.isoformat()
            and (as_of - datetime.fromisoformat(e["ts"])).total_seconds() <= 5 * 3600]
    vx = chg(q, "VXX")
    if not done or vx is None:
        return None
    e = done[-1]
    sc = 1 if vx < -1.5 else -1 if vx > 2.5 else 0
    return sig("crush", "Post-release vol", sc, 0.5,
               f"{e['title']} is out ({e['time']}). VXX {vx:+.1f}% on the day. " +
               ("Vol is being sold after the print: dealers buy back hedges." if sc > 0 else
                "Vol is rising after the print: the uncertainty didn't resolve." if sc < 0 else
                "Vol is little changed after the print."), "Volatility")


def overnight_signals(cfg, q):
    """Evening read for the 6 PM–8:30 AM Globex session, built from data up to 6 PM ET."""
    S = [s for s in macro_signals(cfg, q) if s["key"] not in ("global", "volpre")]   # overseas closes are stale at 6 PM
    v = q.get("^VIX")
    if v and v.get("fresh", True):
        c = pct(v["last"], v["prev"])
        sc = 1 if c < -4 else -1 if c > 4 else 0
        S.append(sig("vixday", "VIX today", sc, 1.0,
                     f"VIX {v['last']:.2f} ({c:+.1f}% on the day). " +
                     ("Fear came out of the market today." if sc > 0 else "Fear rose into the close." if sc < 0
                      else "Vol is little changed."), "Volatility"))
    return S


def fetch_bars(ticker, start: datetime, end: datetime):
    """1-minute futures bars (incl. Globex) between start and end, ET-indexed. None on failure."""
    if FIXTURES:
        return None
    try:
        import yfinance as yf
        age = (datetime.now(ET).date() - start.date()).days
        iv = "1m" if age <= 6 else "5m" if age <= 55 else "60m"
        h = yf.Ticker(ticker).history(start=start.date(), end=end.date() + timedelta(days=1),
                                      interval=iv, prepost=True, auto_adjust=False)
        if h.empty:
            return None
        h.index = h.index.tz_convert(ET)
        h = h[(h.index >= start) & (h.index <= end)]
        return h if not h.empty else None
    except Exception as e:  # noqa: BLE001
        print(f"bars {ticker}: {type(e).__name__} {e}", file=sys.stderr)
        return None


def scheduled_release_ahead(events, as_of, hours=6):
    """A high-impact US release later in this session (CPI, NFP, FOMC...)."""
    if not events or not as_of:
        return None
    end = (as_of + timedelta(hours=hours)).isoformat()
    for e in events:
        if e.get("impact") == "High" and as_of.isoformat() < e["ts"] <= end:
            return e
    return None


def dealer_signals(cfg, pos, q, events=None, as_of=None):
    S, px = [], pos["px"]
    if pos["flip"]:
        d, band = px - pos["flip"], px * 0.0015
        sc = 1 if d > band else -1 if d < -band else 0
        S.append(sig("flip", "Gamma flip", sc, 0.75,
                     f"Price is {abs(d):,.0f} pts {'above' if d > 0 else 'below'} the gamma flip ({pos['flip']:,.0f}). " +
                     ("Dealers are long gamma, so they buy dips and sell rips." if d > 0 else
                      "Dealers are short gamma, so their hedging extends moves."), "Dealer positioning"))
    tilt = pos["call_gamma"] / pos["put_gamma"] if pos["put_gamma"] else 1
    sc = 1 if tilt > 1.25 else -1 if tilt < 0.8 else 0
    S.append(sig("tilt", "Gamma tilt", sc, 0.25, f"Call ÷ put gamma {tilt:.2f}. " +
                 ("Call-heavy." if sc > 0 else "Put-heavy, so hedges sit below." if sc < 0 else "Balanced."),
                 "Dealer positioning"))
    up, dn = pos["call_wall"] - px, px - pos["put_wall"]
    loc = dn / max(up + dn, 1e-9)
    sc = 1 if loc < 0.33 else -1 if loc > 0.67 else 0
    S.append(sig("walls", "Wall range", sc, 0.5,
                 f"Put wall {pos['put_wall']:,.0f} / call wall {pos['call_wall']:,.0f}: {up:,.0f} pts up vs {dn:,.0f} down. " +
                 ("Near support." if sc > 0 else "Pressed into resistance." if sc < 0 else "Mid-range."),
                 "Dealer positioning"))
    thr = 0.0015 * pos["gross_delta"]
    ch = pos["charm_hedge_flow"]
    sc = 1 if ch > thr else -1 if ch < -thr else 0
    if abs(ch) > 0: S.append(sig("charm", "Charm into close", sc, 0.4,
                 f"Time decay shifts dealer hedges by ${abs(ch)/1e9:,.2f}B by 4 PM. " +
                 ("Dealers buy back hedges, an afternoon bid." if sc > 0 else
                  "Dealers sell, an afternoon offer." if sc < 0 else "Small."), "Dealer positioning"))
    vs = vol_shift_pts(cfg, q)
    if vs is not None:
        fv = -pos["vanna_per_volpt"] * vs
        sc = 1 if fv > thr else -1 if fv < -thr else 0
        S.append(sig("vanna", "Vanna", sc, 0.4,
                     f"Implied vol ≈ {vs:+.2f} pts pre-market, worth about ${abs(fv)/1e9:,.2f}B of dealer "
                     f"{'buying' if fv > 0 else 'selling'}.", "Dealer positioning"))
    f = pos["flow"]
    if f and f["call"] + f["put"] > 0:
        # Index option flow carries no informed signal and leans contrarian (Pan & Poteshman 2006,
        # SSRN 622869); our own graded history agreed (as-is 8/27 right, flipped 19/27). Option buying
        # also says little ahead of scheduled releases (SSRN 2544344), so it sits those out.
        pc = f["put"] / f["call"] if f["call"] else 9.9
        sc = 1 if pc > 1.25 else -1 if pc < 0.8 else 0
        ahead = scheduled_release_ahead(events, as_of)
        txt = f"Short-dated premium put/call {pc:.2f}. " + (
            "Heavy put buying: crowded hedging tends to unwind, a contrarian lean up." if sc > 0 else
            "Heavy call buying: crowded upside chasing tends to fade, a contrarian lean down." if sc < 0 else
            "Balanced.")
        if ahead and sc:
            sc = 0
            txt += f" Muted until {ahead['title']} at {ahead['time']}: flow before scheduled news isn't informative."
        S.append(sig("flow", "Options flow (contrarian)", sc, 0.25, txt, "Dealer positioning"))
    # Expiration day: gamma near the money peaks and hedging gets heavier (SSRN 7222220), so the
    # positioning votes count for more. Kept modest: the paper is a framework, not a tested result.
    if as_of and opex_kind(as_of.date()):
        for s_ in S:
            if s_["key"] in ("flip", "walls", "charm"):
                s_["weight"] = round(s_["weight"] * OPEX_BOOST, 3)
                s_["text"] += f" Weighted up for {opex_kind(as_of.date())}."
    # Dealer positioning votes at its own weights; the Record retunes it like every other signal.
    return S


OPEX_BOOST = 1.4


def opex_kind(d):
    """Third Friday = monthly expiration (quarterly in Mar/Jun/Sep/Dec)."""
    if d.weekday() != 4 or not 15 <= d.day <= 21:
        return None
    return "quarterly expiration" if d.month in (3, 6, 9, 12) else "monthly expiration"


def pin_risk(pos, as_of):
    """Long-gamma dealers near a heavy strike on expiration day tend to pin price to it."""
    if not pos or not as_of or not opex_kind(as_of.date()) or pos["net_gex"] <= 0:
        return None
    px = pos["px"]
    cands = [m for m in pos.get("majors", []) if m["net_gex"] > 0 and abs(m["level"] / px - 1) < 0.004]
    return min(cands, key=lambda m: abs(m["level"] - px))["level"] if cands else None


NEUTRAL_BAND = 0.08   # every call picks a side; under this the evidence is close to balanced,
                      # so conviction reads "Very low" (then Low < 0.30 ≤ Medium < 0.50 ≤ High).
EVENT_DAMP = 0.6      # FOMC / CPI / NFP etc. later the same day: positioning, not overnight data, drives the tape
EARNINGS_DAMP = 0.85  # mega-cap reporting after the close tonight (or last night): pre-report de-risking
LEARN_MIN_N = 20      # graded samples a signal needs before the Record starts retuning its weight


def own_score(signals):
    """Only directional signals count. Dealer positioning is shown as regime (range vs trend), not scored."""
    d = [s for s in signals if s.get("directional", True)]
    tot = sum(s["weight"] for s in d) or 1
    return max(-1.0, min(1.0, sum(s["raw"] * s["weight"] for s in d) / tot))


def label_of(x):
    """Always a side: price ends a session up or down. A tie goes to the long-run upward drift."""
    return "Bearish" if x < 0 else "Bullish"


def learned_multipliers(hist_dir: Path, sym: str, before: str, limit: int = 120) -> dict:
    """
    Scale each signal's weight by how often it has actually pointed the right way.
    Shrunk toward 50% (10 phantom right + 10 phantom wrong) so a lucky streak can't
    swing it, and only applied once a signal has LEARN_MIN_N real graded samples.
    """
    if not hist_dir.exists():
        return {}
    out = {}
    for row in record_stats(hist_dir, sym, before, limit)["signals"]:
        if row["n"] < LEARN_MIN_N:
            continue
        rate = (row["right"] + 10) / (row["n"] + 20)
        out[row["key"]] = round(min(1.5, max(0.25, 1 + 2 * (rate - 0.5))), 2)
    return out


def apply_learning(signals, mult):
    for s in signals:
        m = mult.get(s["key"])
        if m is not None and s.get("directional", True):
            s["base_weight"] = s["weight"]
            s["weight"] = round(s["weight"] * m, 3)
            s["learned"] = m
    return signals


HOLD_BAND = NEUTRAL_BAND   # an earlier call for the same window only flips once the new score clears this
# Conviction is ranked against the model's own recent reads, so the labels mean the same thing
# whatever the scale of the score: High = strongest ~10% of reads, Medium = next 20%, Low = next 35%.
CONV_PCTS = (0.35, 0.70, 0.90)
CONV_MIN_READS = 40
CONV_FALLBACK = (0.06, 0.15, 0.22)   # used until there's enough history; set from the first 74 reads


def conviction_cuts(sym, before: str):
    """|score| cut-offs for Low / Medium / High from up to 400 earlier reads of this market (all three calls)."""
    vals = []
    for sub in ("", "overnight", "now"):
        d = OUT / "history" / sub if sub else OUT / "history"
        if not d.exists():
            continue
        for f in sorted(d.glob("20*.json"))[-400:]:
            if f.stem >= before:
                continue
            try:
                m = json.loads(f.read_text()).get("markets", {}).get(sym, {})
                b = m.get("bias") or {}
                x = b.get("raw_score", b.get("score"))
                if x is not None:
                    vals.append(abs(x))
            except Exception:  # noqa: BLE001
                continue
    if len(vals) < CONV_MIN_READS:
        return CONV_FALLBACK, len(vals)
    vals.sort()
    q = lambda p: vals[min(len(vals) - 1, int(p * len(vals)))]  # noqa: E731
    return tuple(max(q(p), floor) for p, floor in zip(CONV_PCTS, (NEUTRAL_BAND * .75, 0.10, 0.15))), len(vals)


def narrate(cfg, sym, pos, signals, events, earnings, as_of, live, peer, view="preopen", sess=None, prev=None, cuts=None):
    own = own_score(signals)
    score = own if not peer else (1 - PEER_WEIGHT) * own + PEER_WEIGHT * peer[1]
    if peer and own and (score > 0) != (own > 0):
        score = own * (1 - PEER_WEIGHT)   # the peer can soften a market's own lean, never reverse it
    raw_score = score
    # pre-open: anything later today (positioning into it starts at the open);
    # now / overnight: only releases that land before the session being called ends
    end = sess[2].isoformat() if sess else None
    if view == "overnight":
        end = datetime.combine(next_weekday(as_of.date()), time(9, 30), ET).isoformat()
    upcoming = [e for e in events if e["impact"] == "High" and e["ts"] > as_of.isoformat()
                and (end is None or e["ts"] <= end)]
    mega = [e for e in earnings if e["symbol"] in cfg["megacaps"]]
    damp = []
    if upcoming:
        score *= EVENT_DAMP
        damp.append("high-impact release later today")
    bias = label_of(score)
    held = False
    if prev in ("Bullish", "Bearish") and prev != bias and abs(score) < HOLD_BAND:
        bias, held = prev, True          # hysteresis: a near-zero wobble doesn't flip a standing call
    a = abs(raw_score)                    # rank the undamped score; event days are capped below
    c1, c2, c3 = cuts or CONV_FALLBACK
    lvl = 0 if a < c1 else 1 if a < c2 else 2 if a < c3 else 3
    if held:
        lvl = 0
    if upcoming:
        lvl = min(lvl, 1)            # never better than Low into a binary print
    conviction = ["Very low", "Low", "Medium", "High"][lvl]
    directional = [s for s in signals if s.get("directional", True)]
    ups = [s for s in directional if s["score"] > 0]
    downs = [s for s in directional if s["score"] < 0]
    top = lambda L: sorted(L, key=lambda s: s["weight"], reverse=True)  # noqa: E731

    # Desk-note style: the app already shows the call, conviction and window above this.
    names = lambda L, n: ", ".join(s["label"] for s in top(L)[:n])  # noqa: E731
    lead, against = (ups, downs) if bias == "Bullish" else (downs, ups)
    parts = ["Futures are closed."] if view == "now" and not sess else []
    if held:
        parts.append(f"Holding the earlier {bias.lower()} call: the score ({score:+.2f}) hasn't moved far enough the other way to flip it.")
    parts.append(f"For: {names(lead, 3)}." if lead else "The votes are dead even, so the call defaults long.")
    if against:
        parts.append(f"Against: {names(against, 2)}.")
    if abs(score) < NEUTRAL_BAND:
        parts.append("Close to a coin flip. Let price confirm.")
    if pos:
        parts.append("Dealers long gamma: expect chop and fades." if pos["net_gex"] > 0
                     else "Dealers short gamma: moves can run.")
    if peer:
        parts.append(f"{peer[0]} leans the other way." if label_of(own) != label_of(peer[1]) else f"{peer[0]} agrees.")
    if upcoming:
        parts.append("On deck: " + ", ".join(f"{e['title']} {e['time']}" for e in upcoming) + ". Reassess after.")

    invalid = None
    if pos and pos["flip"]:
        f, above = pos["flip"], pos["px"] >= pos["flip"]
        invalid = {
            "Bullish": f"A 15-minute close below the gamma flip at {f:,.0f}" if above
            else f"A failure to reclaim {f:,.0f}, or a break of the put wall at {pos['put_wall']:,.0f}",
            "Bearish": f"A reclaim and hold of the gamma flip at {f:,.0f}" if not above
            else f"A hold above {f:,.0f}, or a push through the call wall at {pos['call_wall']:,.0f}",
            "Neutral": f"A break of {pos['put_wall']:,.0f} or {pos['call_wall']:,.0f}",
        }[bias]
    elif view == "overnight":
        invalid = "Futures reversing through today's 4 PM close in Asia"
    elif view == "now" and sess:
        invalid = f"Price reversing through the {sess[0]} session open"
    elif cfg["future"] in ("NQ=F", "ES=F"):
        invalid = "Futures reversing through yesterday's 4 PM close after the open"

    macro_w = sum(s["weight"] for s in signals if s["group"] != "Dealer positioning")
    return dict(bias=bias, score=round(score, 3), raw_score=round(raw_score, 3), dampened=damp, held=held,
                own_score=round(own, 3), conviction=conviction,
                ups=len(ups), downs=len(downs), total=len(signals), overall=" ".join(parts),
                invalidation=invalid,
                macro_share=round(100 * macro_w / (sum(s["weight"] for s in signals) or 1)))


def expected_range(cfg, pos, q, px, hours=6.5):
    v = q.get(cfg["vol_index"])
    if not (v and px):
        return None
    scale = math.sqrt(max(min(hours, 6.5), 0.25) / 6.5)   # 1-sigma for the window being called
    move = px * v["prev"] / 100 / math.sqrt(252) * scale * (0.85 if pos and pos["net_gex"] > 0 else 1.0 if not pos else 1.15)
    return {"move": round(move, 2), "low": round(px - move, 2), "high": round(px + move, 2)}


def divergence(markets):
    if not {"NQ", "ES"} <= markets.keys():
        return None
    n, e = markets["NQ"]["bias"]["own_score"], markets["ES"]["bias"]["own_score"]
    if label_of(n) == label_of(e):
        return None
    lead = "Tech is lagging the broad market" if n < e else "Tech is leading the broad market"
    return (f"On their own, NQ reads {label_of(n).lower()} ({n:+.2f}) and ES {label_of(e).lower()} ({e:+.2f}). {lead}. "
            "Each call is blended toward the other, so size down or wait for them to line up.")


# ─────────────────────────────── grading ──────────────────────────────────
def next_weekday(d):
    d += timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def grade_window(kind, d):
    """(start, end) of the move each session's call is graded on, in ET."""
    if kind == "overnight":   # all of Asia + London, into the NY open
        return datetime.combine(d, time(18, 0), ET), datetime.combine(next_weekday(d), time(9, 30), ET)
    return datetime.combine(d, time(9, 30), ET), datetime.combine(d, OPEN_MOVE_END, ET)


def grade_history(hist_dir, kind):
    if FIXTURES:
        return
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        return
    now = datetime.now(ET)
    for f in sorted(hist_dir.glob("20*.json"))[-40:]:
        d = date.fromisoformat(f.stem)
        t0, t1 = grade_window(kind, d)
        if now < t1 + timedelta(minutes=5):
            continue
        rep = json.loads(f.read_text())
        changed = False
        for sym, m in rep.get("markets", {}).items():
            if not m.get("bias") or (m.get("outcome") and m["outcome"].get("end") == t1.isoformat()):
                continue   # already graded on the current window (an old window gets re-graded)
            try:
                age = (now.date() - d).days
                iv = "1m" if age <= 6 else "5m" if age <= 55 else "60m"
                h = yf.Ticker(MARKETS[sym]["future"]).history(start=d, end=t1.date() + timedelta(days=1),
                                                               interval=iv, prepost=True)
                h.index = h.index.tz_convert(ET)
                if kind == "overnight":   # Globex reopens at 6 PM: start from the last print at or before it
                    o = float(h[h.index <= t0]["Close"].iloc[-1])
                else:
                    o = float(h[h.index >= t0]["Open"].iloc[0])
                c = float(h[h.index <= t1]["Close"].iloc[-1])
                mv, b = c - o, m["bias"]["bias"]
                hit = (mv > 0 and b == "Bullish") or (mv < 0 and b == "Bearish")
                m["outcome"] = {"open": o, "noon": c, "close": c, "move": round(mv, 2), "start": t0.isoformat(),
                                "end": t1.isoformat(),
                                "result": "n/a" if b == "Neutral" else ("hit" if hit else "miss")}
                changed = True
            except Exception:  # noqa: BLE001
                continue
        if changed:
            f.write_text(json.dumps(rep, indent=1))


def record_stats(hist_dir, sym, before: str | None = None, limit: int = 120):
    """Hit rate overall, by call, by conviction, and which signals actually pointed the right way."""
    rows, attrib = [], {}
    files = [f for f in sorted(hist_dir.glob("20*.json")) if before is None or f.stem < before]
    for f in files[-limit:]:
        m = json.loads(f.read_text()).get("markets", {}).get(sym, {})
        o = m.get("outcome")
        if not o:
            continue
        rows.append({"date": f.stem, "bias": m["bias"]["bias"], "conviction": m["bias"]["conviction"],
                     "score": m["bias"]["score"], "move": o["move"], "result": o["result"]})
        if o["move"] == 0:
            continue
        for sg in m.get("signals", []):
            if not sg["score"]:
                continue
            a = attrib.setdefault(sg["key"], {"label": sg["label"], "n": 0, "right": 0})
            a["n"] += 1
            a["right"] += (sg["score"] > 0) == (o["move"] > 0)

    def rate(rs):
        h = sum(r["result"] == "hit" for r in rs)
        ms = sum(r["result"] == "miss" for r in rs)
        return {"hits": h, "misses": ms, "pct": round(100 * h / (h + ms)) if h + ms else None}

    streak, kind = 0, None
    for r in reversed([r for r in rows if r["result"] != "n/a"]):
        if kind in (None, r["result"]):
            kind, streak = r["result"], streak + 1
        else:
            break
    sig_rows = sorted(({"key": k, **v, "pct": round(100 * v["right"] / v["n"])} for k, v in attrib.items()),
                      key=lambda x: (-x["pct"], -x["n"]))
    return {
        **rate(rows),
        "by_bias": {b: rate([r for r in rows if r["bias"] == b]) for b in ("Bullish", "Bearish")},
        "by_conviction": {c: rate([r for r in rows if r["conviction"] == c]) for c in ("Very low", "Low", "Medium", "High")},
        "streak": {"n": streak, "kind": kind},
        "signals": sig_rows,
        "recent": rows[-40:],
    }


# ──────────────────────────────── main ────────────────────────────────────
def previous_call(hist, view, target, as_of, sess):
    """The last call already published for this same window, so near-zero wobbles don't flip it."""
    try:
        if view == "now" and sess:
            # The session's call is the one made right before it opened; later reads only flip it
            # if they clear the hold band. If the session opened before any read, use the first one inside it.
            end, start = sess[2].isoformat(), sess[1].isoformat()
            same = []
            for f in sorted(hist.glob("20*.json"))[-40:]:
                d = json.loads(f.read_text())
                if (d.get("window") or {}).get("end") == end and d.get("as_of", "") < as_of.isoformat():
                    same.append(d)
            before = [d for d in same if d["as_of"] < start]
            pick = before[-1] if before else (same[0] if same else None)
            if not pick:
                return {}
            return {k: label_of(m["bias"].get("own_score", m["bias"]["score"]))
                    for k, m in pick.get("markets", {}).items()}
        f = hist / f"{target.isoformat()}.json"
        if f.exists():
            d = json.loads(f.read_text())
            if d.get("as_of", "") < as_of.isoformat():
                return {k: label_of(m["bias"].get("own_score", m["bias"]["score"])) for k, m in d.get("markets", {}).items()}
    except Exception:  # noqa: BLE001
        pass
    return {}


def build_view(target, as_of, live, now, view, hist, sess=None):
    """One complete read: view = 'preopen' (9:25 ET cut), 'now' (latest) or 'overnight' (6 PM cut)."""
    NOTES.clear(); SOURCES.clear()
    print(f"[{view}] session {target} · evidence cut at {as_of:%Y-%m-%d %H:%M} ET", file=sys.stderr)
    marks = None
    if view == "overnight":
        marks = {"close": datetime.combine(target, time(16, 0), ET)}
    quotes = fetch_quotes(MACRO_TICKERS, target, as_of, marks)
    events = fetch_calendar(target if view != "overnight" else next_weekday(target))
    if view == "overnight":   # tonight's reporters plus tomorrow morning's
        earnings = [e for e in fetch_earnings(target) if e["when"] == "after close"] + \
                   [dict(e, when="tomorrow before open") for e in fetch_earnings(next_weekday(target))
                    if e["when"] == "before open"]
    else:
        earnings = fetch_earnings(target) + [dict(e, when="last night") for e in
                                             fetch_earnings(prev_weekday(target)) if e["when"] == "after close"]
    prior_files = [f for f in sorted(hist.glob("20*.json")) if f.stem < target.isoformat()]
    prior = json.loads(prior_files[-1].read_text()) if prior_files else None

    pre = {}
    for sym, cfg in MARKETS.items():
        pos, why = None, None
        try:
            pos, why = analyse_positioning(cfg, quotes, target, as_of, live)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            why = f"positioning failed ({type(e).__name__})"
        if why:
            note(f"{sym}: dealer positioning skipped, {why}. The read is macro only.")
        if view == "overnight":
            base = overnight_signals(cfg, quotes)
        elif view == "now" and sess:
            # no intraday price-action signals: the call comes from macro, positioning and news
            base = macro_signals(cfg, quotes)
        else:
            base = macro_signals(cfg, quotes)
        extra = [x for x in (fed_pressure_signal(as_of), surprise_signal(events, as_of),
                             None if view == "overnight" else vol_crush_signal(events, quotes, as_of)) if x] + liquidity_signals(cfg, as_of)
        base = base + extra
        # learn only from sessions graded before this one, so every read stays point-in-time
        key = as_of.strftime("%Y-%m-%dT%H%M") if view == "now" else target.isoformat()
        mult = learned_multipliers(hist, sym, key, 1000 if view == "now" else 120)
        pre[sym] = (pos, apply_learning(base, mult) + (dealer_signals(cfg, pos, quotes, events, as_of) if pos else []))

    if view == "overnight":
        hours = 6.5          # ~15h of Globex carries roughly one cash session of variance
    elif view == "now" and sess:
        hours = (sess[2] - as_of).total_seconds() / 3600
    else:
        hours = 0.5
    prior_call = previous_call(hist, view, target, as_of, sess)
    stamp_key = as_of.strftime("%Y-%m-%dT%H%M") if view == "now" else target.isoformat()
    markets = {}
    for sym, cfg in MARKETS.items():
        pos, sigs = pre[sym]
        peers = [k for k in pre if k != sym and pre[k][1]]
        peer = (peers[0], own_score(pre[peers[0]][1])) if peers and sigs else None
        cuts, n_hist = conviction_cuts(sym, stamp_key)
        bias = narrate(cfg, sym, pos, sigs, events, earnings, as_of, live, peer, view, sess, prior_call.get(sym), cuts)
        bias["conv_cuts"] = [round(c, 3) for c in cuts]; bias["conv_basis"] = n_hist
        pin = pin_risk(pos, as_of)
        if pin:
            bias["pin"] = pin
            bias["overall"] = (bias.get("overall") or "") + f" {opex_kind(as_of.date()).capitalize()}: dealers are long gamma near {pin:,.0f}, so price may pin there."
        px = pos["px"] if pos else (quotes.get(cfg["future"]) or {}).get("last")
        changed = None
        if view != "now" and prior and sym in prior.get("markets", {}):
            pb = prior["markets"][sym]["bias"]
            changed = {"prev_date": prior["date"], "prev_bias": pb["bias"], "prev_score": pb["score"],
                       "delta": round(bias["score"] - pb["score"], 3)}
        markets[sym] = {
            "name": cfg["name"], "bias": bias, "signals": sigs, "changed": changed,
            "expected_range": expected_range(cfg, pos, quotes, px, hours),
            "levels": {"price": px and round(px, 2), **({} if not pos else {
                "flip": pos["flip"] and round(pos["flip"], 2),
                "flip_close": pos["flip_close"] and round(pos["flip_close"], 2),
                "call_wall": pos["call_wall"], "put_wall": pos["put_wall"], "majors": pos["majors"]})},
            "dealer": None if not pos else {
                "net_gex_bn": round(pos["net_gex"] / 1e9, 3),
                "call_put_gamma": round(pos["call_gamma"] / pos["put_gamma"], 2) if pos["put_gamma"] else None,
                "dealer_delta_bn": round(pos["dealer_delta"] / 1e9, 2),
                "charm_bn": round(pos["charm_hedge_flow"] / 1e9, 3),
                "vanna_bn_per_pt": round(pos["vanna_per_volpt"] / 1e9, 3), "chains": pos["chains"]},
            "flow": None if not (pos and pos["flow"]) else {
                "calls_m": round(pos["flow"]["call"] / 1e6, 1), "puts_m": round(pos["flow"]["put"] / 1e6, 1),
                "top": pos["top_flow"]},
            "profile": pos["profile"] if pos else [],
            "earnings": [e for e in earnings if e["symbol"] in cfg["megacaps"]],
            "scorecard": None, "outcome": None,
        }
    return {
        "view": view, "session": "overnight" if view == "overnight" else "ny", "date": target.isoformat(),
        "window": {"name": sess[0], "start": sess[1].isoformat(), "end": sess[2].isoformat()} if sess else
                  {"name": "Asia + London", "start": datetime.combine(target, time(18), ET).isoformat(),
                   "end": datetime.combine(next_weekday(target), time(9, 30), ET).isoformat()} if view == "overnight" else
                  {"name": "NY open", "start": datetime.combine(target, time(9, 30), ET).isoformat(),
                   "end": datetime.combine(target, OPEN_MOVE_END, ET).isoformat()},
        "mode": "live" if live else "point-in-time",
        "as_of": as_of.isoformat(timespec="seconds"),
        "generated": now.isoformat(timespec="seconds"),
        "events": events,
        "macro": {t: {"last": round(v["last"], 4), "chg_pct": round(pct(v["last"], v["prev"]), 2), "ts": v.get("ts")}
                  for t, v in quotes.items() if v.get("fresh", True)},
        "markets": markets, "divergence": divergence(markets),
        "notes": list(NOTES), "sources": dict(SOURCES),
    }


def finish(read, hist, kind):
    """Save a session read to history, grade it if its window has passed, attach its record."""
    path = hist / f"{read['date']}.json"
    old = json.loads(path.read_text()) if path.exists() else None
    t1 = grade_window(kind, date.fromisoformat(read["date"]))[1].isoformat()
    for sym, m in read["markets"].items():            # keep an existing grade on re-runs
        if old and (old.get("markets", {}).get(sym, {}).get("outcome") or {}).get("end") == t1:
            m["outcome"] = old["markets"][sym]["outcome"]
    path.write_text(json.dumps(read, indent=1, default=float))
    grade_history(hist, kind)
    saved = json.loads(path.read_text())
    for sym in read["markets"]:
        read["markets"][sym]["outcome"] = saved["markets"][sym].get("outcome")
        read["markets"][sym]["scorecard"] = record_stats(hist, sym)
    return read


def finish_now(read, hist):
    """Save a compact copy of each Now call, grade the ones whose session has ended, attach the record."""
    stem = datetime.fromisoformat(read["as_of"]).strftime("%Y-%m-%dT%H%M")
    keep = {"date": read["date"], "as_of": read["as_of"], "window": read["window"], "markets": {
        sym: {"bias": m["bias"], "signals": m["signals"], "outcome": None} for sym, m in read["markets"].items()}}
    path = hist / f"{stem}.json"
    if not path.exists():
        path.write_text(json.dumps(keep, indent=1, default=float))
    grade_now(hist)
    for sym in read["markets"]:
        read["markets"][sym]["scorecard"] = record_stats(hist, sym, limit=1000)
    return read


def grade_now(hist):
    """Each Now call is graded from the price when it was made to the close of its session."""
    if FIXTURES:
        return
    now = datetime.now(ET)
    for f in sorted(hist.glob("20*.json"))[-400:]:
        rep_ = json.loads(f.read_text())
        t0, t1 = datetime.fromisoformat(rep_["as_of"]), datetime.fromisoformat(rep_["window"]["end"])
        if now < t1 + timedelta(minutes=5):
            continue
        changed = False
        for sym, m in rep_["markets"].items():
            if m.get("outcome"):
                continue
            h = fetch_bars(MARKETS[sym]["future"], t0 - timedelta(minutes=30), t1)
            if h is None:
                continue
            a, b = h[h.index <= t0], h[h.index <= t1]
            if a.empty or b.empty:
                continue
            o, c = float(a["Close"].iloc[-1]), float(b["Close"].iloc[-1])
            mv, bias = c - o, m["bias"]["bias"]
            hit = (mv > 0 and bias == "Bullish") or (mv < 0 and bias == "Bearish")
            m["outcome"] = {"open": o, "noon": c, "close": c, "move": round(mv, 2), "start": t0.isoformat(),
                            "end": t1.isoformat(), "result": "n/a" if bias == "Neutral" else ("hit" if hit else "miss")}
            changed = True
        if changed:
            f.write_text(json.dumps(rep_, indent=1))


# ───────────────────────── red folder: CPI and FOMC previews ─────────────────────────
def daily_closes(ticker, days=70):
    """[(date, close)] daily, oldest → newest, from Yahoo."""
    if FIXTURES:
        return []
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period=f"{days}d", interval="1d", auto_adjust=False)
        return [(i.date(), float(v)) for i, v in h["Close"].dropna().items()]
    except Exception as e:  # noqa: BLE001
        print(f"daily {ticker}: {type(e).__name__} {e}", file=sys.stderr)
        return []


def two_year():
    rows = fred("DGS2")
    return rows if len(rows) > 25 else daily_closes("2YY=F")


def _chg_n(rows, n, pct_=True):
    if len(rows) <= n:
        return None
    a, b = rows[-n - 1][1], rows[-1][1]
    return pct(b, a) if pct_ else b - a


def _vote(key, label, v, w, text):
    return {"key": key, "label": label, "score": v, "weight": w, "text": text}


def _preview(votes, pos_word, neg_word):
    tot = sum(x["weight"] for x in votes) or 1
    sc = sum(x["score"] * x["weight"] for x in votes) / tot
    lean = pos_word if sc >= 0 else neg_word
    a = abs(sc)
    return round(sc, 3), lean, ("Very low" if a < .1 else "Low" if a < .3 else "Medium" if a < .55 else "High")


def cpi_preview(as_of):
    """Will CPI land hot or cool vs the forecast? Leading inflation inputs, per the macro notes (PPI leads CPI)."""
    V = []
    ppi = _upto(fred("PPIFIS"), as_of, 45)
    if len(ppi) > 2:
        x = _mom(ppi)
        V.append(_vote("ppi", "PPI m/m (leads CPI)", 1 if x > .2 else -1 if x < 0 else 0, 1.5,
                       f"Producer prices {x:+.2f}% last month. Costs at producers pass through to consumers."))
    core_sa, core_nsa = _upto(fred("CPILFESL"), as_of, 45), _upto(fred("CPILFENS"), as_of, 45)
    if len(core_sa) > 4 and len(core_nsa) > 13:
        ann3 = ((core_sa[-1][1] / core_sa[-4][1]) ** 4 - 1) * 100
        yoy = _yoy(core_nsa)
        V.append(_vote("coremom", "Core CPI momentum", 1 if ann3 > yoy + .3 else -1 if ann3 < yoy - .3 else 0, 1.0,
                       f"Core CPI is running {ann3:.1f}% annualized over 3 months vs {yoy:.1f}% over 12."))
        m = _mom(core_sa)
        V.append(_vote("sticky", "Last core print", 1 if m > .3 else -1 if m < .2 else 0, .75,
                       f"Core CPI rose {m:+.2f}% last month. Over 0.3% is sticky, under 0.2% is cooling."))
    oil = daily_closes("CL=F")
    c = _chg_n(oil, 21)
    if c is not None:
        V.append(_vote("oil", "Oil over the month", 1 if c > 5 else -1 if c < -5 else 0, .75,
                       f"WTI {c:+.1f}% over the past month. Energy feeds straight into headline CPI."))
    two = two_year()
    c = _chg_n(two, 21, False)
    if c is not None:
        bp = c * 100
        V.append(_vote("two", "2-year yield trend", 1 if bp > 15 else -1 if bp < -15 else 0, .75,
                       f"2-year yield {bp:+.0f} bp over the past month. Rates traders lean the way they expect the print."))
    if not V:
        return None
    sc, lean, conv = _preview(V, "Hotter", "Cooler")
    return {"score": sc, "lean": f"{lean} than forecast", "call": "Bearish" if sc >= 0 else "Bullish", "conviction": conv,
            "votes": V, "why": "A hot CPI pushes yields up and cuts the odds of Fed easing, which hits stocks, tech hardest."}


def fomc_preview(as_of):
    """Will the Fed sound more hawkish or dovish than the market has priced?"""
    V, priced, effr = [], None, None
    dff = fred("DFF")
    if dff:
        effr = dff[-1][1]
    else:
        zq = daily_closes("ZQ=F", 10)
        effr = 100 - zq[-1][1] if zq else None
    bill = daily_closes("^IRX", 10)
    if effr is not None and bill:
        gap = (bill[-1][1] - effr) * 100
        priced = "a cut" if gap < -15 else "a hike" if gap > 15 else "a hold"
    # the data side: the same Fed-pressure gauge (hot data = hawkish)
    fp = fed_pressure_signal(as_of)
    hawk = 0
    if fp:
        hawk = -fp["score"]
        V.append(_vote("data", "Official data", hawk, 1.5, fp["text"].split(". Most")[0].split(". The data")[0] + "."))
    if priced:
        mis = 1 if (priced == "a cut" and hawk > 0) else -1 if (priced != "a cut" and hawk < 0) else 0
        V.append(_vote("priced", "What's priced", mis, 1.25,
                       f"Bills vs the fed funds rate price {priced}. " +
                       ("Hot data against priced cuts is the classic hawkish surprise." if mis > 0 else
                        "Cool data against no priced cut leaves room for a dovish surprise." if mis < 0 else
                        "Data and pricing agree, so a surprise is less likely.")))
    two = two_year()
    c = _chg_n(two, 21, False)
    if c is not None:
        bp = c * 100
        V.append(_vote("two", "2-year yield trend", 1 if bp > 15 else -1 if bp < -15 else 0, .75,
                       f"2-year yield {bp:+.0f} bp over the past month: the market's own read on the Fed path."))
    be = fred("T10YIE")
    if len(be) > 22:
        d = (be[-1][1] - be[-22][1]) * 100
        V.append(_vote("breakeven", "Inflation expectations", 1 if d > 10 else -1 if d < -10 else 0, .75,
                       f"10-year breakeven inflation {d:+.0f} bp over the past month ({be[-1][1]:.2f}%)."))
    if not V:
        return None
    sc, lean, conv = _preview(V, "Hawkish", "Dovish")
    return {"score": sc, "lean": lean, "call": "Bearish" if sc >= 0 else "Bullish", "conviction": conv,
            "priced": priced, "effr": effr, "votes": V,
            "why": "A hawkish Fed (fewer cuts, higher for longer) lifts yields and the dollar, which pressures stocks."}


def red_folder(now):
    """This week's high-impact US releases, plus CPI and FOMC previews built from macro data only."""
    hi = [e for e in WEEK_EVENTS if e["impact"] == "High"]
    def when(words):
        m = [e for e in hi if any(w in e["title"] for w in words) and e["ts"] >= now.isoformat()]
        return {"date": m[0]["date"], "time": m[0]["time"], "titles": sorted({e["title"] for e in m})} if m else None
    out = {"generated": now.isoformat(timespec="seconds"), "events": sorted(hi, key=lambda e: e["ts"]), "previews": {}}
    for key, fn, words, name in (("cpi", cpi_preview, ("CPI",), "CPI"), ("fomc", fomc_preview, ("FOMC", "Federal Funds Rate"), "FOMC")):
        try:
            p = fn(now)
        except Exception:  # noqa: BLE001
            traceback.print_exc(); p = None
        if p:
            p.update(name=name, scheduled=when(words))
            out["previews"][key] = p
    return out


def resolve_overnight(now):
    forced = os.environ.get("REPORT_DATE", "").strip()
    if forced:
        return date.fromisoformat(forced)
    d = now.date()
    while d.weekday() >= 5 or now < datetime.combine(d, time(18, 0), ET):
        d -= timedelta(days=1)
    return d


def main():
    now = datetime.now(ET)
    if os.environ.get("REPORT_NOW"):
        now = datetime.fromisoformat(os.environ["REPORT_NOW"]).astimezone(ET)
    backfill = bool(os.environ.get("REPORT_DATE", "").strip())
    hist_ny = OUT / "history"
    hist_on = OUT / "history" / "overnight"
    hist_on.mkdir(parents=True, exist_ok=True)
    grade_history(hist_ny, "ny")
    grade_history(hist_on, "overnight")

    # NY morning: pre-open (9:25 ET cut, or live before then) + "now"
    target, as_of, live = resolve_session(now)
    preopen = finish(build_view(target, as_of, live, now, "preopen", hist_ny), hist_ny, "ny")
    # Now: the rest of whatever session is trading (Asia, London or NY), graded at its close
    hist_now = OUT / "history" / "now"
    hist_now.mkdir(parents=True, exist_ok=True)
    sess = current_session(now)
    if backfill or not sess:
        current = None
        if not backfill:
            grade_now(hist_now)
    else:
        current = finish_now(build_view(sess[3], now, True, now, "now", hist_now, sess), hist_now)

    # Overnight: the most recent weekday evening whose 6 PM cutoff has passed
    on_day = resolve_overnight(now)
    on_cut = datetime.combine(on_day, time(18, 0), ET)
    overnight = finish(build_view(on_day, on_cut, False, now, "overnight", hist_on), hist_on, "overnight")

    report = {"generated": now.isoformat(timespec="seconds"), "red": red_folder(now),
              "ny": {"preopen": preopen, "now": current}, "overnight": overnight,
              "week": sorted(WEEK_EVENTS, key=lambda e: e["ts"])}
    latest = OUT / "latest.json"
    if backfill and latest.exists():
        # a backfill only adds history; keep today's reads on the page, refresh the records
        try:
            old = json.loads(latest.read_text())
            if "ny" in old:
                report = old
                for sess, hist in (("ny", hist_ny), ("overnight", hist_on)):
                    r = old[sess]["preopen"] if sess == "ny" else old[sess]
                    for sym in r["markets"]:
                        r["markets"][sym]["scorecard"] = record_stats(hist, sym)
        except Exception:  # noqa: BLE001
            pass
    report["preopen"] = report["ny"]["preopen"]       # older app versions read these keys
    report["now"] = report["ny"]["now"]
    latest.write_text(json.dumps(report, indent=1, default=float))
    for name, v in (("NY PRE-OPEN", preopen), ("NY NOW", current), ("OVERNIGHT", overnight)):
        if v:
            print(name, v["date"], json.dumps({s: f"{m['bias']['bias']} {m['bias']['score']:+.2f}"
                                               for s, m in v["markets"].items()}))


if __name__ == "__main__":
    main()
