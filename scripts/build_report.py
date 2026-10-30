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
from datetime import date, datetime, time, timedelta, timezone
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
                 "^VIX", "^VIX3M", "^VXN", "^TNX", "JPY=X", "CNH=X"]
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


# Carson's windows, in Central time (ET minus one hour; both shift for daylight saving together):
#   Now (Asia)     5 PM → 2 AM CT  = Globex reopen through Asia, called before the reopen
#   Overnight      2 AM → 8 AM CT  = London open to the US pre-market, locked at 1:50 AM CT
#   Pre-open       8:30 → 9 AM CT  = the first half hour of New York
#   Now (NY)       9 AM → 3 PM CT  = the rest of New York, called before 9
ON_START = time(3, 0)      # 2 AM CT, London open
ON_END = time(9, 0)        # 8 AM CT
ON_CUT = time(2, 50)       # the overnight call locks ten minutes before London
NY_NOW_START = time(10, 0)  # 9 AM CT, when the pre-open call's window ends


def current_session(now: datetime):
    """
    Which session the Now call is for, as (name, start, end, ny_date), in ET.
    Asia 6 PM → 3 AM ET (5 PM → 2 AM CT), built from 4 PM ET so it's in place for the reopen;
    London 3 → 9:30 AM ET; NY 10 AM → 4 PM ET (built from 9:30). ny_date is the NY day the
    session feeds into. None when futures are shut.
    """
    wd, t, d = now.weekday(), now.time(), now.date()
    at = lambda day, hh, mm=0: datetime.combine(day, time(hh, mm), ET)  # noqa: E731
    if wd == 5 or (wd == 6 and t < time(16)) or (wd == 4 and t >= time(16)):
        return None   # Sunday from 4 PM ET: the Asia call builds ahead of the 5 PM CT reopen
    if time(9, 30) <= t < time(16):
        return "NY", at(d, 10), at(d, 16), d
    if t >= time(16):
        return "Asia", at(d, 18), at(d + timedelta(days=1), 3), next_weekday(d)
    if t < ON_START:
        return "Asia", at(d - timedelta(days=1), 18), at(d, 3), d
    return "London", at(d, 3), at(d, 9, 30), d

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
        elif not out and series in cache:
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


def _zchg(rows, step=5, window=250):
    """Latest step-period change as a z-score against the trailing window of step-period changes."""
    if len(rows) < step + 30:
        return None, None
    v = np.array([x for _, x in rows[-(window + step):]], dtype=float)
    ch = v[step:] - v[:-step]
    sd = ch[:-1].std()
    return (None, ch[-1]) if sd <= 0 else (float((ch[-1] - ch[:-1].mean()) / sd), float(ch[-1]))


def policy_gap_signal(as_of):
    """
    2-year Treasury minus the effective Fed funds rate: how far the market's Fed path sits from
    today's policy rate. In machine-learning yield-curve forecasts these Treasury-minus-Fed-funds
    spreads were the most important rate predictors (Kang 2026, arXiv 2608.07536). A gap that
    widens over a week means a more hawkish path is being priced: a headwind for duration-heavy tech.
    """
    y2, ff = _upto(fred("DGS2"), as_of, 1), _upto(fred("EFFR"), as_of, 1)
    if len(y2) < 7 or not ff:
        return None
    g = [(d, v - _at(ff, d)) for d, v in y2[-7:] if _at(ff, d) is not None]
    if len(g) < 6:
        return None
    lvl, d5 = g[-1][1], (g[-1][1] - g[-6][1]) * 100
    sc = -1 if d5 > 8 else 1 if d5 < -8 else 0
    c3 = _upto(fred("T10Y3M"), as_of, 1)
    curve = f" The 10Y minus 3M curve is {c3[-1][1]:+.2f}%." if c3 else ""
    return sig("policygap", "Policy gap (2Y − Fed funds)", sc, 0.75,
               f"2-year yield minus Fed funds {lvl*100:+.0f} bp, {d5:+.0f} bp over 5 sessions. " +
               ("The market is pricing a tighter Fed path." if sc < 0 else
                "The market is pricing in more cuts." if sc > 0 else "The priced Fed path is steady.") + curve, "Macro")


def stress_signal(as_of):
    """St. Louis Fed Financial Stress Index (weekly, 18 market series). Rising stress = risk-off."""
    r = _upto(fred("STLFSI4"), as_of, 6)
    if len(r) < 5:
        return None
    lvl, d = r[-1][1], r[-1][1] - r[-5][1]
    sc = -1 if d > 0.25 else 1 if d < -0.25 else 0
    return sig("stress", "Financial stress", sc, 0.5,
               f"St. Louis Fed stress index {lvl:+.2f} ({'above' if lvl > 0 else 'below'} normal), {d:+.2f} over 4 weeks. " +
               ("Stress is building." if sc < 0 else "Stress is easing." if sc > 0 else "Stress is steady."), "Macro")


def macro_z_signal(as_of):
    """
    A rule-based reading of FRED macro z-scores, read twice and averaged, after Wang, Dai, Ma & Geng
    (2026, arXiv 2606.08283): a hawkish lens (inflation expectations, real rates, the priced Fed path,
    the dollar) and a dovish lens (credit spreads, financial stress). Averaging the two lenses was what
    the paper's 'debate' setup added: it cancels one lens's bias rather than inventing new signal.
    Each input is its 5-session change scored against the past year of 5-session changes.
    """
    def z(series, lag=1, step=5):
        return _zchg(_upto(fred(series), as_of, lag), step)[0]
    gap = None
    y2, ff = _upto(fred("DGS2"), as_of, 1), _upto(fred("EFFR"), as_of, 1)
    if len(y2) > 40 and ff:
        g = [(d, v - _at(ff, d)) for d, v in y2 if _at(ff, d) is not None]
        gap = _zchg(g)[0]
    hawk_in = {"5Y inflation expectations": z("T5YIE"), "Real 10Y yield": z("DFII10"),
               "Priced Fed path": gap, "Dollar": z("DTWEXBGS", lag=7)}
    dove_in = {"High-yield spreads": z("BAMLH0A0HYM2"), "Financial stress": z("STLFSI4", lag=6, step=4)}
    hv = [v for v in hawk_in.values() if v is not None]
    dv = [v for v in dove_in.values() if v is not None]
    if not hv or not dv:
        return None
    hawk = -float(np.clip(np.mean(hv), -3, 3))     # rising inflation/rates/dollar = bad for stocks
    dove = -float(np.clip(np.mean(dv), -3, 3))     # widening spreads / rising stress = bad for stocks
    both = (hawk + dove) / 2
    sc = 1 if both > 0.5 else -1 if both < -0.5 else 0
    word = lambda x: "bullish" if x > 0.5 else "bearish" if x < -0.5 else "neutral"
    top = max({**hawk_in, **dove_in}.items(), key=lambda kv: abs(kv[1]) if kv[1] is not None else -1)
    return sig("macroz", "Macro z-score (hawk + dove)", sc, 0.5,
               f"Hawkish lens {hawk:+.1f}σ ({word(hawk)}), dovish lens {dove:+.1f}σ ({word(dove)}), "
               f"together {both:+.1f}σ. Biggest mover: {top[0]} ({top[1]:+.1f}σ over 5 sessions).", "Macro")


def fomc_drift_signal(events, as_of):
    """
    Fed announcement days. Long-term Treasury yields tend to fall the day before an FOMC announcement,
    and that bond move predicts the equity drift into the statement (Pan & Peng, SSRN 4764451). But the
    pre-FOMC drift and announcement premium only show up when key macro data came out right before the
    meeting; on the other two thirds of FOMC days there is neither (Alam, SSRN 4065084).
    """
    fomc = [e for e in (events or []) if e.get("impact") == "High" and
            any(w in e.get("title", "") for w in ("FOMC", "Federal Funds Rate"))]
    if not fomc or fomc[0]["ts"][:10] != as_of.date().isoformat() or fomc[0]["ts"] <= as_of.isoformat():
        return None
    day = as_of.date()
    lo = prev_weekday(prev_weekday(day))
    pool = {(e["ts"], e["title"]): e for e in WEEK_EVENTS + list(events)}.values()
    macro = [e for e in pool if e.get("impact") == "High" and e is not fomc[0]
             and not any(w in e.get("title", "") for w in ("FOMC", "Federal Funds Rate", "Press Conference", "Statement", "Projections"))
             and lo.isoformat() <= e["ts"][:10] and e["ts"] < fomc[0]["ts"]]
    y = _upto(fred("DGS10"), as_of, 1)
    if len(y) < 2:
        return None
    dy = (y[-1][1] - y[-2][1]) * 100
    if not macro:
        return sig("fomcdrift", "Fed-day drift", 0, 0.75,
                   f"Fed decision at {fomc[0]['time']}. 10-year yield {dy:+.0f} bp yesterday. No key data came out just before "
                   "this meeting, and on days like that there's historically no pre-Fed drift.", "Macro")
    sc = 1 if dy <= -2 else -1 if dy >= 3 else 0
    return sig("fomcdrift", "Fed-day drift", sc, 0.75,
               f"Fed decision at {fomc[0]['time']}, right after {macro[-1]['title']}. 10-year yield {dy:+.0f} bp yesterday. " +
               ("Long bonds rallied into the meeting, which has led a stock drift into the statement." if sc > 0 else
                "Long yields rose into the meeting: no bond-led drift." if sc < 0 else "Bonds were flat into the meeting."), "Macro")


def vrp_signal(cfg, q, target):
    """
    Variance risk premium: implied vol (the vol index at yesterday's close) minus the index's realized vol
    over the past month. A fat premium means investors are paying up for protection, which has preceded
    better equity returns; realized vol above implied marks real stress (Bardgett, Gourier & Leippold,
    SSRN 2787161, on the premium's predictive power). Slow-moving, so a background vote.
    """
    v = q.get(cfg["vol_index"])
    rows = [r for r in daily_closes(cfg["future"], 45) if r[0] < target]
    if not v or len(rows) < 22:
        return None
    px = np.array([x for _, x in rows[-22:]])
    rv = float(np.std(np.diff(np.log(px)), ddof=1) * math.sqrt(252) * 100)
    iv = float(v["prev"])
    gap = iv - rv
    sc = 1 if gap > 4 else -1 if gap < -2 else 0
    return sig("vrp", "Volatility risk premium", sc, 0.4,
               f"Implied vol {iv:.1f} vs realized {rv:.1f} over the past month: premium {gap:+.1f} pts. " +
               ("Protection is expensive relative to actual moves, a backdrop where dips get bought." if sc > 0 else
                "Realized vol is running above implied: genuine stress." if sc < 0 else "A normal premium."), "Volatility")


# ─────────────── Fed funds futures, pre-Fed positioning, VIX futures roll, Treasury auctions ───────────────
FOMC_DATES = [date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18), date(2025, 7, 30),
              date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 10),
              date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17), date(2026, 7, 29),
              date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9)]   # decision days; the calendar adds later ones
ZQ_CODES = "FGHJKMNQUVXZ"


def next_fomc(as_of):
    """The next Fed decision day on or after as_of's date (before 2 PM on the day itself)."""
    days = set(FOMC_DATES)
    for e in WEEK_EVENTS:
        if e.get("impact") == "High" and any(w in e.get("title", "") for w in ("FOMC", "Federal Funds Rate")):
            days.add(date.fromisoformat(e["ts"][:10]))
    cut = as_of.date() if as_of.time() < time(14) else as_of.date() + timedelta(days=1)
    later = sorted(d for d in days if d >= cut)
    return later[0] if later else None


def zq_implied(y, m, before):
    """[(date, implied average Fed funds rate %)] for the fed funds futures contract of month m, closes before `before`."""
    sym = f"ZQ{ZQ_CODES[m - 1]}{y % 100:02d}.CBT"
    return [(d, 100 - v) for d, v in daily_closes(sym, 20) if d < before and 80 < v <= 100]


def fed_odds(as_of, meet):
    """Market-implied change at `meet`, in bp, from the fed funds futures (the CME FedWatch method)."""
    import calendar
    if not meet:
        return None
    # The rate going into the meeting: CME's method reads it off the previous month's contract when that
    # month has no meeting; otherwise fall back to the latest effective Fed funds rate.
    py, pm = (meet.year - 1, 12) if meet.month == 1 else (meet.year, meet.month - 1)
    pre = zq_implied(py, pm, as_of.date()) if not any(d.year == py and d.month == pm for d in FOMC_DATES) else []
    if pre:
        r0 = pre[-1][1]
    else:
        effr = _upto(fred("EFFR"), as_of, 1)
        if not effr:
            return None
        r0 = effr[-1][1]
    n = calendar.monthrange(meet.year, meet.month)[1]
    after = n - meet.day
    if after >= 10:   # enough of the month left after the decision: read the meeting month itself
        rows = zq_implied(meet.year, meet.month, as_of.date())
        if not rows:
            return None
        delta = (rows[-1][1] * n - r0 * meet.day) / after - r0   # month average = r0 before, r0 + delta after
    else:             # late-month meeting: the next month's contract is all post-decision
        y, m = (meet.year + 1, 1) if meet.month == 12 else (meet.year, meet.month + 1)
        rows = zq_implied(y, m, as_of.date())
        if not rows:
            return None
        delta = rows[-1][1] - r0
    steps = delta / 0.25
    frac = abs(steps) - math.floor(abs(steps))
    return {"bp": delta * 100, "certainty": max(frac, 1 - frac), "rows": rows}


def fedwatch_signal(as_of):
    """
    The CME FedWatch read, rebuilt from free fed funds futures prices. FedWatch called FOMC decisions
    right 88% of the time 30 days out and its signal carried economic value (Bonini, Huang & Simaan,
    SSRN 5093703). The vote is the 5-session shift in the priced outcome of the next meeting:
    more easing priced is a tailwind, more tightening a headwind.
    """
    meet = next_fomc(as_of)
    od = fed_odds(as_of, meet) if meet else None
    if not od or len(od["rows"]) < 6:
        return None
    rows = od["rows"]
    d5 = (rows[-1][1] - rows[-6][1]) * 100
    sc = 1 if d5 <= -3 else -1 if d5 >= 3 else 0
    steps, word = od["bp"] / 25, "cut" if od["bp"] < 0 else "hike"
    a = abs(steps)
    odds = (f"{a*100:.0f}% odds of a 25 bp {word}" if a < 1 else
            f"a 25 bp {word} fully priced, {(a - 1)*100:.0f}% odds of 50 bp" if a < 2 else f"about {a*25:.0f} bp of {word}s priced")
    return sig("fedwatch", "FedWatch (fed funds futures)", sc, 0.4,
               f"{meet:%b %-d} Fed meeting: {odds}. "
               f"The priced rate moved {d5:+.0f} bp over 5 sessions: " +
               ("more easing is being priced." if sc > 0 else "less easing is being priced." if sc < 0 else "steady."), "Macro")


def fed_leak_signal(as_of):
    """
    Before scheduled FOMC announcements, institutions trade rate-sensitive stocks in the direction of the
    coming policy surprise, most strongly before easing surprises (Mano, SSRN 3830271). Proxy with public
    prices: small caps (IWM) and homebuilders (XHB) against the S&P 500 (SPY) over the 3 sessions
    before the meeting.
    """
    meet = next_fomc(as_of)
    if not meet or np.busday_count(as_of.date(), meet) > 3:
        return None
    def r3(t):
        rows = [r for r in daily_closes(t, 15) if r[0] < as_of.date()]
        return pct(rows[-1][1], rows[-4][1]) if len(rows) >= 4 else None
    spy = r3("SPY")
    if spy is None:
        return None
    rel = {k: x - spy for k, x in (("Small caps", r3("IWM")), ("Homebuilders", r3("XHB"))) if x is not None}
    if not rel:
        return None
    avg = sum(rel.values()) / len(rel)
    sc = 1 if avg > 1.0 else -1 if avg < -1.0 else 0
    return sig("fedleak", "Pre-Fed positioning", sc, 0.4,
               f"Fed decision {meet:%a %b %-d}. Over the last 3 sessions, " +
               ", ".join(f"{k} {v:+.1f}%" for k, v in rel.items()) + " against the S&P 500. " +
               ("Rate-sensitive stocks are being bought ahead of the Fed, which has preceded dovish surprises." if sc > 0 else
                "Rate-sensitive stocks are being sold ahead of the Fed, which has preceded hawkish surprises." if sc < 0 else
                "No clear positioning ahead of the Fed."), "Macro")


def vx_expiry(y, m):
    """VIX futures for month m settle on the Wednesday 30 days before the third Friday of the next month."""
    ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
    d = date(ny, nm, 15)
    while d.weekday() != 4:
        d += timedelta(days=1)
    return d - timedelta(days=30)


VX_CACHE = {}


def vix_roll_signal(as_of):
    """
    The VIX futures roll: (front VIX future − spot VIX) ÷ business days to settlement. In contango
    VIX futures tend to roll down to a lower VIX, and in backwardation the VIX doesn't fall
    (Simon, SSRN 2624713; Simon & Campasano 2014, threshold ±0.10 a day). Contango is the calm
    backdrop vol sellers lean on; backwardation marks stress. Settlements from CBOE, yesterday's close.
    """
    if FIXTURES:
        return None
    day = as_of.date()
    y, m = day.year, day.month
    exp = None
    for _ in range(3):
        e = vx_expiry(y, m)
        if e > day and np.busday_count(day, e) >= 5:
            exp = e
            break
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if not exp:
        return None
    try:
        if exp not in VX_CACHE:
            r = requests.get(f"https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/VX_{exp.isoformat()}.csv",
                             headers=UA, timeout=30)
            r.raise_for_status()
            VX_CACHE[exp] = r.text
        lines = [l.split(",") for l in VX_CACHE[exp].strip().splitlines()]
        hdr = [h.strip().lower() for h in lines[0]]
        ti, si = hdr.index("trade date"), hdr.index("settle")
        rows = [(date.fromisoformat(l[ti].strip()), float(l[si])) for l in lines[1:] if l[si].strip() not in ("", "0")]
        rows = [r for r in rows if r[0] < day]
        vix = dict(daily_closes("^VIX", 15))
        both = [(d, f) for d, f in rows if d in vix]
        if not both:
            raise ValueError("no matching VIX close")
        d0, fut = both[-1]
    except Exception as e:  # noqa: BLE001
        note(f"VIX futures roll skipped ({type(e).__name__})")
        return None
    bd = max(int(np.busday_count(d0, exp)), 1)
    roll = (fut - vix[d0]) / bd
    sc = 1 if roll > 0.10 else -1 if roll < -0.10 else 0
    return sig("vxroll", "VIX futures roll", sc, 0.3,
               f"Front VIX future {fut:.2f} vs VIX {vix[d0]:.2f}, {bd} sessions to settlement: roll {roll:+.2f} a day. " +
               ("Steep contango: vol is priced to drift lower, the calm backdrop." if sc > 0 else
                "Backwardation: near-term fear is priced above spot, a stress backdrop." if sc < 0 else
                "A mild roll, no signal."), "Volatility")


AUCTIONS = None


def fetch_auctions():
    """Treasury note and bond auctions (announced and completed) from the Treasury's fiscal data API."""
    global AUCTIONS
    if AUCTIONS is not None:
        return AUCTIONS
    AUCTIONS = []
    if FIXTURES:
        return AUCTIONS
    start = (datetime.now(ET).date() - timedelta(days=400)).isoformat()
    try:
        r = requests.get("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query",
                         params={"filter": f"auction_date:gte:{start}", "sort": "-auction_date", "page[size]": "1000"},
                         headers=UA, timeout=45)
        r.raise_for_status()
        for a in r.json().get("data", []):
            if a.get("security_type") not in ("Note", "Bond"):
                continue
            yrs = int(str(a.get("security_term", "0")).split("-")[0] or 0)
            if yrs < 9:
                continue
            tenor = 10 if yrs < 15 else 20 if yrs < 25 else 30
            btc = a.get("bid_to_cover_ratio")
            AUCTIONS.append({"date": date.fromisoformat(a["auction_date"][:10]), "tenor": tenor,
                             "btc": float(btc) if btc not in (None, "", "null") else None})
        stamp("Treasury auctions", True, None, f"{len(AUCTIONS)} long-end auctions")
    except Exception as e:  # noqa: BLE001
        stamp("Treasury auctions", False)
        note(f"Treasury auctions unavailable ({type(e).__name__})")
    return AUCTIONS


def auction_signal(as_of):
    """
    Long-end Treasury auctions (10, 20, 30-year). Auctions clear below secondary-market prices because
    dealers want paying for the price risk they take on (Goldreich, SSRN 449181; Herb, SSRN 4247371), so
    yields tend to back up into the 1 PM auction: a headwind that morning. Afterwards, demand matters:
    bid-to-cover above its recent average means the supply went down easily (rates relief); below it means
    the market struggled to absorb it (Garbade & Ingber, SSRN 678301 on the process).
    """
    rows = fetch_auctions()
    if not rows:
        return None
    day, t = as_of.date(), as_of.time()
    today = [a for a in rows if a["date"] == day]
    if today and t < time(13, 5):
        a = today[0]
        return sig("auction", "Treasury auction", -1, 0.3,
                   f"{a['tenor']}-year auction at 1 PM ET today. Yields tend to rise into long-end supply as dealers "
                   "demand a concession, a mild headwind until it clears.", "Macro")
    done = [a for a in rows if a["btc"] and (a["date"] < day or (a["date"] == day and t >= time(13, 5)))
            and a["date"] >= prev_weekday(day)]
    if not done:
        return None
    a = max(done, key=lambda x: x["date"])
    past = [x["btc"] for x in rows if x["tenor"] == a["tenor"] and x["btc"] and x["date"] < a["date"]][:6]
    if len(past) < 3:
        return None
    avg = sum(past) / len(past)
    diff = a["btc"] - avg
    sc = 1 if diff > 0.08 else -1 if diff < -0.08 else 0
    return sig("auction", "Treasury auction", sc, 0.3,
               f"{a['tenor']}-year auction {'today' if a['date'] == day else a['date'].strftime('%a')}: bid-to-cover "
               f"{a['btc']:.2f} vs {avg:.2f} average. " +
               ("Strong demand: the supply went down easily, relief for rates." if sc > 0 else
                "Weak demand: the market struggled to absorb the supply, pressure on rates." if sc < 0 else
                "Ordinary demand."), "Macro")


def funding_signal(as_of):
    """
    Short-term funding stress: 3-month financial commercial paper minus the 3-month T-bill (FRED DCPF3M,
    DTB3), its level and how jumpy it has been. Funding-spread factors and their volatility predicted the
    VIX, the variance risk premium and next month's S&P 500 excess returns beyond the VIX itself
    (Andreou & Ghysels, SSRN 2510196). Rising funding stress is a headwind. Slow, so a background vote.
    """
    cp, tb = _upto(fred("DCPF3M"), as_of, 1), _upto(fred("DTB3"), as_of, 1)
    if len(cp) < 120 or not tb:
        return None
    sp = [(d, v - _at(tb, d)) for d, v in cp if _at(tb, d) is not None]
    if len(sp) < 120:
        return None
    x = np.array([v for _, v in sp[-260:]])
    lvl_z = (x[-1] - x.mean()) / (x.std() or 1)
    dx = np.diff(x)
    vols = np.array([dx[i - 21:i].std() for i in range(21, len(dx) + 1)])
    vol_z = (vols[-1] - vols.mean()) / (vols.std() or 1)
    z = (lvl_z + vol_z) / 2
    sc = -1 if z > 1.0 else 1 if z < -0.5 else 0
    return sig("funding", "Funding stress", sc, 0.3,
               f"Commercial paper minus T-bills {x[-1]*100:+.0f} bp ({lvl_z:+.1f}σ vs the past year), its day-to-day swings "
               f"{vol_z:+.1f}σ. " + ("Funding markets are getting tight and jumpy, which has led higher vol and weaker stocks." if sc < 0 else
                                     "Funding is calm and cheap, a supportive backdrop." if sc > 0 else "Funding is normal."), "Macro")


def month_end_signal(as_of):
    """
    Month-end rebalancing. Pension funds and balanced funds rebalance back to fixed stock/bond weights
    around month-end: after a month where stocks beat bonds they sell stocks, and after one where bonds
    won they buy. The flows are predictable from month-to-date returns and move prices in the last days
    of the month (Harvey, Mazzoleni & Melone, "The Unintended Consequences of Rebalancing", NBER 2025).
    Active over the last 3 business days of the month.
    """
    day = as_of.date()
    nxt = date(day.year + (day.month == 12), day.month % 12 + 1, 1)
    left = int(np.busday_count(day, nxt))
    if left > 3:
        return None
    def mtd(t):
        rows = [r for r in daily_closes(t, 45) if r[0] < day]
        base = [r for r in rows if r[0] < date(day.year, day.month, 1)]
        return pct(rows[-1][1], base[-1][1]) if rows and base else None
    eq, bd = mtd("SPY"), mtd("AGG")
    if eq is None or bd is None:
        return None
    gap = eq - bd
    sc = -1 if gap > 3 else 1 if gap < -3 else 0
    return sig("monthend", "Month-end rebalancing", sc, 0.5,
               f"{left} business day{'s' if left != 1 else ''} left in the month. Month to date: stocks {eq:+.1f}%, bonds {bd:+.1f}% "
               f"({gap:+.1f} pts). " + ("Stocks beat bonds by a lot, so pension funds have stocks to sell." if sc < 0 else
                                       "Bonds beat stocks, so pension funds have stocks to buy." if sc > 0 else
                                       "Too close for big rebalancing flows."), "Macro")


def move_signal(as_of):
    """
    The MOVE index, implied volatility in Treasuries. Bond-market fear tends to spill into equity
    volatility, and the bond variance risk premium carries information about future returns
    (Choi, Mueller & Vedolin, "Bond Variance Risk Premiums", Review of Finance 2017).
    The vote is MOVE's 5-session change, with its level against the past month for context.
    """
    rows = [r for r in daily_closes("^MOVE", 45) if r[0] < as_of.date()]
    if len(rows) < 22:
        return None
    last, ago = rows[-1][1], rows[-6][1]
    c5 = pct(last, ago)
    avg = sum(v for _, v in rows[-21:]) / 21
    sc = -1 if c5 > 10 else 1 if c5 < -10 else 0
    return sig("move", "Bond volatility (MOVE)", sc, 0.5,
               f"MOVE {last:.0f}, {c5:+.0f}% over 5 sessions ({'above' if last > avg else 'below'} its 1-month average of {avg:.0f}). " +
               ("Bond volatility is jumping, which tends to spill into stocks." if sc < 0 else
                "Bond volatility is draining out, a calmer backdrop for stocks." if sc > 0 else "Bond vol is steady."),
               "Volatility")


def fomc_cycle_signal(day):
    """
    The FOMC cycle. Since 1994 the equity premium has been earned almost entirely in even weeks of the
    cycle counted from each Fed meeting (week 0 = the day before to 3 days after, then weeks 2, 4, 6),
    when the Fed's informal communication clusters; odd weeks have been flat
    (Cieslak, Morse & Vissing-Jorgensen, "Stock Returns over the FOMC Cycle", JF 2019).
    Even weeks lean bullish; odd weeks don't vote.
    """
    days = sorted(set(FOMC_DATES) | {date.fromisoformat(e["ts"][:10]) for e in WEEK_EVENTS
                                     if e.get("impact") == "High" and any(w in e.get("title", "") for w in ("FOMC", "Federal Funds Rate"))})
    past = [d for d in days if d <= day]
    fut = [d for d in days if d > day]
    if not past:
        return None
    if fut and int(np.busday_count(day, fut[0])) == 1:
        k, ref = -1, fut[0]
    else:
        k, ref = int(np.busday_count(past[-1], day)), past[-1]
    week = (k + 1) // 5
    if week > 7:
        return None
    even = week % 2 == 0
    return sig("fomccycle", "FOMC cycle week", 1 if even else 0, 0.4,
               f"Week {week} of the cycle from the {ref:%b %-d} Fed meeting. " +
               ("Even weeks are when stocks have earned their returns since 1994." if even else
                "Odd weeks have historically been flat, so no vote."), "Macro")


NYSE_HOLIDAYS = {date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
                 date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
                 date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
                 date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25), date(2027, 12, 24)}


def pre_holiday_signal(day):
    """The trading day before a US market holiday has averaged several times a normal day's return
    (Ariel, "High Stock Returns before Holidays", JFE 1990)."""
    nxt = next_weekday(day)
    if nxt not in NYSE_HOLIDAYS:
        return None
    return sig("preholiday", "Pre-holiday", 1, 0.4,
               f"Markets are shut {nxt:%A %b %-d}. The session before a holiday has historically been one of the strongest of the year.",
               "Macro")


def refunding_day(day):
    """Treasury's quarterly refunding statement: a Wednesday in the first week of Feb, May, Aug and Nov at 8:30 ET."""
    if day.month not in (2, 5, 8, 11) and not (day.month in (1, 4, 7, 10) and day.day >= 28):
        return False
    if day.weekday() != 2:
        return False
    for e in WEEK_EVENTS:
        if "refunding" in e.get("title", "").lower():
            return e["ts"][:10] == day.isoformat()
    return day.month in (2, 5, 8, 11) and day.day <= 7


def refunding_signal(day):
    """
    Quarterly refunding: how much Treasury will borrow at which maturities. Bigger long-end supply than
    expected lifts the term premium and yields (the November 2023 statement set off a big rally by
    doing the opposite). There's no free consensus to score it against, so it doesn't vote: it marks
    the day and caps conviction like any other high-impact release.
    """
    if not refunding_day(day):
        return None
    return sig("refunding", "Treasury refunding", 0, 0.4,
               "Treasury's quarterly refunding statement is out at 8:30 ET today. Long-end supply surprises move yields, "
               "so treat the open with care.", "Macro")


def copper_gold_signal(as_of):
    """
    Copper over gold: copper prices growth, gold prices fear. A rising ratio has tracked rising growth
    expectations and yields (a widely used practitioner gauge, e.g. Gundlach). Its 10-session change votes.
    """
    cu = dict(r for r in daily_closes("HG=F", 40) if r[0] < as_of.date())
    au = dict(r for r in daily_closes("GC=F", 40) if r[0] < as_of.date())
    both = sorted(set(cu) & set(au))
    if len(both) < 12:
        return None
    r1, r0 = cu[both[-1]] / au[both[-1]], cu[both[-11]] / au[both[-11]]
    c = pct(r1, r0)
    sc = 1 if c > 3 else -1 if c < -3 else 0
    return sig("copgold", "Copper vs gold", sc, 0.4,
               f"Copper/gold ratio {c:+.1f}% over 10 sessions. " +
               ("Copper is beating gold: markets are pricing more growth." if sc > 0 else
                "Gold is beating copper: markets are pricing fear over growth." if sc < 0 else "No clear growth signal."),
               "Macro")


def market_regime(q, as_of):
    """
    Calm or stressed, from the St. Louis Fed stress index and the VIX curve. Signals don't work
    equally well in every environment (Koa, Liu & Li 2026, arXiv 2610.26456), so the Record learns
    weights separately for each regime once it has enough graded calls in it.
    """
    stl = _upto(fred("STLFSI4"), as_of, 6)
    v, v3 = q.get("^VIX"), q.get("^VIX3M")
    ts = v["prev"] / v3["prev"] if v and v3 and v3.get("prev") else None
    stressed = (stl and stl[-1][1] > 0) or (ts is not None and ts >= 1.0)
    return "stressed" if stressed else "calm"


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
    gex_ex0, gross_ex0 = 0.0, 0.0      # net / gross gamma without same-day expiries
    gex_0, gross_0 = 0.0, 0.0          # the same-day part on its own
    zdte = dict(call=0.0, put=0.0)     # gamma-weighted same-day volume, live only
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
        later = c["T"] * 365 > 1.0
        gex_ex0 += float((c["CP"] * g)[later].sum()); gross_ex0 += float(g[later].sum())
        same = c["T"] * 365 < 0.5
        gex_0 += float((c["CP"] * g)[same].sum()); gross_0 += float(g[same].sum())
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
            gv = bs_gamma(S, c["K"], c["T"], c["IV"]) * c["VOL"] * 100 * S**2 * 0.01
            zdte["call"] += float(gv[same & (c["CP"] > 0)].sum())
            zdte["put"] += float(gv[same & (c["CP"] < 0)].sum())
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
        gamma_ratio_ex0=(gex_ex0 / gross_ex0) if gross_ex0 else None,
        gamma_ratio_0=(gex_0 / gross_0) if gross_0 else None,
        zdte=zdte if live and zdte["call"] + zdte["put"] > 0 else None,
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
    # Yen: a sharp yen rally means carry trades funded in yen are being unwound, forcing risk
    # selling worldwide, hardest in Asia hours (Brunnermeier, Nagel & Pedersen, "Carry Trades and
    # Currency Crashes", NBER Macro Annual 2008; the August 2024 unwind).
    c = chg(q, "JPY=X")
    if c is not None:
        sc = -1 if c < -0.6 else 1 if c > 0.5 else 0
        S.append(sig("yen", "Yen (USD/JPY)", sc, 0.5,
                     f"USD/JPY {q['JPY=X']['last']:.2f} ({c:+.2f}%). " +
                     ("The yen is ripping higher: carry trades are being unwound, which forces risk selling." if sc < 0 else
                      "The yen is weakening: carry trades are on, a risk-on backdrop." if sc > 0 else "The yen is steady."),
                     "Macro"))
    # Yuan: a sharp drop in China's currency has preceded Asia-led global selloffs (the August 2015 and
    # January 2016 devaluations); a firmer yuan is a mild support.
    c = chg(q, "CNH=X")
    if c is not None:
        sc = -1 if c > 0.4 else 1 if c < -0.3 else 0
        S.append(sig("yuan", "Chinese yuan (USD/CNH)", sc, 0.4,
                     f"USD/CNH {q['CNH=X']['last']:.4f} ({c:+.2f}%). " +
                     ("The yuan is sliding: China stress tends to hit Asia first, then everyone." if sc < 0 else
                      "The yuan is firming, a mild support for Asia and risk." if sc > 0 else "The yuan is steady."),
                     "Macro"))
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


def overnight_signals(cfg, q, target=None):
    """London-open read for 2 AM → 8 AM CT, built from data up to 1:50 AM CT: macro since the evening's
    4 PM close, including Asia's session."""
    S = [s for s in macro_signals(cfg, q) if s["key"] != "volpre"]   # VXX doesn't trade before 4 AM ET
    for s_ in S:
        if s_["key"] == "yen":   # carry unwinds hit hardest in Asia hours
            s_["weight"] = 1.0
        if s_["key"] == "yuan":
            s_["weight"] = 0.75
        if s_["key"] == "global":
            s_["text"] = s_["text"].replace("going into the open", "going into London")
    rows = [r for r in daily_closes("^VIX", 10) if target is None or r[0] <= target]
    if len(rows) >= 2:
        c = pct(rows[-1][1], rows[-2][1])
        sc = 1 if c < -4 else -1 if c > 4 else 0
        S.append(sig("vixday", "VIX today", sc, 1.0,
                     f"VIX closed {rows[-1][1]:.2f} ({c:+.1f}% on the day). " +
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


def intraday_dealer_signals(cfg, pos, as_of, sess):
    """New York session only."""
    S = []
    if not pos or not sess or sess[0] != "NY":
        return S
    z = pos.get("zdte")
    if z and time(9, 40) <= as_of.time() < time(11, 0):
        # Gamma-weighted put/call of same-day options after the open: put-heavy opening flow is followed
        # by a weaker next hour and a higher chance of a fall in it (Maurer & Müller, SSRN 7339718).
        # Their out-of-sample rule made no money, so it's a risk tilt at low weight.
        r = z["put"] / z["call"] if z["call"] else 9.9
        sc = -1 if r > 1.2 else 1 if r < 0.83 else 0
        S.append(sig("zdte", "0DTE opening flow", sc, 0.3,
                     f"Same-day option volume since the open, weighted by gamma: put/call {r:.2f}. " +
                     ("Put-heavy: a weaker next hour and a higher chance of a dip." if sc < 0 else
                      "Call-heavy: a firmer next hour." if sc > 0 else "Balanced."), "Dealer positioning"))
    if as_of.time() >= time(15, 0):
        # End-of-day hedging: when dealers are short gamma, their rebalancing pushes the day's move further
        # into the close; when long gamma, it leans against it (Barbon, Beckmeyer, Buraschi & Moerke, SSRN 3925725).
        # The one vote that uses today's own price move.
        h = fetch_bars(cfg["future"], datetime.combine(as_of.date(), time(9, 30), ET), as_of)
        if h is not None:
            r = pct(float(h["Close"].iloc[-1]), float(h["Open"].iloc[0]))
            short = pos["net_gex"] < 0
            sc = 0 if abs(r) < 0.3 else int(np.sign(r)) * (1 if short else -1)
            S.append(sig("eodgamma", "End-of-day gamma hedging", sc, 0.5,
                         f"Futures {r:+.2f}% since the open, dealers {'short' if short else 'long'} gamma. " +
                         ("Too small a move to force much hedging." if sc == 0 else
                          "Short-gamma hedging tends to push the day's move further into the close." if short else
                          "Long-gamma hedging tends to lean against the day's move into the close."), "Dealer positioning"))
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


def learned_multipliers(hist_dir: Path, sym: str, before: str, limit: int = 120, regime: str | None = None) -> dict:
    """
    Scale each signal's weight by how often it has actually pointed the right way.
    Shrunk toward 50% (10 phantom right + 10 phantom wrong) so a lucky streak can't
    swing it, and only applied once a signal has LEARN_MIN_N real graded samples.
    With a regime, a signal uses its record from that regime alone once it has enough
    samples there, and its all-weather record until then.
    """
    if not hist_dir.exists():
        return {}
    def fit(rows):
        out = {}
        for row in rows:
            if row["n"] < LEARN_MIN_N:
                continue
            rate = (row["right"] + 10) / (row["n"] + 20)
            out[row["key"]] = round(min(1.5, max(0.25, 1 + 2 * (rate - 0.5))), 2)
        return out
    out = fit(record_stats(hist_dir, sym, before, limit)["signals"])
    if regime:
        out.update(fit(record_stats(hist_dir, sym, before, limit, regime)["signals"]))
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
    upcoming = [e for e in events if e["impact"] == "High" and e["ts"] > as_of.isoformat()
                and (end is None or e["ts"] <= end)]
    ref = next((s for s in signals if s["key"] == "refunding"), None)
    if ref and not any("refunding" in e.get("title", "").lower() for e in upcoming) and             (view == "overnight" or as_of.time() < time(8, 30)):
        upcoming.append({"title": "Treasury refunding", "time": "8:30 AM", "ts": "", "impact": "High"})
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
        invalid = "Futures reversing through the 2 AM CT price once London opens"
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


def expected_range(cfg, pos, q, px, hours=6.5, intraday=False):
    v = q.get(cfg["vol_index"])
    if not (v and px):
        return None
    scale = math.sqrt(max(min(hours, 6.5), 0.25) / 6.5)   # 1-sigma for the window being called
    # Dealer gamma damps or amplifies the move. Ardia & Vaudescal (2026, SSRN 7202999): with public open
    # interest, net gamma that excludes same-day expiries predicts lower realized variance (about a third
    # lower per standard deviation, roughly 20% less vol); the 0DTE part does not survive with public data.
    ratio = pos.get("gamma_ratio_ex0") if pos else None
    if ratio is None:
        gm = 1.0 if not pos else (0.85 if pos["net_gex"] > 0 else 1.15)
    else:
        gm = 1.0 - 0.2 * max(-1.0, min(1.0, ratio / 0.5))
    r0 = pos.get("gamma_ratio_0") if pos else None
    if intraday and r0 is not None:
        # During New York, the same-day gamma state adds a little: more of it has gone with calmer next
        # minutes, but the effect flipped sign between 2024 and 2025 (Singh, SSRN 7350465), so it's capped at ±5%.
        gm *= 1.0 - 0.05 * max(-1.0, min(1.0, r0 / 0.5))
    move = px * v["prev"] / 100 / math.sqrt(252) * scale * gm
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
    if kind == "overnight":   # Asia open to the US pre-market: 8 PM → 8 AM CT
        return datetime.combine(next_weekday(d), ON_START, ET), datetime.combine(next_weekday(d), ON_END, ET)
    return datetime.combine(d, time(9, 30), ET), datetime.combine(d, OPEN_MOVE_END, ET)


def grade_history(hist_dir, kind):
    if FIXTURES:
        return
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        return
    now = datetime.now(ET)
    for f in sorted(hist_dir.glob("20*.json"))[-90:]:
        d = date.fromisoformat(f.stem)
        t0, t1 = grade_window(kind, d)
        if now < t1 + timedelta(minutes=5):
            continue
        rep = json.loads(f.read_text())
        changed = False
        for sym, m in rep.get("markets", {}).items():
            if not m.get("bias") or (m.get("outcome") and m["outcome"].get("end") == t1.isoformat()
                                     and m["outcome"].get("start") == t0.isoformat()):
                continue   # already graded on the current window (an old window gets re-graded)
            try:
                age = (now.date() - d).days
                iv = "1m" if age <= 6 else "5m" if age <= 55 else "60m"
                h = yf.Ticker(MARKETS[sym]["future"]).history(start=d, end=t1.date() + timedelta(days=1),
                                                               interval=iv, prepost=True)
                h.index = h.index.tz_convert(ET)
                if kind == "overnight":   # start from the last print at or before 8 PM CT
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


def record_stats(hist_dir, sym, before: str | None = None, limit: int = 120, regime: str | None = None):
    """Hit rate overall, by call, by conviction, and which signals actually pointed the right way."""
    rows, attrib = [], {}
    files = [f for f in sorted(hist_dir.glob("20*.json")) if before is None or f.stem < before]
    for f in files[-limit:]:
        m = json.loads(f.read_text()).get("markets", {}).get(sym, {})
        o = m.get("outcome")
        if not o:
            continue
        if regime and m.get("bias", {}).get("regime") != regime:
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
    """One complete read: view = 'preopen' (9:25 ET cut), 'now' (latest) or 'overnight' (7:50 PM CT cut)."""
    NOTES.clear(); SOURCES.clear()
    print(f"[{view}] session {target} · evidence cut at {as_of:%Y-%m-%d %H:%M} ET", file=sys.stderr)
    marks = None
    # overnight: changes run from the evening's 4 PM close to the cut, so Asia's session counts
    quotes = fetch_quotes(MACRO_TICKERS, next_weekday(target) if view == "overnight" else target, as_of, marks)
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

    regime = market_regime(quotes, as_of)
    pre = {}
    for sym, cfg in MARKETS.items():
        pos, why = None, None
        try:
            pos, why = analyse_positioning(cfg, quotes, next_weekday(target) if view == "overnight" else target,
                                           as_of, live and view != "overnight")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            why = f"positioning failed ({type(e).__name__})"
        if why:
            note(f"{sym}: dealer positioning skipped, {why}. The read is macro only.")
        if view == "overnight":
            base = overnight_signals(cfg, quotes, target)
        elif view == "now" and sess:
            # no intraday price-action signals: the call comes from macro, positioning and news
            base = macro_signals(cfg, quotes)
        else:
            base = macro_signals(cfg, quotes)
        day = next_weekday(target) if view == "overnight" else target   # the session being called
        extra = [x for x in (fed_pressure_signal(as_of), surprise_signal(events, as_of),
                             None if view == "overnight" else vol_crush_signal(events, quotes, as_of),
                             policy_gap_signal(as_of), stress_signal(as_of), macro_z_signal(as_of),
                             None if view == "overnight" else fomc_drift_signal(events, as_of),
                             vrp_signal(cfg, quotes, target), fedwatch_signal(as_of), fed_leak_signal(as_of),
                             vix_roll_signal(as_of), auction_signal(as_of), funding_signal(as_of),
                             month_end_signal(as_of), move_signal(as_of), copper_gold_signal(as_of),
                             fomc_cycle_signal(day), pre_holiday_signal(day), refunding_signal(day)) if x] + liquidity_signals(cfg, as_of)
        drift = next((x for x in extra if x["key"] == "fomcdrift" and x["score"]), None)
        od = fed_odds(as_of, next_fomc(as_of)) if drift else None
        if od and od["certainty"] >= 0.9:
            # A near-certain FedWatch read lowers pre-meeting uncertainty, which mutes the pre-FOMC drift
            # (Bonini, Huang & Simaan, SSRN 5093703).
            drift["weight"] = round(drift["weight"] * 0.6, 3)
            drift["text"] += f" Futures price the outcome about {od['certainty']*100:.0f}% one-sided, which has muted the drift, so it counts less."
        base = base + extra
        # learn only from sessions graded before this one, so every read stays point-in-time
        key = as_of.strftime("%Y-%m-%dT%H%M") if view == "now" else target.isoformat()
        mult = learned_multipliers(hist, sym, key, 1000 if view == "now" else 120, regime)
        dealer = (dealer_signals(cfg, pos, quotes, events, as_of) + (intraday_dealer_signals(cfg, pos, as_of, sess)
                  if view == "now" else [])) if pos else []
        pre[sym] = (pos, apply_learning(base, mult) + dealer)

    if view == "overnight":
        hours = 3.5          # 2 to 8 AM CT carries about half a cash session of variance
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
        bias["regime"] = regime
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
            "expected_range": expected_range(cfg, pos, quotes, px, hours, view == "now" and bool(sess) and sess[0] == "NY"),
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
                  {"name": "London to pre-market", "start": datetime.combine(next_weekday(target), ON_START, ET).isoformat(),
                   "end": datetime.combine(next_weekday(target), ON_END, ET).isoformat()} if view == "overnight" else
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
    t0, t1 = [x.isoformat() for x in grade_window(kind, date.fromisoformat(read["date"]))]
    for sym, m in read["markets"].items():            # keep an existing grade on re-runs
        oo = (old.get("markets", {}).get(sym, {}).get("outcome") or {}) if old else {}
        if oo.get("end") == t1 and oo.get("start") == t0:
            m["outcome"] = old["markets"][sym]["outcome"]
    path.write_text(json.dumps(read, indent=1, default=float))
    grade_history(hist, kind)
    saved = json.loads(path.read_text())
    for sym in read["markets"]:
        read["markets"][sym]["outcome"] = saved["markets"][sym].get("outcome")
        read["markets"][sym]["scorecard"] = record_stats(hist, sym)
    return read


def reuse_locked(hist, day, cut, now):
    """
    Keep a locked overnight call as it was. Once the clock passes midnight the option snapshot is
    from the next day and can't be used point-in-time, so rebuilding would drop the dealer votes.
    Reused when it was built at the cut, or (if the first locked run was late) from the last preview.
    """
    path = hist / f"{day.isoformat()}.json"
    if not path.exists():
        return None
    try:
        old = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None
    at_cut = old.get("as_of") == cut.isoformat(timespec="seconds")
    late = now > cut + timedelta(minutes=40) and old.get("as_of", "") < cut.isoformat()
    if not (at_cut or late):
        return None
    if old.get("mode") == "live":
        old["mode"] = "point-in-time"
        path.write_text(json.dumps(old, indent=1, default=float))
    grade_history(hist, "overnight")
    old = json.loads(path.read_text())
    for sym in old.get("markets", {}):
        old["markets"][sym]["scorecard"] = record_stats(hist, sym)
    return old


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
DAILY_CACHE = {}


def daily_closes(ticker, days=70):
    """[(date, close)] daily, oldest → newest, from Yahoo. Cached per run (callers filter by date)."""
    if FIXTURES:
        return []
    if (ticker, days) in DAILY_CACHE:
        return DAILY_CACHE[(ticker, days)]
    DAILY_CACHE[(ticker, days)] = _daily_closes(ticker, days)
    return DAILY_CACHE[(ticker, days)]


def _daily_closes(ticker, days):
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


# ─────────────────────────────── Sunday gap ────────────────────────────────
GAP_LOCK = time(17, 55)    # 4:55 PM CT, five minutes before Globex reopens
HOURLY_CACHE = {}


def hourly_bars(ticker, days=400):
    """60-minute bars (incl. Globex), ET-indexed. Yahoo keeps about two years of them."""
    if FIXTURES:
        return None
    if ticker in HOURLY_CACHE:
        return HOURLY_CACHE[ticker]
    h = None
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period=f"{days}d", interval="60m", prepost=True, auto_adjust=False)
        h = None if h.empty else h
        if h is not None:
            h.index = h.index.tz_convert(ET)
    except Exception as e:  # noqa: BLE001
        print(f"hourly {ticker}: {type(e).__name__} {e}", file=sys.stderr)
        h = None
    HOURLY_CACHE[ticker] = h
    return h


def gap_sunday(now):
    """The Sunday whose reopen gap is in play: from Friday 4 PM ET until Monday's New York open."""
    d, wd, t = now.date(), now.weekday(), now.time()
    if wd == 4 and t >= time(16):
        return d + timedelta(days=2)
    if wd == 5:
        return d + timedelta(days=1)
    if wd == 6:
        return d
    if wd == 0 and t < time(9, 30):
        return d - timedelta(days=1)
    return None


def _px_at(h, when):
    x = h[h.index < when] if h is not None else None
    return float(x["Close"].iloc[-1]) if x is not None and not x.empty else None


def gap_history(fut, sunday):
    """Past weekends before `sunday`: (gap % from Friday's close to Sunday's open, BTC % over the weekend)."""
    h, b = hourly_bars(fut), hourly_bars("BTC-USD")
    if h is None:
        return []
    out = []
    for s_ in sorted({i.date() for i in h.index if i.weekday() == 6}):
        if s_ >= sunday:
            continue
        fri = s_ - timedelta(days=2)
        fclose = datetime.combine(fri, time(17, 0), ET)
        c = h[(h.index < fclose) & (h.index.date == fri)]
        o = h[(h.index >= datetime.combine(s_, time(18), ET)) & (h.index < datetime.combine(s_, time(19), ET))]
        if c.empty or o.empty:
            continue
        c0, o0 = float(c["Close"].iloc[-1]), float(o["Open"].iloc[0])
        b0, b1 = _px_at(b, fclose + timedelta(minutes=1)), _px_at(b, datetime.combine(s_, time(18), ET))
        out.append((pct(o0, c0), pct(b1, b0) if b0 and b1 else None))
    return out


def friday_lean(sym, fri):
    """The model's own macro + options read at the end of Friday's New York session (the last Now call)."""
    d = OUT / "history" / "now"
    for f in sorted(d.glob(f"{fri.isoformat()}T*.json"), reverse=True) if d.exists() else []:
        try:
            m = json.loads(f.read_text())["markets"].get(sym)
            if m:
                return m["bias"].get("own_score", m["bias"]["score"]), f.stem
        except Exception:  # noqa: BLE001
            continue
    return None, None


def build_gap(sunday, as_of):
    """
    The Sunday reopen gap: points from Friday's 4 PM CT settle-time close to the 5 PM CT reopen.
    Size comes from the market's own past weekends; direction from what trades or changes over the
    weekend (Bitcoin, the Saudi market's Sunday session) plus the model's Friday macro read.
    """
    fri = sunday - timedelta(days=2)
    fclose = datetime.combine(fri, time(17, 0), ET)
    btc = hourly_bars("BTC-USD")
    b0, b1 = _px_at(btc, fclose + timedelta(minutes=1)), _px_at(btc, as_of)
    r_btc = pct(b1, b0) if b0 and b1 else None
    tasi = None
    rows = [r for r in daily_closes("^TASI.SR", 10) if r[0] <= sunday]
    if len(rows) >= 2 and rows[-1][0] == sunday and as_of >= datetime.combine(sunday, time(8, 0), ET):
        tasi = pct(rows[-1][1], rows[-2][1])
    markets = {}
    for sym, cfg in MARKETS.items():
        fut = cfg["future"]
        px = _px_at(hourly_bars(fut), fclose)
        hist = gap_history(fut, sunday)
        if not px or len(hist) < 12:
            continue
        typical = float(np.median([abs(g) for g, _ in hist]))
        pairs = [(g, r) for g, r in hist if r is not None]
        a = beta = 0.0
        if len(pairs) >= 20:
            x = np.array([r for _, r in pairs]); y = np.array([g for g, _ in pairs])
            beta, a = (np.polyfit(x, y, 1) if np.std(x) > 0 else (0.0, float(np.mean(y))))
        votes = []
        if r_btc is not None:
            votes.append(sig("btc", "Bitcoin over the weekend", 1 if r_btc > 1.5 else -1 if r_btc < -1.5 else 0, 1.0,
                             f"Bitcoin {r_btc:+.1f}% since Friday's futures close. It trades all weekend, so it is the live read on risk appetite.",
                             "Weekend"))
        if tasi is not None:
            votes.append(sig("tasi", "Saudi market (Sunday)", 1 if tasi > 0.5 else -1 if tasi < -0.5 else 0, 0.5,
                             f"Tadawul {tasi:+.1f}% in its Sunday session, the first big equity market to trade after the weekend's news.",
                             "Weekend"))
        lean, src = friday_lean(sym, fri)
        if lean is not None:
            votes.append(sig("friday", "Friday macro read", 1 if lean > NEUTRAL_BAND else -1 if lean < -NEUTRAL_BAND else 0, 1.0,
                             f"The model's last Friday read scored {lean:+.2f} on macro and options, before the weekend.", "Weekend"))
        tot = sum(v["weight"] for v in votes) or 1
        score = sum(v["score"] * v["weight"] for v in votes) / tot
        other = [v for v in votes if v["key"] != "btc"]
        tilt = sum(v["score"] * v["weight"] for v in other) / (sum(v["weight"] for v in other) or 1)
        pred = (a + beta * r_btc if r_btc is not None else float(a)) + 0.5 * typical * tilt
        if abs(pred) < 1e-9:
            pred = 0.01 * (1 if score >= 0 else -1)
        ratio = abs(pred) / typical if typical else 0
        markets[sym] = {
            "name": cfg["name"], "bias": "Bullish" if pred > 0 else "Bearish",
            "conviction": "High" if ratio >= 0.7 else "Medium" if ratio >= 0.3 else "Low",
            "friday_close": round(px, 2), "pred_pts": round(px * pred / 100, 1), "pred_pct": round(pred, 3),
            "typical_pts": round(px * typical / 100, 1), "typical_pct": round(typical, 3),
            "beta_btc": round(float(beta), 3), "weekends": len(hist), "score": round(score, 3), "signals": votes,
            "outcome": None,
        }
    return {"sunday": sunday.isoformat(), "friday": fri.isoformat(), "as_of": as_of.isoformat(timespec="seconds"),
            "locked": as_of >= datetime.combine(sunday, GAP_LOCK, ET), "btc_pct": r_btc and round(r_btc, 2),
            "tasi_pct": tasi and round(tasi, 2), "markets": markets}


def grade_gaps(d):
    now = datetime.now(ET)
    for f in sorted(d.glob("20*.json"))[-20:]:
        g = json.loads(f.read_text())
        s_ = date.fromisoformat(g["sunday"])
        t0 = datetime.combine(s_, time(18), ET)
        if now < t0 + timedelta(minutes=10):
            continue
        changed = False
        for sym, m in g["markets"].items():
            if m.get("outcome"):
                continue
            h = fetch_bars(MARKETS[sym]["future"], t0, t0 + timedelta(minutes=30))
            if h is None:
                hb = hourly_bars(MARKETS[sym]["future"])
                h = None if hb is None else hb[(hb.index >= t0) & (hb.index < t0 + timedelta(hours=1))]
                h = None if h is None or h.empty else h
            if h is None:
                continue
            o = float(h["Open"].iloc[0])
            mv = o - m["friday_close"]
            m["outcome"] = {"open": round(o, 2), "gap_pts": round(mv, 2),
                            "result": "hit" if (mv > 0) == (m["bias"] == "Bullish") and mv != 0 else "miss"}
            changed = True
        if changed:
            f.write_text(json.dumps(g, indent=1, default=float))


def sunday_gap(now):
    d = OUT / "history" / "gap"
    d.mkdir(parents=True, exist_ok=True)
    s_ = gap_sunday(now)
    if s_ and not os.environ.get("REPORT_DATE", "").strip():
        lock = datetime.combine(s_, GAP_LOCK, ET)
        path = d / f"{s_.isoformat()}.json"
        old = json.loads(path.read_text()) if path.exists() else None
        if not (old and old.get("locked")):
            try:
                g = build_gap(s_, min(now, lock))
                if g["markets"]:
                    path.write_text(json.dumps(g, indent=1, default=float))
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                note("Sunday gap read failed")
    grade_gaps(d)
    files = sorted(d.glob("20*.json"))
    if not files:
        return None
    cur = json.loads(files[-1].read_text())
    past = []
    for f in files[-26:]:
        g = json.loads(f.read_text())
        for sym, m in g["markets"].items():
            if m.get("outcome"):
                past.append({"sunday": g["sunday"], "sym": sym, "bias": m["bias"], "pred_pts": m["pred_pts"],
                             "gap_pts": m["outcome"]["gap_pts"], "result": m["outcome"]["result"]})
    cur["record"] = past
    return cur


# ─────────────────────────── news, gamma map, move ladder ───────────────────────────
NEWS_FEEDS = [
    ("Fox Business", "https://moxie.foxbusiness.com/google-publisher/markets.xml"),
    ("Fox Business", "https://moxie.foxbusiness.com/google-publisher/economy.xml"),
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
    ("CNBC", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
    ("Investing.com", "https://www.investing.com/rss/news_25.rss"),
]


def _strip_html(x):
    import html as _h
    import re as _re
    return _re.sub(r"\s+", " ", _h.unescape(_re.sub(r"<[^>]+>", " ", x or ""))).strip()


def _og_image(url):
    """The article's share image, when its feed item carries none."""
    import re as _re
    try:
        r = requests.get(url, headers=BROWSER_UA, timeout=(4, 6))
        m = _re.search(r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)', r.text[:200000]) or \
            _re.search(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image', r.text[:200000])
        return m.group(1) if m else None
    except Exception:  # noqa: BLE001
        return None


def fetch_news(now, limit=40):
    """Headlines with pictures from free market news feeds. Links open the publisher's own article."""
    import xml.etree.ElementTree as ET_
    from email.utils import parsedate_to_datetime
    items, seen, ok = [], set(), 0
    for src, url in NEWS_FEEDS:
        try:
            r = requests.get(url, headers=BROWSER_UA, timeout=(5, 10))
            r.raise_for_status()
            root = ET_.fromstring(r.content)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"news {src}: {type(e).__name__}", file=sys.stderr)
            continue
        for it in root.iter("item"):
            title = _strip_html(it.findtext("title"))
            link = (it.findtext("link") or "").strip()
            if not title or not link.startswith("http"):
                continue
            key = title.lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            img = None
            for el in it.iter():
                tag = el.tag.split("}")[-1]
                u = el.attrib.get("url") or el.attrib.get("href")
                if tag in ("content", "thumbnail") and u and not u.endswith((".mp4", ".m3u8")):
                    img = u
                    if tag == "content":
                        break
                if tag == "enclosure" and u and "image" in el.attrib.get("type", "image"):
                    img = img or u
            ts = None
            for tg in ("pubDate", "{http://purl.org/dc/elements/1.1/}date"):
                raw = it.findtext(tg)
                if raw:
                    try:
                        ts = parsedate_to_datetime(raw)
                    except Exception:  # noqa: BLE001
                        try:
                            ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                        except Exception:  # noqa: BLE001
                            ts = None
                    break
            if ts and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            summ = _strip_html(it.findtext("description"))
            items.append({"title": title, "link": link, "source": src, "image": img,
                          "summary": (summ[:240] + "…") if len(summ) > 240 else summ,
                          "ts": ts.astimezone(ET).isoformat(timespec="minutes") if ts else None})
    items = [i for i in items if i["ts"] and (now - datetime.fromisoformat(i["ts"])).total_seconds() < 3 * 86400]
    items.sort(key=lambda i: i["ts"], reverse=True)
    items = items[:limit]
    for i in [i for i in items if not i["image"]][:15]:
        i["image"] = _og_image(i["link"])
    stamp("News feeds", ok > 0, now.isoformat(), f"{ok} of {len(NEWS_FEEDS)} feeds")
    return {"generated": now.isoformat(timespec="seconds"), "items": items}


def gamma_map(days=20):
    """Dealer levels from each morning's locked pre-open read: how the walls and flip have moved."""
    out = {}
    for f in sorted((OUT / "history").glob("20*.json"))[-60:]:
        try:
            rep_ = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        for sym, m in rep_.get("markets", {}).items():
            L, dl = m.get("levels") or {}, m.get("dealer") or {}
            if L.get("call_wall") is None:
                continue
            out.setdefault(sym, []).append({"date": rep_["date"], "price": L.get("price"), "flip": L.get("flip"),
                                            "call_wall": L.get("call_wall"), "put_wall": L.get("put_wall"),
                                            "net_gex_bn": dl.get("net_gex_bn")})
    return {k: v[-days:] for k, v in out.items()}


# Share of a full day's variance in each session (cash hours carry roughly two thirds of it;
# a rough split for index futures, used only to size the ladder).
SESSION_VAR = [("Asia", "5 PM – 2 AM CT", 0.15), ("London", "2 – 8:30 AM CT", 0.20), ("New York", "8:30 AM – 3 PM CT", 0.65)]


def move_ladder(read):
    """1σ and 2σ price levels for each session from the vol index, scaled by dealer gamma."""
    out = {}
    q = read.get("macro", {})
    for sym, m in read.get("markets", {}).items():
        cfg = MARKETS[sym]
        px = (m.get("levels") or {}).get("price")
        v = q.get(cfg["vol_index"], {}).get("last")
        if not px or not v:
            continue
        day = px * v / 100 / math.sqrt(252)
        gm = 1.0
        net = (m.get("dealer") or {}).get("net_gex_bn")
        if net is not None:
            gm = 0.85 if net > 0 else 1.15
        rows = []
        for name, hours, share in SESSION_VAR:
            sd = day * math.sqrt(share) * gm
            rows.append({"name": name, "hours": hours, "sd": round(sd, 1),
                         "up1": round(px + sd, 2), "dn1": round(px - sd, 2), "up2": round(px + 2 * sd, 2), "dn2": round(px - 2 * sd, 2)})
        out[sym] = {"price": px, "vol": round(v, 2), "vol_name": cfg["vol_index"].lstrip("^"), "day_sd": round(day * gm, 1),
                    "gamma": None if net is None else ("long" if net > 0 else "short"), "sessions": rows}
    return out


def resolve_overnight(now):
    """
    The evening whose overnight (London) call is showing, its evidence cut, and whether it's still live.
    The call for the next morning previews from 4 PM ET and locks at 2:50 AM ET (1:50 AM CT).
    Sunday evening gets one (for Monday); Friday has none, since futures shut for the weekend.
    """
    forced = os.environ.get("REPORT_DATE", "").strip()
    if forced:
        d = date.fromisoformat(forced)
        return d, datetime.combine(next_weekday(d), ON_CUT, ET), False
    d = now.date()
    while d.weekday() in (4, 5) or now < datetime.combine(d, time(16, 0), ET):
        d -= timedelta(days=1)
    cut = datetime.combine(next_weekday(d), ON_CUT, ET)
    return (d, now, True) if now < cut else (d, cut, False)


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

    # Overnight: 8 PM → 8 AM CT. A live preview from 4 PM ET, locked at 7:50 PM CT
    on_day, on_cut, on_live = resolve_overnight(now)
    overnight = None if on_live or backfill else reuse_locked(hist_on, on_day, on_cut, now)
    if backfill and (on_day.weekday() == 4 or (hist_on / f"{on_day.isoformat()}.json").exists()):
        overnight = {"skip": True}
    if overnight is None:
        t0, t1 = grade_window("overnight", on_day)
        overnight = finish(build_view(on_day, on_cut, on_live, now, "overnight", hist_on,
                                      ("London to pre-market", t0, t1, next_weekday(on_day))), hist_on, "overnight")

    report = {"generated": now.isoformat(timespec="seconds"), "red": red_folder(now),
              "ny": {"preopen": preopen, "now": current}, "overnight": overnight, "gap": sunday_gap(now),
              "week": sorted(WEEK_EVENTS, key=lambda e: e["ts"])}
    if not backfill:
        try:
            report["gamma_map"] = gamma_map()
            report["ladder"] = move_ladder(current if current and current.get("markets") else preopen)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        try:
            (OUT / "news.json").write_text(json.dumps(fetch_news(now), indent=1))
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            note("News feeds failed")
    latest = OUT / "latest.json"
    if backfill and latest.exists():
        # a backfill only adds history; keep today's reads on the page, refresh the records
        try:
            old = json.loads(latest.read_text())
            if "ny" in old:
                report = old
                for sess, hist in (("ny", hist_ny), ("overnight", hist_on)):
                    if not old.get(sess):
                        continue
                    r = old[sess]["preopen"] if sess == "ny" else old[sess]
                    for sym in r["markets"]:
                        r["markets"][sym]["scorecard"] = record_stats(hist, sym)
        except Exception:  # noqa: BLE001
            pass
    report["preopen"] = report["ny"]["preopen"]       # older app versions read these keys
    report["now"] = report["ny"]["now"]
    latest.write_text(json.dumps(report, indent=1, default=float))
    for name, v in (("NY PRE-OPEN", preopen), ("NY NOW", current), ("OVERNIGHT", overnight)):
        if v and "markets" in v:
            print(name, v["date"], json.dumps({s: f"{m['bias']['bias']} {m['bias']['score']:+.2f}"
                                               for s, m in v["markets"].items()}))


def backfill_range():
    """
    REPORT_BACKFILL=YYYY-MM-DD:YYYY-MM-DD rebuilds every weekday in the range point-in-time, skipping
    days that already have a live record (those keep their options data). Yahoo keeps 5-minute bars for
    about 60 days, so older mornings can't be graded on the 30-minute window and are skipped.
    """
    a, b = [date.fromisoformat(x) for x in os.environ["REPORT_BACKFILL"].split(":")]
    a = max(a, datetime.now(ET).date() - timedelta(days=55))
    d = a
    while d <= b:
        if d.weekday() < 5 and not (OUT / "history" / f"{d.isoformat()}.json").exists():
            os.environ["REPORT_DATE"] = d.isoformat()
            _FRED.clear()                     # each day asks ALFRED for its own vintage
            try:
                main()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                print(f"backfill {d}: failed", file=sys.stderr)
        d += timedelta(days=1)
    os.environ["REPORT_DATE"] = ""


if __name__ == "__main__":
    if os.environ.get("REPORT_BACKFILL", "").strip():
        backfill_range()
    else:
        main()
