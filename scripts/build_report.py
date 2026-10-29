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


# ───────────────────────── news: earnings + headlines ──────────────────────
def report_time(e) -> datetime | None:
    """When an earnings release hit the tape, from the calendar's date and session."""
    d = date.fromisoformat(e["date"])
    if e["when"].startswith("before"):
        return datetime.combine(d, time(7, 0), ET)
    if e["when"].startswith("after") or e["when"] == "last night":
        return datetime.combine(d, time(16, 5), ET)
    return None


def eps_surprise(sym, rep_at: datetime, as_of: datetime):
    """(reported, estimate, surprise %) from Yahoo, only if that report is public by as_of."""
    if FIXTURES:
        return None
    try:
        import yfinance as yf
        df = yf.Ticker(sym.replace(".", "-")).get_earnings_dates(limit=8)
        if df is None or df.empty:
            return None
        df.index = df.index.tz_convert(ET)
        near = df[(abs((df.index - rep_at).total_seconds()) < 36 * 3600) & (df.index <= as_of)]
        if near.empty:
            return None
        row = near.iloc[0]
        act, est = row.get("Reported EPS"), row.get("EPS Estimate")
        if act is None or est is None or (isinstance(act, float) and math.isnan(act)):
            return None
        sp = row.get("Surprise(%)")
        if sp is None or (isinstance(sp, float) and math.isnan(sp)):
            sp = (act - est) / abs(est) * 100 if est else None
        return float(act), float(est), (float(sp) if sp is not None else None)
    except Exception as e:  # noqa: BLE001
        print(f"eps {sym}: {type(e).__name__} {e}", file=sys.stderr)
        return None


def earnings_signal(cfg, earnings, as_of):
    """
    Mega-caps that have already reported by the cutoff: how each stock is trading vs its
    pre-report close (the market's verdict), with the EPS beat or miss alongside.
    """
    seen, rows = set(), []
    for e in earnings:
        sym = e["symbol"]
        if sym not in cfg["megacaps"] or sym in seen:
            continue
        rep_at = report_time(e)
        if not rep_at or rep_at > as_of:
            continue                                  # not out yet at this cutoff
        seen.add(sym)
        base_day = rep_at.date() if rep_at.time() >= time(16) else prev_weekday(rep_at.date())
        h = fetch_bars(sym.replace(".", "-"), datetime.combine(base_day, time(15, 0), ET), as_of)
        react = None
        if h is not None:
            b0 = h[h.index <= datetime.combine(base_day, time(16, 0), ET)]
            if not b0.empty:
                react = pct(float(h["Close"].iloc[-1]), float(b0["Close"].iloc[-1]))
        es = eps_surprise(sym, rep_at, as_of)
        if es is None and e.get("eps") is not None and e.get("eps_est") is not None:
            sp = e.get("surprise")
            es = (e["eps"], e["eps_est"], sp if sp is not None else
                  ((e["eps"] - e["eps_est"]) / abs(e["eps_est"]) * 100 if e["eps_est"] else None))
        if react is None and es is None:
            continue
        rows.append({"symbol": sym, "react": react, "eps": es})
    if not rows:
        return None
    reacts = [r["react"] for r in rows if r["react"] is not None]
    surps = [r["eps"][2] for r in rows if r["eps"] and r["eps"][2] is not None]
    if reacts:   # price reaction is the verdict; EPS can beat and still sell off on guidance
        avg = sum(reacts) / len(reacts)
        sc = 1 if avg > 1.5 else -1 if avg < -1.5 else 0
    else:
        avg = sum(surps) / len(surps)
        sc = 1 if avg > 5 else -1 if avg < -5 else 0
    parts = []
    for r in rows:
        bits = []
        if r["eps"]:
            a, est, sp = r["eps"]
            bits.append(f"EPS {a:.2f} vs {est:.2f} est" + (f" ({'beat' if sp > 0 else 'miss'} {sp:+.0f}%)" if sp is not None else ""))
        if r["react"] is not None:
            bits.append(f"stock {r['react']:+.1f}%")
        parts.append(f"{r['symbol']}: " + ", ".join(bits))
    return sig("earnings", "Mega-cap earnings", sc, 1.25,
               "; ".join(parts) + ". " +
               ("The market is buying the results, and mega-caps carry the index." if sc > 0 else
                "The market is selling the results, and mega-caps drag the index." if sc < 0 else
                "A mixed or muted reaction, so earnings aren't moving the index much."), "News")


# ───────────────────── official macro data (Fed reaction function) ────────────────
# Framework: the user's macro notes. The Fed reacts to three things (jobs, inflation,
# activity); hot data → contractionary Fed → bearish for stocks, cool data → expansionary
# → bullish. Only official numbers count, because those are what the Fed watches.
# Free source: FRED's public CSV download (no key). ISM, Chicago PMI and Conference
# Board confidence are licensed and not on FRED, so they can't be scored here.
_FRED: dict = {}


def fred(series):
    """[(date, value)] oldest → newest from FRED's public CSV. Cached per run."""
    if series in _FRED:
        return _FRED[series]
    out = []
    if not FIXTURES:
        try:
            r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}",
                             headers=UA, timeout=30)
            r.raise_for_status()
            for line in r.text.strip().splitlines()[1:]:
                d, _, v = line.partition(",")
                try:
                    out.append((date.fromisoformat(d.strip()), float(v)))
                except ValueError:
                    continue          # "." marks a missing value
        except Exception as e:  # noqa: BLE001
            print(f"fred {series}: {type(e).__name__} {e}", file=sys.stderr)
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
    "Initial jobless claims (4-wk avg)": ("ICSA", lambda r: sum(v for _, v in r[-4:]) / 4 / 1000,
                                          lambda x: x < 250, lambda x: x > 350, "K", 5),
    "Payrolls (3-mo avg change)": ("PAYEMS", lambda r: (r[-1][1] - r[-4][1]) / 3,
                                   lambda x: x > 200, lambda x: x < 75, "K", 36),
    "CPI y/y": ("CPIAUCNS", _yoy, lambda x: x > 2.2, lambda x: x < 1.8, "%", 45),
    "Core CPI y/y": ("CPILFENS", _yoy, lambda x: x > 2.2, lambda x: x < 1.8, "%", 45),
    "PPI m/m": ("PPIFIS", _mom, lambda x: x > 0.2, lambda x: x < 0.0, "%", 45),
}


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
    if len(votes) < 3:
        note("Official macro data (FRED) unavailable, Fed-pressure gauge skipped")
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
}


def _fresh_for(rows, released: date, kind):
    """The newest observation must be the period this release reports, or FRED hasn't updated yet."""
    if not rows:
        return False
    obs = rows[-1][0]
    if kind == "week":
        return 0 <= (released - obs).days <= 8
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


NEWS_FEEDS = {
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "CNBC Top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "MarketWatch Pulse": "https://feeds.content.dowjones.io/public/rss/mw_marketpulse",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "Investing.com": "https://www.investing.com/rss/news_25.rss",
}
NEWS_DIR = OUT / "news"
# A small finance lexicon in the spirit of Loughran & McDonald (2011): words that move
# risk appetite, not general sentiment. Phrases are checked before single words.
POS_PHRASES = ["record high", "all-time high", "beats estimates", "tops estimates", "raises guidance",
               "rate cut", "cuts rates", "soft landing", "cooling inflation", "inflation cools",
               "trade deal", "ceasefire", "better than expected", "beat expectations"]
NEG_PHRASES = ["sell-off", "selloff", "misses estimates", "cuts guidance", "lowers guidance",
               "rate hike", "raises rates", "hotter than expected", "worse than expected",
               "inflation heats", "trade war", "government shutdown", "missed expectations"]
POS_WORDS = {"rally", "rallies", "rallied", "surge", "surges", "surged", "soar", "soars", "soared", "jump",
             "jumps", "jumped", "gain", "gains", "climb", "climbs", "rebound", "rebounds", "upgrade",
             "upgrades", "beat", "beats", "strong", "stronger", "optimism", "bullish", "boom", "eases",
             "easing", "dovish", "stimulus", "outperform", "upbeat", "accelerates", "higher", "rises", "rose",
             "rise", "climbed", "gained", "advance", "advances", "rallying", "spike", "spikes", "spiked"}
# pure direction words: their meaning for stocks flips when the subject is yields, oil, VIX...
DIRECTION = {"rally", "rallies", "rallied", "surge", "surges", "surged", "soar", "soars", "soared", "jump",
             "jumps", "jumped", "gain", "gains", "climb", "climbs", "climbed", "gained", "rebound", "rebounds",
             "higher", "rises", "rose", "rise", "advance", "advances", "rallying", "spike", "spikes", "spiked",
             "plunge", "plunges", "plunged", "tumble", "tumbles", "tumbled", "slump", "slumps", "sink", "sinks",
             "sank", "slide", "slides", "drop", "drops", "dropped", "fall", "falls", "fell", "lower"}
NEG_WORDS = {"plunge", "plunges", "plunged", "tumble", "tumbles", "tumbled", "slump", "slumps", "sink",
             "sinks", "sank", "slide", "slides", "drop", "drops", "dropped", "fall", "falls", "fell",
             "selloff", "miss", "misses", "missed", "downgrade", "downgrades", "weak", "weaker", "fears",
             "fear", "recession", "hawkish", "tariff", "tariffs", "war", "crisis", "bearish", "slowdown",
             "layoffs", "warns", "warning", "default", "shutdown", "lower", "losses", "slowing", "worst", "turmoil"}


INVERSE_SUBJECTS = {"yield", "yields", "treasury", "treasuries", "vix", "volatility", "oil", "crude",
                    "inflation", "unemployment", "jobless", "dollar", "cpi", "pce"}
EQUITY_WORDS = {"stock", "stocks", "nasdaq", "s&p", "dow", "futures", "shares", "equities", "market",
                "markets", "tech", "wall", "index", "chip", "chips"}


def tone(title: str) -> int:
    """+1 risk-on, -1 risk-off, 0 neutral, for one headline."""
    t = title.lower()
    p = sum(t.count(x) for x in POS_PHRASES)
    n = sum(t.count(x) for x in NEG_PHRASES)
    for x in POS_PHRASES + NEG_PHRASES:
        t = t.replace(x, " ")
    words = re.findall(r"[a-z&\-]+", t)
    ws = set(words)
    inverse = bool(ws & INVERSE_SUBJECTS) and not ws & EQUITY_WORDS   # "yields drop", "oil surges"
    softens = {"ease", "eases", "easing", "fade", "fades", "recede", "recedes", "abate"}
    for i, w in enumerate(words):
        prev, nxt = words[max(0, i - 3):i], words[i + 1:i + 3]
        if w in POS_WORDS:
            v = -1 if ("not" in prev or "no" in prev) else 1
        elif w in NEG_WORDS:
            if "despite" in prev or "without" in prev or "no" in prev:   # "rise despite tariff fears"
                continue
            v = 1 if set(nxt) & softens else -1                          # "recession fears ease"
        else:
            continue
        if inverse and w in DIRECTION:
            v = -v
        if v > 0:
            p += 1
        else:
            n += 1
    return (p > n) - (p < n)


def _parse_rss(xml_text, source):
    import xml.etree.ElementTree as ElT
    from email.utils import parsedate_to_datetime
    out = []
    root = ElT.fromstring(xml_text)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        pub = it.findtext("pubDate")
        if not title or not pub:
            continue
        try:
            ts = parsedate_to_datetime(pub)
            ts = (ts if ts.tzinfo else ts.replace(tzinfo=ZoneInfo("UTC"))).astimezone(ET)
        except Exception:  # noqa: BLE001
            continue
        out.append({"title": title, "source": source, "ts": ts.isoformat()})
    return out


_NEWS: dict | None = None


def collect_headlines():
    """
    Pull every feed once per run and add new headlines to a dated archive, so a
    read rebuilt later still sees exactly what was public before its cutoff.
    """
    global _NEWS
    if _NEWS is not None:
        return _NEWS
    live, ok = [], 0
    if not FIXTURES:
        for name, url in NEWS_FEEDS.items():
            try:
                r = requests.get(url, headers=UA, timeout=20)
                r.raise_for_status()
                live += _parse_rss(r.text, name)
                ok += 1
            except Exception as e:  # noqa: BLE001
                print(f"news {name}: {type(e).__name__} {e}", file=sys.stderr)
    _NEWS = {"ok": ok, "n": len(live)}
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    by_day: dict[str, list] = {}
    for h in live:
        by_day.setdefault(h["ts"][:10], []).append(h)
    for day, items in by_day.items():
        f = NEWS_DIR / f"{day}.json"
        old = json.loads(f.read_text()) if f.exists() else []
        have = {x["title"].lower() for x in old}
        new = [x for x in items if x["title"].lower() not in have and not have.add(x["title"].lower())]
        if new:
            f.write_text(json.dumps(sorted(old + new, key=lambda x: x["ts"]), indent=0))
    for old in sorted(NEWS_DIR.glob("20*.json"))[:-45]:   # keep ~6 weeks
        old.unlink()
    return _NEWS


def load_headlines(as_of):
    """Archived headlines from the cutoff's day and the two days before it."""
    out = []
    for k in range(3):
        f = NEWS_DIR / f"{(as_of.date() - timedelta(days=k)).isoformat()}.json"
        if f.exists():
            out += json.loads(f.read_text())
    return out


def news_signal(as_of, hours=16):
    """Net risk-on vs risk-off tone of market headlines in the hours before the cutoff."""
    start = (as_of - timedelta(hours=hours)).isoformat()
    st = collect_headlines()
    stamp("News headlines", st["ok"] > 0,
          detail=f"{st['ok']} of {len(NEWS_FEEDS)} feeds live · {st['n']} headlines")
    seen, items = set(), []
    for h in load_headlines(as_of):
        k = h["title"].lower()
        if start <= h["ts"] <= as_of.isoformat() and k not in seen:
            seen.add(k)
            items.append(dict(h, tone=tone(h["title"])))
    scored = [h for h in items if h["tone"]]
    if len(items) < 8 or len(scored) < 4:
        if items or not FIXTURES:
            note(f"Too few headlines before the cutoff ({len(items)}) to score news tone")
        return None
    pos = sum(h["tone"] > 0 for h in scored)
    neg = len(scored) - pos
    net = (pos - neg) / len(scored)
    sc = 1 if net > 0.15 else -1 if net < -0.15 else 0
    lead = sorted(scored, key=lambda h: h["ts"], reverse=True)
    ex = [h for h in lead if (h["tone"] > 0) == (sc > 0)][:2] if sc else lead[:2]
    return sig("news", "Headline tone", sc, 0.75,
               f"{pos} risk-on vs {neg} risk-off of {len(items)} market headlines in the {hours}h before the cutoff "
               f"(net {net:+.2f}). Latest: " + " · ".join(f"\"{h['title']}\" ({h['source']})" for h in ex) + ". " +
               ("The news flow is constructive." if sc > 0 else "The news flow is risk-off." if sc < 0
                else "The news flow is mixed."), "News")


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


def dealer_gex(S, c, T=None):
    T = c["T"] if T is None else T
    S = np.asarray(S, dtype=float)[..., None]
    return (c["CP"] * bs_gamma(S, c["K"], T, c["IV"]) * c["OI"] * 100 * S**2 * 0.01).sum(-1)


def dealer_delta(S, c, T=None, IV=None):
    T = c["T"] if T is None else T
    IV = c["IV"] if IV is None else IV
    d = bs_delta(S, c["K"], T, IV, c["CP"])
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
    # overnight futures
    c = chg(q, cfg["future"])
    if c is not None:
        sc = 1 if c > 0.25 else -1 if c < -0.25 else 0
        S.append(sig("overnight", "Overnight futures", sc, 1.5,
                     f"{cfg['future'][:2]} {c:+.2f}% vs yesterday's 4 PM ({q[cfg['future']]['last']:,.2f}). " +
                     ("Globex buyers did the lifting, and overnight strength tends to carry into the open."
                      if sc > 0 else "Selling came in overnight, so the open starts on the back foot."
                      if sc < 0 else "A quiet overnight. Direction will be set at the open."), "Macro"))
    # global equities
    g = {GLOBAL[t]: chg(q, t) for t in GLOBAL if chg(q, t) is not None}
    if g:
        avg = sum(g.values()) / len(g)
        sc = 1 if avg > 0.4 else -1 if avg < -0.4 else 0
        S.append(sig("global", "Global equities", sc, 1.0,
                     " · ".join(f"{k} {v:+.1f}%" for k, v in g.items()) + ". " +
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
    c = chg(q, "BTC-USD")
    if c is not None:
        sc = 1 if c > 2 else -1 if c < -2 else 0
        S.append(sig("crypto", "Crypto risk appetite", sc, 0.5,
                     f"Bitcoin {c:+.1f}% since yesterday's 4 PM. " +
                     ("Speculative appetite is firm overnight." if sc > 0 else
                      "Speculative de-risking overnight." if sc < 0 else "No signal."), "Macro"))
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


def overnight_signals(cfg, q):
    """Evening read for the 6 PM–8:30 AM Globex session, built from data up to 6 PM ET."""
    S = [s for s in macro_signals(cfg, q) if s["key"] not in ("overnight", "global", "volpre")]
    f = q.get(cfg["future"])
    mk = (f or {}).get("marks", {})
    sym = cfg["future"][:2]
    if f and "close" in mk:
        day = pct(mk["close"], f["prev"])
        sc = 1 if day > 0.4 else -1 if day < -0.4 else 0
        S.append(sig("day", "Day session", sc, 1.0,
                     f"{sym} {day:+.2f}% from yesterday's 4 PM to today's 4 PM. " +
                     ("A strong cash session tends to carry into Globex." if sc > 0 else
                      "A weak cash session tends to keep sellers active overnight." if sc < 0
                      else "A flat day, with no momentum handed off."), "Session"))
        ah = pct(f["last"], mk["close"])
        sc = 1 if ah > 0.2 else -1 if ah < -0.2 else 0
        S.append(sig("afterhours", "After-hours reaction", sc, 1.25,
                     f"{sym} {ah:+.2f}% from 4 PM to the 6 PM cutoff. " +
                     ("After-hours buyers, often an earnings beat, set the overnight tone." if sc > 0 else
                      "After-hours selling, often an earnings miss, sets a heavy tone." if sc < 0
                      else "No after-hours reaction."), "Session"))
    if f and "close" in mk and "lasthour" in mk:
        lh = pct(mk["close"], mk["lasthour"])
        sc = 1 if lh > 0.15 else -1 if lh < -0.15 else 0
        S.append(sig("lasthour", "Last-hour trend", sc, 0.75,
                     f"{sym} {lh:+.2f}% in the final hour (3 to 4 PM). " +
                     ("Buyers into the bell, the late-day flow that tends to persist (Baltussen et al.)." if sc > 0 else
                      "Sellers into the bell, and late-day weakness tends to persist." if sc < 0
                      else "A neutral close."), "Session"))
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


def session_signals(cfg, sess, as_of):
    """What the session in progress has done so far. Only bars up to as_of."""
    name, start, end, _ = sess
    h = fetch_bars(cfg["future"], start - timedelta(minutes=1), as_of)
    sym = cfg["future"][:2]
    if h is None or len(h) < 5:
        note(f"{sym}: no intraday bars for the {name} session yet")
        return []
    S, last, first = [], float(h["Close"].iloc[-1]), float(h["Open"].iloc[0])
    mins = (as_of - start).total_seconds() / 60
    drift = pct(last, first)
    sc = 1 if drift > 0.2 else -1 if drift < -0.2 else 0
    S.append(sig("sessdrift", f"{name} session so far", sc, 1.5,
                 f"{sym} {drift:+.2f}% since the {name} session opened ({first:,.2f} → {last:,.2f}). " +
                 ("Early strength tends to persist through a session (Gao, Han, Li & Zhou 2018)." if sc > 0 else
                  "Early weakness tends to persist through a session." if sc < 0 else
                  "Little net movement, so no momentum to lean on."), "Session"))
    hr = h[h.index <= as_of - timedelta(minutes=60)]
    if mins >= 75 and not hr.empty:
        lh = pct(last, float(hr["Close"].iloc[-1]))
        sc = 1 if lh > 0.15 else -1 if lh < -0.15 else 0
        S.append(sig("lasthr", "Last hour", sc, 0.75,
                     f"{sym} {lh:+.2f}% over the past hour. " +
                     ("Buyers are in control right now." if sc > 0 else "Sellers are in control right now." if sc < 0
                      else "The last hour went nowhere."), "Session"))
    if name == "NY" and as_of.time() >= OPEN_MOVE_END:
        orb = h[h.index < datetime.combine(start.date(), OPEN_MOVE_END, ET)]
        if len(orb) >= 5:
            hi, lo = float(orb["High"].max()), float(orb["Low"].min())
            sc = 1 if last > hi else -1 if last < lo else 0
            S.append(sig("orange", "Opening range", sc, 1.25,
                         f"9:30–10:00 range {lo:,.2f}–{hi:,.2f}, price {last:,.2f}. " +
                         ("Broke above the opening range: range breaks tend to run." if sc > 0 else
                          "Broke below the opening range: range breaks tend to run." if sc < 0 else
                          "Still inside the opening range, so expect rotation until it breaks."), "Session"))
    return S


def dealer_signals(cfg, pos, q):
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
        pc = f["put"] / f["call"] if f["call"] else 9.9
        sc = 1 if pc < 0.8 else -1 if pc > 1.25 else 0
        S.append(sig("flow", "Options flow (prior session)", sc, 0.25,
                     f"Short-dated premium put/call {pc:.2f}.", "Dealer positioning"))
    # Dealer positioning votes at its own weights; the Record retunes it like every other signal.
    return S


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


def narrate(cfg, sym, pos, signals, events, earnings, as_of, live, peer, view="preopen", sess=None):
    own = own_score(signals)
    score = own if not peer else (1 - PEER_WEIGHT) * own + PEER_WEIGHT * peer[1]
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
    if mega:
        score *= EARNINGS_DAMP
        damp.append("mega-cap earnings")
    bias = label_of(score)
    a = abs(score)
    lvl = 0 if a < NEUTRAL_BAND else 1 if a < 0.30 else 2 if a < 0.50 else 3
    if upcoming:
        lvl = min(lvl, 1)            # never better than Low into a binary print
    conviction = ["Very low", "Low", "Medium", "High"][lvl]
    directional = [s for s in signals if s.get("directional", True)]
    ups = [s for s in directional if s["score"] > 0]
    downs = [s for s in directional if s["score"] < 0]
    top = lambda L: sorted(L, key=lambda s: s["weight"], reverse=True)  # noqa: E731

    if view == "overnight":
        parts = [f"{bias}, {conviction.lower()}-conviction read on {sym} for the whole Asia and London move "
                 "(6 PM–9:30 AM ET), from macro data up to the 6 PM cutoff."]
    elif view == "now" and sess:
        parts = [f"{bias}, {conviction.lower()}-conviction read on {sym} for the rest of the {sess[0]} session "
                 f"({as_of:%-I:%M %p} → {sess[2]:%-I:%M %p} ET), from data up to {as_of:%-I:%M %p} ET."]
    elif view == "now":
        parts = [f"{bias}, {conviction.lower()}-conviction read on {sym} from data up to {as_of:%-I:%M %p} ET. Futures are closed."]
    else:
        parts = [f"{bias}, {conviction.lower()}-conviction read on {sym} for the New York open (09:30–10:00 ET)."]
    if bias != "Neutral":
        lead, against = (ups, downs) if bias == "Bullish" else (downs, ups)
        if lead:
            parts.append("The case rests on " + ", ".join(s["label"] for s in top(lead)[:3]) + ".")
        else:
            parts.append("The signals are dead even, so the call follows the market's long-run upward drift.")
        if abs(score) < NEUTRAL_BAND:
            parts.append("The two sides nearly cancel out, so wait for price to confirm before leaning on it.")
        if against:
            parts.append("Working against it: " + ", ".join(s["label"] for s in top(against)[:2]) + ".")
    else:
        lean = "up" if raw_score > 0.05 else "down" if raw_score < -0.05 else None
        parts.append(f"{len(ups)} signal{'' if len(ups)==1 else 's'} lean{'s' if len(ups)==1 else ''} up and {len(downs)} lean down" +
                     (f", a slight tilt {lean} ({raw_score:+.2f}) that isn't strong enough to call" if lean else "") +
                     ". No edge, so trade the levels and not a story.")
    if damp:
        parts.append(f"Score shrunk {raw_score:+.2f} → {score:+.2f} for " + " and ".join(damp) +
                     ": event days are driven by positioning into the release more than by the data before it.")
    if pos:
        parts.append(f"Dealers are net {'long' if pos['net_gex'] > 0 else 'short'} gamma, so expect " +
                     ("contained, mean-reverting trade." if pos["net_gex"] > 0 else "wider ranges and follow-through on breaks."))
    if peer:
        if label_of(own) != label_of(peer[1]):
            parts.append(f"On its own {sym} reads {label_of(own).lower()} ({own:+.2f}) and {peer[0]} "
                         f"{label_of(peer[1]).lower()} ({peer[1]:+.2f}). They move together, so the split pulls this toward neutral.")
        else:
            parts.append(f"{peer[0]} agrees ({peer[1]:+.2f}).")
    if upcoming:
        parts.append("Scheduled risk: " + "; ".join(f"{e['title']} at {e['time']}" for e in upcoming) +
                     ". A binary print can override all of this, so reassess after the release.")
    mega = [e for e in earnings if e["symbol"] in cfg["megacaps"]]
    if mega:
        parts.append("Mega-cap earnings: " + ", ".join(f"{m['symbol']} ({m['when']})" for m in mega) + ".")

    invalid = None
    if pos and pos["flip"]:
        f, above = pos["flip"], pos["px"] >= pos["flip"]
        invalid = {
            "Bullish": f"A 15-minute close below the gamma flip at {f:,.0f}" if above
            else f"Failure to reclaim {f:,.0f}, or a break of the put wall at {pos['put_wall']:,.0f}",
            "Bearish": f"A reclaim and hold above the gamma flip at {f:,.0f}" if not above
            else f"Holding above {f:,.0f}, or a push through the call wall at {pos['call_wall']:,.0f}",
            "Neutral": f"A decisive break of {pos['put_wall']:,.0f} or {pos['call_wall']:,.0f} sets direction",
        }[bias]
    elif view == "overnight":
        invalid = "Futures reversing through today's 4 PM close during the Asia session"
    elif view == "now" and sess:
        invalid = f"Price reversing through where the {sess[0]} session opened"
    elif cfg["future"] in ("NQ=F", "ES=F"):
        invalid = "Overnight futures reversing through yesterday's 4 PM level after the open"

    macro_w = sum(s["weight"] for s in signals if s["group"] != "Dealer positioning")
    return dict(bias=bias, score=round(score, 3), raw_score=round(raw_score, 3), dampened=damp,
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
    lead = "tech is weaker than the broad market" if n < e else "tech is leading the broad market"
    return (f"NQ reads {label_of(n)} ({n:+.2f}) and ES reads {label_of(e)} ({e:+.2f}) on their own evidence. "
            f"They are highly correlated, so both final reads are blended toward each other. The split says {lead}. "
            "Treat it as a lower-conviction day, size down, or wait for them to line up after the open.")


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
def build_view(target, as_of, live, now, view, hist, sess=None):
    """One complete read: view = 'preopen' (9:25 ET cut), 'now' (latest) or 'overnight' (6 PM cut)."""
    NOTES.clear(); SOURCES.clear()
    print(f"[{view}] session {target} · evidence cut at {as_of:%Y-%m-%d %H:%M} ET", file=sys.stderr)
    marks = None
    if view == "overnight":
        marks = {"close": datetime.combine(target, time(16, 0), ET),
                 "lasthour": datetime.combine(target, time(15, 0), ET)}
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
            # "since yesterday's 4 PM" is stale mid-session: the session's own tape replaces it
            base = [x for x in macro_signals(cfg, quotes) if x["key"] != "overnight"] + session_signals(cfg, sess, as_of)
        else:
            base = macro_signals(cfg, quotes)
        extra = [x for x in (earnings_signal(cfg, earnings, as_of), news_signal(as_of),
                             fed_pressure_signal(as_of), surprise_signal(events, as_of)) if x]
        base = base + extra
        # learn only from sessions graded before this one, so every read stays point-in-time
        key = as_of.strftime("%Y-%m-%dT%H%M") if view == "now" else target.isoformat()
        mult = learned_multipliers(hist, sym, key, 1000 if view == "now" else 120)
        pre[sym] = (pos, apply_learning(base, mult) + (dealer_signals(cfg, pos, quotes) if pos else []))

    if view == "overnight":
        hours = 6.5          # ~15h of Globex carries roughly one cash session of variance
    elif view == "now" and sess:
        hours = (sess[2] - as_of).total_seconds() / 3600
    else:
        hours = 0.5
    markets = {}
    for sym, cfg in MARKETS.items():
        pos, sigs = pre[sym]
        peers = [k for k in pre if k != sym and pre[k][1]]
        peer = (peers[0], own_score(pre[peers[0]][1])) if peers and sigs else None
        bias = narrate(cfg, sym, pos, sigs, events, earnings, as_of, live, peer, view, sess)
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

    report = {"generated": now.isoformat(timespec="seconds"),
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
