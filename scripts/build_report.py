#!/usr/bin/env python3
"""
Pre-Market Bias Report  ·  builds docs/data/latest.json every morning.

Free data only:
  • CBOE delayed option chains (greeks, IV, open interest, volume)  — cdn.cboe.com
  • Yahoo Finance via yfinance (futures, VIX family, 10Y yield, dollar, crude)
  • ForexFactory weekly calendar JSON (US economic releases)
  • Nasdaq earnings calendar (mega-cap reporters)

Dealer-positioning convention (standard in the GEX literature): customers are
net sellers of calls and net buyers of puts, so dealers are LONG calls and
SHORT puts. That is an assumption from open interest, not observed inventory.

Every source is wrapped so one outage never kills the report; failed pieces
are listed in the "notes" field and simply drop out of the score.
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
FIXTURES = os.environ.get("REPORT_FIXTURES")  # offline test mode
UA = {"User-Agent": "Mozilla/5.0 (premarket-report; personal research)"}
RISK_FREE = 0.04
NOTES: list[str] = []

MARKETS = {
    "NQ": {
        "name": "Nasdaq-100 futures",
        "future": "NQ=F",
        "chains": ["_NDX", "QQQ"],          # combined, mapped into NQ points
        "vol": "^VXN", "vol_label": "VXN",
        "bin": 25,                          # level bucket size in NQ points
        "rates_weight": 1.0,                # long-duration index: rates matter more
        "megacaps": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
                     "TSLA", "COST", "NFLX", "AMD"],
    },
    "ES": {
        "name": "S&P 500 futures",
        "future": "ES=F",
        "chains": ["_SPX", "SPY"],
        "vol": "^VIX", "vol_label": "VIX",
        "bin": 5,
        "rates_weight": 0.75,
        "megacaps": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO",
                     "TSLA", "BRK.B", "JPM", "LLY", "V", "XOM", "UNH"],
    },
}


def note(msg: str) -> None:
    print("NOTE:", msg, file=sys.stderr)
    NOTES.append(msg)


# ───────────────────────────── data fetchers ──────────────────────────────
def _get_json(url: str, fixture: str):
    if FIXTURES:
        p = Path(FIXTURES) / fixture
        return json.loads(p.read_text()) if p.exists() else None
    r = requests.get(url, headers=UA, timeout=45)
    r.raise_for_status()
    return r.json()


def fetch_chain(symbol: str):
    url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
    js = _get_json(url, f"cboe_{symbol}.json")
    if not js:
        raise RuntimeError("empty chain")
    d = js["data"]
    spot = float(d.get("current_price") or d.get("close"))
    prev = float(d.get("prev_day_close") or d.get("close") or spot)
    return spot, prev, d["options"]


def fetch_quotes(tickers: list[str]) -> dict:
    """Return {ticker: {"last": float, "prev": float}} from Yahoo."""
    if FIXTURES:
        p = Path(FIXTURES) / "quotes.json"
        return json.loads(p.read_text()) if p.exists() else {}
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        note("yfinance not installed: futures, vol and macro quotes skipped")
        return {}
    out = {}
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            last, prev = float(fi["lastPrice"]), float(fi["previousClose"])
            if math.isfinite(last) and math.isfinite(prev) and prev:
                out[t] = {"last": last, "prev": prev}
        except Exception as e:  # noqa: BLE001
            note(f"Quote unavailable for {t} ({type(e).__name__})")
    return out


def fetch_calendar(today: date) -> list[dict]:
    try:
        js = _get_json("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                       "calendar.json") or []
    except Exception as e:  # noqa: BLE001
        note(f"Economic calendar unavailable ({type(e).__name__})")
        return []
    events = []
    for ev in js:
        if ev.get("country") != "USD" or ev.get("impact") not in ("High", "Medium"):
            continue
        try:
            when = datetime.fromisoformat(ev["date"]).astimezone(ET)
        except Exception:  # noqa: BLE001
            continue
        if when.date() != today:
            continue
        events.append({
            "time": when.strftime("%-I:%M %p") if when.time() != time(0, 0) else "All day",
            "ts": when.isoformat(),
            "title": ev.get("title", ""),
            "impact": ev.get("impact"),
            "forecast": ev.get("forecast") or "",
            "previous": ev.get("previous") or "",
        })
    events.sort(key=lambda e: e["ts"])
    return events


def fetch_earnings(day: date) -> list[dict]:
    try:
        js = _get_json(f"https://api.nasdaq.com/api/calendar/earnings?date={day:%Y-%m-%d}",
                       f"earnings_{day:%Y-%m-%d}.json")
        rows = ((js or {}).get("data") or {}).get("rows") or []
    except Exception as e:  # noqa: BLE001
        note(f"Earnings calendar unavailable for {day} ({type(e).__name__})")
        return []
    out = []
    for r in rows:
        t = (r.get("time") or "").lower()
        when = "before open" if "pre" in t else "after close" if "after" in t else "time n/a"
        out.append({"symbol": r.get("symbol", "").upper(), "when": when,
                    "eps_est": r.get("epsForecast") or ""})
    return out


# ─────────────────────────── option math helpers ──────────────────────────
OCC = re.compile(r"^([A-Z]+?)(\d{6})([CP])(\d{8})$")
AM_SETTLED = {"SPX", "NDX"}  # monthly roots; SPXW / NDXP are PM-settled


def parse_chain(options: list[dict], now: datetime, max_days: int = 60):
    rows = []
    for o in options:
        m = OCC.match(o.get("option", ""))
        if not m:
            continue
        root, ymd, cp, k = m.groups()
        exp_d = datetime.strptime(ymd, "%y%m%d").date()
        exp_t = time(9, 30) if root in AM_SETTLED else time(16, 0)
        exp = datetime.combine(exp_d, exp_t, ET)
        days = (exp - now).total_seconds() / 86400
        if days <= 0 or days > max_days:
            continue
        oi = float(o.get("open_interest") or 0)
        vol = float(o.get("volume") or 0)
        if oi <= 0 and vol <= 0:
            continue
        bid, ask = float(o.get("bid") or 0), float(o.get("ask") or 0)
        mid = (bid + ask) / 2 if ask > 0 else float(o.get("last_trade_price") or 0)
        rows.append((float(k) / 1000, days / 365.0, float(o.get("iv") or 0),
                     oi, vol, 1 if cp == "C" else -1, mid, exp_d))
    if not rows:
        raise RuntimeError("no usable contracts")
    K, T, IV, OI, VOL, CP, MID, EXP = map(np.array, zip(*rows))
    # repair missing IVs with the median of valid ones
    good = (IV > 0.01) & (IV < 3)
    IV = np.where(good, IV, np.median(IV[good]) if good.any() else 0.2)
    return dict(K=K, T=T, IV=IV, OI=OI, VOL=VOL, CP=CP, MID=MID, EXP=EXP)


def bs_d1(S, K, T, iv):
    T = np.maximum(T, 1e-6)
    return (np.log(S / K) + (RISK_FREE + 0.5 * iv**2) * T) / (iv * np.sqrt(T))


def bs_gamma(S, K, T, iv):
    T = np.maximum(T, 1e-6)
    return norm.pdf(bs_d1(S, K, T, iv)) / (S * iv * np.sqrt(T))


def bs_delta(S, K, T, iv, cp):
    nd = norm.cdf(bs_d1(S, K, T, iv))
    return np.where(cp > 0, nd, nd - 1)


def dealer_gex(S, c, T=None, IV=None):
    """Dealer $ gamma per 1% move at price S (array-broadcast over S)."""
    T = c["T"] if T is None else T
    IV = c["IV"] if IV is None else IV
    S = np.asarray(S, dtype=float)[..., None]
    g = bs_gamma(S, c["K"], T, IV)
    return (c["CP"] * g * c["OI"] * 100 * S**2 * 0.01).sum(-1)


def dealer_delta(S, c, T=None, IV=None):
    """Dealer option delta in $ notional (long calls, short puts)."""
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


# ───────────────────────────── analysis core ──────────────────────────────
def analyse_positioning(cfg, quotes, now):
    fut = quotes.get(cfg["future"])
    chains = []
    for sym in cfg["chains"]:
        try:
            spot, prev, opts = fetch_chain(sym)
            c = parse_chain(opts, now)
            # map underlying → futures using prior-close basis (captures carry)
            ratio = (fut["prev"] / prev) if fut else 1.0
            c.update(sym=sym.lstrip("_"), ratio=ratio, spot=spot, prev=prev)
            chains.append(c)
        except Exception as e:  # noqa: BLE001
            note(f"{sym} option chain unavailable ({type(e).__name__}: {e})")
    if not chains:
        return None
    px = fut["last"] if fut else chains[0]["prev"] * chains[0]["ratio"]

    close_dt = datetime.combine(now.date(), time(16, 0), ET)
    hrs_to_close = max((close_dt - now).total_seconds() / 3600, 0)
    dT = hrs_to_close / (24 * 365)

    grid = np.linspace(px * 0.94, px * 1.06, 361)
    gex_now = np.zeros_like(grid)
    gex_close = np.zeros_like(grid)
    tot = dict(call_g=0.0, put_g=0.0, D=0.0, gross=0.0, D_close=0.0, vanna=0.0)
    levels: dict[float, list[float]] = {}
    flow = dict(call=0.0, put=0.0)
    strikes_flow: dict[tuple, float] = {}

    for c in chains:
        r = c["ratio"]
        S = px / r                                  # futures-equivalent underlying
        gex_now += dealer_gex(grid / r, c)
        alive = c["T"] - dT > 1e-6                  # contracts still alive at the close
        cc = {k: (v[alive] if isinstance(v, np.ndarray) else v) for k, v in c.items()}
        gex_close += dealer_gex(grid / r, cc, T=cc["T"] - dT)

        g = bs_gamma(S, c["K"], c["T"], c["IV"]) * c["OI"] * 100 * S**2 * 0.01
        tot["call_g"] += float(g[c["CP"] > 0].sum())
        tot["put_g"] += float(g[c["CP"] < 0].sum())
        for k, gg, cp in zip(c["K"] * r, g, c["CP"]):
            b = round(k / cfg["bin"]) * cfg["bin"]
            levels.setdefault(b, [0.0, 0.0])[0 if cp > 0 else 1] += gg

        D, gross = dealer_delta(S, c)
        Dc, _ = dealer_delta(S, c, T=np.maximum(c["T"] - dT, 1e-7))
        Du, _ = dealer_delta(S, c, IV=c["IV"] + 0.005)
        Dd, _ = dealer_delta(S, c, IV=np.maximum(c["IV"] - 0.005, 0.01))
        tot["D"] += D; tot["gross"] += gross; tot["D_close"] += Dc
        tot["vanna"] += (Du - Dd)                  # $ delta change per +1 vol point

        # prior-session short-dated flow (≤7 days) by premium
        short = c["T"] * 365 <= 7
        prem = c["VOL"] * c["MID"] * 100
        flow["call"] += float(prem[short & (c["CP"] > 0)].sum())
        flow["put"] += float(prem[short & (c["CP"] < 0)].sum())
        for k, p, cp, s in zip(c["K"] * r, prem, c["CP"], short):
            if s and p > 0:
                key = (round(k / cfg["bin"]) * cfg["bin"], "call" if cp > 0 else "put")
                strikes_flow[key] = strikes_flow.get(key, 0) + p

    net_gex = float(np.interp(px, grid, gex_now))
    flip = find_flip(grid, gex_now, px)
    flip_close = find_flip(grid, gex_close, px) if hrs_to_close > 0.5 else None

    near = {k: v for k, v in levels.items() if abs(k / px - 1) < 0.06}
    above = {k: v for k, v in near.items() if k >= px}
    below = {k: v for k, v in near.items() if k <= px}
    call_wall = max(above or near, key=lambda k: near[k][0])
    put_wall = max(below or near, key=lambda k: near[k][1])
    majors = sorted(near, key=lambda k: abs(near[k][0] - near[k][1]), reverse=True)[:5]

    top_flow = sorted(strikes_flow.items(), key=lambda kv: kv[1], reverse=True)[:5]

    return dict(
        px=px, chains=[c["sym"] for c in chains], hrs_to_close=hrs_to_close,
        net_gex=net_gex, flip=flip, flip_close=flip_close,
        call_gamma=tot["call_g"], put_gamma=tot["put_g"],
        call_wall=call_wall, put_wall=put_wall,
        majors=[{"level": k, "net_gex": near[k][0] - near[k][1]} for k in sorted(majors)],
        dealer_delta=tot["D"], gross_delta=tot["gross"],
        charm_hedge_flow=-(tot["D_close"] - tot["D"]),   # + = dealers BUY into close
        vanna_per_volpt=tot["vanna"],
        flow=flow,
        top_flow=[{"level": k[0], "side": k[1], "premium": v} for k, v in top_flow],
        profile=[{"px": round(float(x), 2), "gex": float(y)} for x, y in
                 zip(grid[::6], gex_now[::6])],
    )


def pct(a, b):
    return (a / b - 1) * 100


def sig(key, label, score, weight, text, group):
    return {"key": key, "label": label, "score": int(np.sign(score)) if score else 0,
            "raw": score, "weight": weight, "text": text, "group": group}


def build_signals(cfg, pos, q, events, now):
    S = []
    vl = cfg["vol_label"]

    # ── Dealer positioning ────────────────────────────────────────────────
    if pos:
        px = pos["px"]
        if pos["flip"]:
            d = px - pos["flip"]
            band = px * 0.0015
            sc = 1 if d > band else -1 if d < -band else 0
            where = "above" if d > 0 else "below"
            why = ("Above the flip dealers are long gamma, so their hedging buys dips and "
                   "sells rips. That dampens volatility and has historically skewed returns up."
                   if d > 0 else
                   "Below the flip dealers are short gamma, so their hedging sells into "
                   "weakness and buys into strength. Moves extend, and drawdowns get sharper.")
            S.append(sig("flip", "Gamma flip", sc, 1.5,
                         f"Price is {abs(d):,.2f} pts {where} the gamma flip ({pos['flip']:,.2f}). {why}",
                         "Dealer positioning"))
        else:
            sc = 0.5 if pos["net_gex"] > 0 else -0.5
            S.append(sig("flip", "Gamma regime", sc, 1.0,
                         "No flip inside ±6%. Dealers are "
                         + ("long gamma across the whole range, a stabilising backdrop."
                            if sc > 0 else "short gamma across the whole range. Expect trend days."),
                         "Dealer positioning"))

        tilt = pos["call_gamma"] / pos["put_gamma"] if pos["put_gamma"] else 1
        sc = 1 if tilt > 1.25 else -1 if tilt < 0.8 else 0
        S.append(sig("tilt", "Gamma tilt", sc, 0.75,
                     f"Call ÷ put gamma is {tilt:.2f}. " +
                     ("Call-heavy positioning leaves upside supported." if sc > 0 else
                      "Put-heavy positioning: hedges sit below, and dealers sell as price falls." if sc < 0
                      else "Roughly balanced, so no edge here."),
                     "Dealer positioning"))

        up, dn = pos["call_wall"] - px, px - pos["put_wall"]
        span = max(up + dn, 1e-9)
        loc = dn / span
        sc = 1 if loc < 0.33 else -1 if loc > 0.67 else 0
        S.append(sig("walls", "Wall range", sc, 1.0,
                     f"Put wall {pos['put_wall']:,.0f} to call wall {pos['call_wall']:,.0f}: "
                     f"{up:,.0f} pts of room up vs {dn:,.0f} down. " +
                     ("Sitting near support, so the asymmetry favours a bounce." if sc > 0 else
                      "Pressed against the call wall, where dealer supply tends to cap rallies." if sc < 0
                      else "Mid-range. The walls are targets, not a lean."),
                     "Dealer positioning"))

        if pos["flip"] and pos["flip_close"]:
            drift = pos["flip_close"] - pos["flip"]
            sc = 1 if drift < -px * 0.0007 else -1 if drift > px * 0.0007 else 0
            S.append(sig("flip_close", "Flip into close", sc, 0.75,
                         f"As today's options decay, the flip drifts {abs(drift):,.2f} pts "
                         f"{'down' if drift < 0 else 'up'} by 4 PM. " +
                         ("That widens the cushion under price, which is supportive." if sc > 0 else
                          "That pulls the flip toward price, which is a headwind." if sc < 0
                          else "Too small to matter."),
                         "Dealer positioning"))

        thr = 0.0015 * pos["gross_delta"]
        ch = pos["charm_hedge_flow"]
        sc = 1 if ch > thr else -1 if ch < -thr else 0
        S.append(sig("charm", "Charm (theta → delta)", sc, 0.75,
                     f"Time decay alone shifts dealer hedges by ${abs(ch)/1e9:,.2f}B into the close. " +
                     ("Dealers will need to BUY futures back as delta bleeds off. That is a "
                      "steady afternoon bid, the effect Baltussen et al. tie to late-day momentum."
                      if sc > 0 else
                      "Dealers will need to SELL futures as delta bleeds off, a steady afternoon offer."
                      if sc < 0 else "Small relative to positioning."),
                     "Dealer positioning"))

        vchg = None
        vq = q.get(cfg["vol"])
        if vq:
            vchg = vq["last"] - vq["prev"]
            flow_v = -pos["vanna_per_volpt"] * vchg
            sc = 1 if flow_v > thr else -1 if flow_v < -thr else 0
            S.append(sig("vanna", "Vanna (vega → delta)", sc, 1.0,
                         f"{vl} {'down' if vchg < 0 else 'up'} {abs(vchg):.2f} pts overnight. "
                         f"Through vanna that is about ${abs(flow_v)/1e9:,.2f}B of dealer "
                         f"{'buying' if flow_v > 0 else 'selling'}. " +
                         ("Falling implied vol forces hedge buy-backs, which is supportive." if sc > 0 else
                          "Rising implied vol forces hedge selling, which is pressure." if sc < 0
                          else "Too small to move the needle."),
                         "Dealer positioning"))

        f = pos["flow"]
        if f["call"] + f["put"] > 0:
            pc = f["put"] / f["call"] if f["call"] else 9.9
            sc = 1 if pc < 0.8 else -1 if pc > 1.25 else 0
            S.append(sig("flow", "Options flow (prior session)", sc, 0.5,
                         f"Short-dated premium: calls ${f['call']/1e6:,.1f}M vs puts ${f['put']/1e6:,.1f}M "
                         f"(put/call {pc:.2f}). " +
                         ("Call-led, risk appetite." if sc > 0 else
                          "Put-led demand. Pan & Poteshman find heavy put buying precedes weakness."
                          if sc < 0 else "Balanced."),
                         "Options flow"))

    # ── Volatility ───────────────────────────────────────────────────────
    vq = q.get(cfg["vol"])
    if vq:
        ch = pct(vq["last"], vq["prev"])
        sc = 1 if ch < -3 else -1 if ch > 3 else 0
        S.append(sig("volchg", f"{vl} overnight", sc, 1.0,
                     f"{vl} {vq['last']:.2f} ({ch:+.1f}%). " +
                     ("Fear bleeding out. Vol sellers are back in control." if sc > 0 else
                      "Hedging demand is rising into the open." if sc < 0 else "Little changed."),
                     "Volatility"))
    v, v3 = q.get("^VIX"), q.get("^VIX3M")
    if v and v3:
        ts = v["last"] / v3["last"]
        sc = -1 if ts > 1.0 else 1 if ts < 0.9 else 0
        S.append(sig("term", "Vol term structure", sc, 1.0,
                     f"VIX ÷ VIX3M = {ts:.2f}. " +
                     ("Backwardation: near-term fear exceeds longer-term. Rallies tend to get sold "
                      "until this normalises." if sc < 0 else
                      "Healthy contango, the backdrop where dips get bought." if sc > 0
                      else "Flat-ish curve, a mild caution flag."),
                     "Volatility"))

    # ── Macro ────────────────────────────────────────────────────────────
    fq = q.get(cfg["future"])
    if fq:
        ch = pct(fq["last"], fq["prev"])
        sc = 1 if ch > 0.25 else -1 if ch < -0.25 else 0
        S.append(sig("overnight", "Overnight session", sc, 1.0,
                     f"Futures {ch:+.2f}% vs settlement ({fq['last']:,.2f}). " +
                     ("Overseas buyers did the lifting." if sc > 0 else
                      "Selling came in overnight." if sc < 0 else "A quiet Globex session."),
                     "Macro"))
    tq = q.get("^TNX")
    if tq:
        bp = (tq["last"] - tq["prev"]) * 100 if tq["last"] < 20 else (tq["last"] - tq["prev"]) * 10
        sc = -1 if bp > 4 else 1 if bp < -4 else 0
        S.append(sig("rates", "10-year yield", sc, cfg["rates_weight"],
                     f"10Y at {tq['last'] if tq['last'] < 20 else tq['last']/10:.2f}% ({bp:+.1f} bp). " +
                     ("Higher discount rates squeeze long-duration growth multiples." if sc < 0 else
                      "Falling yields lift equity valuations, tech most of all." if sc > 0
                      else "Rates are not in play this morning."),
                     "Macro"))
    dq = q.get("DX-Y.NYB")
    if dq:
        ch = pct(dq["last"], dq["prev"])
        sc = -1 if ch > 0.25 else 1 if ch < -0.25 else 0
        S.append(sig("dollar", "US dollar", sc, 0.5,
                     f"DXY {dq['last']:.2f} ({ch:+.2f}%). " +
                     ("A firmer dollar tightens global financial conditions and dents overseas earnings."
                      if sc < 0 else "A softer dollar eases financial conditions." if sc > 0
                      else "Flat."),
                     "Macro"))
    cq = q.get("CL=F")
    if cq:
        ch = pct(cq["last"], cq["prev"])
        sc = -1 if ch > 2.5 else 0
        S.append(sig("oil", "Crude oil", sc, 0.5,
                     f"WTI {cq['last']:.2f} ({ch:+.1f}%). " +
                     ("An oil spike is an inflation tax and a growth drag." if sc < 0
                      else "No energy shock in play."),
                     "Macro"))
    return S


def score_and_narrate(cfg, sym, pos, signals, events, earnings, now):
    tot_w = sum(s["weight"] for s in signals) or 1
    raw = sum(s["raw"] * s["weight"] for s in signals)
    norm_score = max(-1.0, min(1.0, raw / tot_w))
    bias = "Bullish" if norm_score > 0.12 else "Bearish" if norm_score < -0.12 else "Neutral"

    a = abs(norm_score)
    lvl = 0 if a < 0.25 else 1 if a < 0.45 else 2
    upcoming_high = [e for e in events if e["impact"] == "High" and e["ts"] > now.isoformat()]
    mega = [e for e in earnings if e["symbol"] in cfg["megacaps"]]
    if upcoming_high:
        lvl = max(lvl - 1, 0)
    conviction = ["Low", "Medium", "High"][lvl]

    ups = [s for s in signals if s["score"] > 0]
    downs = [s for s in signals if s["score"] < 0]
    top = lambda L: sorted(L, key=lambda s: s["weight"], reverse=True)  # noqa: E731

    parts = [f"{bias}, {conviction.lower()}-conviction read on {sym} for the New York morning "
             f"(09:30–12:00 ET)."]
    if bias != "Neutral":
        lead, against = (ups, downs) if bias == "Bullish" else (downs, ups)
        parts.append("The case rests on " + ", ".join(s["label"] for s in top(lead)[:3]) + ".")
        if against:
            parts.append("Working against it: " + ", ".join(s["label"] for s in top(against)[:2]) + ".")
    else:
        parts.append(f"{len(ups)} signals lean up and {len(downs)} lean down. That is a tape with "
                     "no edge, so trade the levels and not a story.")
    if pos:
        regime = "long" if pos["net_gex"] > 0 else "short"
        parts.append(f"Dealers are net {regime} gamma at {pos['px']:,.0f}, so expect "
                     + ("mean-reverting, range-bound trade where fades work."
                        if regime == "long" else "wider ranges and follow-through on breaks."))
    if upcoming_high:
        parts.append("Scheduled risk: " + "; ".join(f"{e['title']} at {e['time']}" for e in upcoming_high)
                     + ". Positioning before a binary print is a coin flip. Reassess after the release.")
    if mega:
        parts.append("Mega-cap earnings in play: " + ", ".join(f"{m['symbol']} ({m['when']})" for m in mega) + ".")

    invalid = None
    if pos and pos["flip"]:
        f, above = pos["flip"], pos["px"] >= pos["flip"]
        if bias == "Bullish":
            invalid = (f"A 15-minute close back below the gamma flip at {f:,.0f}" if above else
                       f"Failure to reclaim the gamma flip at {f:,.0f}, or a break of the put wall at {pos['put_wall']:,.0f}")
        elif bias == "Bearish":
            invalid = (f"A reclaim and hold above the gamma flip at {f:,.0f}" if not above else
                       f"Holding above the gamma flip at {f:,.0f}, or a push through the call wall at {pos['call_wall']:,.0f}")
        else:
            invalid = f"A decisive break of {pos['put_wall']:,.0f} or {pos['call_wall']:,.0f} sets the direction"

    return dict(bias=bias, score=round(norm_score, 3), conviction=conviction,
                ups=len(ups), downs=len(downs), total=len(signals),
                overall=" ".join(parts), invalidation=invalid)


def expected_range(cfg, pos, q):
    vq = q.get(cfg["vol"])
    if not (pos and vq):
        return None
    sigma_day = vq["last"] / 100 / math.sqrt(252)
    adj = 0.85 if pos["net_gex"] > 0 else 1.15   # long gamma dampens realized vol
    move = pos["px"] * sigma_day * adj
    return {"move": round(move, 2), "low": round(pos["px"] - move, 2),
            "high": round(pos["px"] + move, 2), "basis": f"{cfg['vol_label']} implied, gamma-adjusted"}


# ─────────────────────────── scorecard (history) ──────────────────────────
def grade_history(hist_dir: Path, today: date):
    """Fill in the 09:30→12:00 outcome for past reports that lack one."""
    if FIXTURES:
        return
    try:
        import yfinance as yf
    except Exception:  # noqa: BLE001
        return
    for f in sorted(hist_dir.glob("20*.json"))[-15:]:
        d = date.fromisoformat(f.stem)
        if d >= today:
            continue
        rep = json.loads(f.read_text())
        changed = False
        for sym, m in rep.get("markets", {}).items():
            if m.get("outcome") or not m.get("bias"):
                continue
            try:
                h = yf.Ticker(MARKETS[sym]["future"]).history(
                    start=d, end=d + timedelta(days=1), interval="5m")
                h.index = h.index.tz_convert(ET)
                o = h.between_time("09:30", "09:34")["Open"].iloc[0]
                c = h.between_time("11:55", "11:59")["Close"].iloc[-1]
                mv = float(c - o)
                b = m["bias"]["bias"]
                hit = (mv > 0 and b == "Bullish") or (mv < 0 and b == "Bearish")
                m["outcome"] = {"open": float(o), "noon": float(c), "move": round(mv, 2),
                                "result": "n/a" if b == "Neutral" else ("hit" if hit else "miss")}
                changed = True
            except Exception:  # noqa: BLE001
                continue
        if changed:
            f.write_text(json.dumps(rep, indent=1))


def scorecard(hist_dir: Path, sym: str):
    hits = misses = 0
    recent = []
    for f in sorted(hist_dir.glob("20*.json"))[-20:]:
        m = json.loads(f.read_text()).get("markets", {}).get(sym, {})
        o = m.get("outcome")
        if not o:
            continue
        hits += o["result"] == "hit"
        misses += o["result"] == "miss"
        recent.append({"date": f.stem, "bias": m["bias"]["bias"], "move": o["move"], "result": o["result"]})
    return {"hits": hits, "misses": misses, "recent": recent[-10:]}


# ──────────────────────────────── main ────────────────────────────────────
def main():
    now = datetime.now(ET)
    if FIXTURES and os.environ.get("REPORT_NOW"):
        now = datetime.fromisoformat(os.environ["REPORT_NOW"]).astimezone(ET)
    today = now.date()
    hist = OUT / "history"
    hist.mkdir(parents=True, exist_ok=True)

    tickers = sorted({m["future"] for m in MARKETS.values()} | {m["vol"] for m in MARKETS.values()}
                     | {"^VIX", "^VIX3M", "^TNX", "DX-Y.NYB", "CL=F"})
    quotes = fetch_quotes(tickers)
    events = fetch_calendar(today)
    earnings = fetch_earnings(today)
    prev_day = today - timedelta(days=3 if today.weekday() == 0 else 1)
    earnings_prev = [e for e in fetch_earnings(prev_day) if e["when"] == "after close"]

    grade_history(hist, today)
    prior_files = [f for f in sorted(hist.glob("20*.json")) if f.stem < today.isoformat()]
    prior = json.loads(prior_files[-1].read_text()) if prior_files else None

    markets = {}
    for sym, cfg in MARKETS.items():
        try:
            pos = analyse_positioning(cfg, quotes, now)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            note(f"{sym} positioning failed ({type(e).__name__})")
            pos = None
        sigs = build_signals(cfg, pos, quotes, events, now)
        ern = earnings + [dict(e, when="last night") for e in earnings_prev]
        bias = score_and_narrate(cfg, sym, pos, sigs, events, ern, now)
        changed = None
        if prior and sym in prior.get("markets", {}):
            pb = prior["markets"][sym]["bias"]
            changed = {"prev_date": prior["date"], "prev_bias": pb["bias"],
                       "prev_score": pb["score"], "delta": round(bias["score"] - pb["score"], 3)}
        mega = [e for e in ern if e["symbol"] in cfg["megacaps"]]
        markets[sym] = {
            "name": cfg["name"], "bias": bias, "signals": sigs, "changed": changed,
            "expected_range": expected_range(cfg, pos, quotes),
            "levels": None if not pos else {
                "price": round(pos["px"], 2),
                "flip": pos["flip"] and round(pos["flip"], 2),
                "flip_close": pos["flip_close"] and round(pos["flip_close"], 2),
                "call_wall": pos["call_wall"], "put_wall": pos["put_wall"],
                "majors": pos["majors"],
            },
            "dealer": None if not pos else {
                "net_gex_bn": round(pos["net_gex"] / 1e9, 3),
                "dealer_delta_bn": round(pos["dealer_delta"] / 1e9, 2),
                "call_put_gamma": round(pos["call_gamma"] / pos["put_gamma"], 2) if pos["put_gamma"] else None,
                "charm_bn": round(pos["charm_hedge_flow"] / 1e9, 3),
                "vanna_bn_per_pt": round(pos["vanna_per_volpt"] / 1e9, 3),
                "chains": pos["chains"],
            },
            "flow": None if not pos else {
                "calls_m": round(pos["flow"]["call"] / 1e6, 1),
                "puts_m": round(pos["flow"]["put"] / 1e6, 1),
                "top": pos["top_flow"],
            },
            "profile": pos["profile"] if pos else [],
            "earnings": mega,
            "scorecard": scorecard(hist, sym),
        }

    report = {
        "date": today.isoformat(),
        "generated": now.isoformat(timespec="seconds"),
        "generated_label": now.strftime("%b %-d, %Y at %-I:%M %p ET"),
        "events": events,
        "macro": {k: {"last": round(v["last"], 3), "chg_pct": round(pct(v["last"], v["prev"]), 2)}
                  for k, v in quotes.items()},
        "markets": markets,
        "notes": NOTES,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "latest.json").write_text(json.dumps(report, indent=1, default=float))
    (hist / f"{today.isoformat()}.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({s: m["bias"] for s, m in markets.items()}, indent=1))


if __name__ == "__main__":
    main()
