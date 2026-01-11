#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Binomial / Black–Scholes option pricer using market data."""


from dataclasses import dataclass
from math import exp, sqrt, log, pi
from typing import Literal, Optional, List
import datetime as dt, sys 
import numpy as np

import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# yfinance (spot, div, sigma)
_HAVE_YF = False
try:
    import yfinance as yf
    _HAVE_YF = True
except Exception:
    pass

try:
    import matplotlib
    try:
        from IPython import get_ipython
        get_ipython().run_line_magic("matplotlib","widget")
    except Exception:
        # try Tk first, then Qt, otherwise fallback to PNG backend
        for be in ("TkAgg","Qt5Agg"):
            try:
                matplotlib.use(be, force=True)
                break
            except Exception:
                continue
        else:
            matplotlib.use("Agg", force=True)  # last resort fallback
    import matplotlib.pyplot as plt; _HAVE_MPL=True
except Exception:
    _HAVE_MPL=False

try:
    import mplcursors; _HAVE_MPLCURSORS=True
except Exception:
    _HAVE_MPLCURSORS=False

def print_missing_deps():
    missing = []
    if not _HAVE_YF: missing.append("yfinance")
    if not _HAVE_MPL: missing.append("matplotlib")
    if not _HAVE_MPLCURSORS: missing.append("mplcursors (tooltips)")
    if missing:
        print("Optional packages not found:", ", ".join(missing))
        print("   Install with:")
        print("   pip install " + " ".join(
            pkg if "tooltips" not in pkg else "mplcursors" for pkg in missing
        ))


# TYPES
OptionType = Literal["call","put"]

@dataclass
class OptionParams:
    S: float; K: float; T: float; r: float; q: float; sigma: float

# HELPERS 
def normalize_ratio(x):
    try: x=float(x)
    except: return None
    if x<=0: return None
    if x<=1.5: return x
    if x<=150: return x/100.0
    if x<=10000: return x/10000.0
    return None

def detect_market_region(ticker: str) -> str:
    EU = (".PA",".AS",".L",".DE",".F",".SW",".MI",".MC",".ST",".OL",".CO",
          ".HE",".BR",".VI",".WA",".LS",".IR",".AT",".BE",".VX",".DU")
    return "europe" if ticker.upper().endswith(EU) else "us"

CURRENCY_SYMBOLS={"USD":"$","EUR":"€","GBP":"£","CHF":"CHF","SEK":"SEK","NOK":"NOK",
                  "DKK":"DKK","CAD":"C$","JPY":"¥"}
def detect_currency(t:str):
    t=t.upper(); m={".PA":"EUR",".AS":"EUR",".L":"GBP",".DE":"EUR",".F":"EUR",".SW":"CHF",
                    ".MI":"EUR",".MC":"EUR",".ST":"SEK",".BR":"EUR",".HE":"EUR",".OL":"NOK",
                    ".CO":"DKK",".VX":"CHF",".TO":"CAD"}
    code=next((c for s,c in m.items() if t.endswith(s)),"USD")
    return code, CURRENCY_SYMBOLS.get(code,code)

def get_risk_free_rate(region: str, T: float) -> float:
    # very simple proxy curve (cont. rate returned)
    if region == "us":
        yS, yL = 0.045, 0.042
    else:  # "europe"
        yS, yL = 0.025, 0.030

    w = min(max(T / 5.0, 0.0), 1.0)
    return log(1.0 + max(-0.009, (1 - w) * yS + w * yL))

def robust_dividend_yield_yf(ticker:str,S:float):
    if not _HAVE_YF: return None
    try:
        tk = yf.Ticker(ticker)
        # Avoid tk.info (too fragile / deprecated in some cases)
        div = None
        try:
            div = tk.dividends
        except Exception:
            div = None
        if div is not None and len(div):
            cutoff = dt.datetime.now(dt.UTC) - dt.timedelta(days=365)
            last12 = div[div.index.tz_localize(None) >= cutoff]
            if len(last12) and S > 0:
                return float(last12.sum()/S)
        return None
    except Exception:
        return None
    
def _to_stooq_symbol(t: str) -> str:
    t = (t or "").strip().lower()
    if '.' in t:          # ex: air.pa, san.pa, vod.l
        return t
    return f"{t}.us"      # ex: aapl -> aapl.us

def fetch_market_params(ticker: str):
    """
    1) Yahoo (yfinance.download) with light retry and custom User-Agent.
    2) Fallback to Stooq (CSV) if Yahoo is empty.
    3) Manual fallback if everything fails.
    """
    # 1) Stooq first 
    try:
        import pandas as pd, requests, io
        sym = _to_stooq_symbol(ticker)                  # ex: "aapl.us", "air.pa"
        url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        txt = (r.text or "").strip()

        if r.ok and txt and "no data" not in txt.lower() and not txt.lstrip().lower().startswith("<html"):
            df = pd.read_csv(io.StringIO(txt))
            if "Close" in df.columns and not df["Close"].dropna().empty:
                S = float(df["Close"].dropna().iloc[-1])
                if len(df) >= 126:
                    px = df["Close"].astype(float)
                    rets = np.log(px).diff().dropna().values[-126:]
                    sigma = float(np.std(rets, ddof=0) * np.sqrt(252))
                else:
                    sigma = 0.25
                sigma = float(min(max(normalize_ratio(sigma) or sigma, 0.05), 1.50))
                q = 0.0
                print("[data] source = stooq")
                return float(S), float(q), float(sigma), "stooq"
    except Exception:
        pass

    #  2) Yahoo fallback 
    if _HAVE_YF:
        import time
        sess = None
        try:
            import requests as _req
            sess = _req.Session()
            sess.headers.update({"User-Agent": "Mozilla/5.0"})
        except Exception:
            pass

        for delay in (0, 1.5, 3.0):
            try:
                h = yf.download(tickers=ticker, period="5d", interval="1d",
                                auto_adjust=False, progress=False, session=sess)
                if h is not None and not h.empty and "Close" in h and not h["Close"].dropna().empty:
                    S = float(h["Close"].dropna().iloc[-1])

                    # build sigma from historical returns
                    hlong = yf.download(tickers=ticker, period="1y", interval="1d",
                                        auto_adjust=False, progress=False, session=sess)
                    if hlong is None or hlong.empty or "Close" not in hlong or hlong["Close"].dropna().size < 3:
                        hlong = yf.download(tickers=ticker, period="6mo", interval="1d",
                                            auto_adjust=False, progress=False, session=sess)

                    sigma = 0.25
                    if hlong is not None and not hlong.empty and "Close" in hlong and hlong["Close"].dropna().size > 2:
                        rets  = np.log(hlong["Close"]).diff().dropna().values
                        sigma = float(np.std(rets, ddof=0) * np.sqrt(252))
                    sigma = float(min(max(normalize_ratio(sigma) or sigma, 0.05), 1.50))

                    q = robust_dividend_yield_yf(ticker, S) or 0.0
                    print("[data] source = yahoo")
                    return float(S), float(q), float(sigma), "yahoo"
            except Exception:
                pass
            time.sleep(delay)

    #  3) Manual 
    print("[data] source = manual")
    return _manual_inputs()

def _manual_inputs():
    print("⚠️ Autoload failed. Manual inputs.")
    S = float(input("Spot S: ") or "100")
    q = float(input("Dividend yield q (cont.) [0.0]: ") or "0.0")
    sigma = float(input("Annual vol σ [0.25]: ") or "0.25")
    return S, q, sigma, "manual"

# normal CDF
try:
    from scipy.stats import norm; N=norm.cdf
except Exception:
    def _phi(x): return (1.0/(2.0*pi)**0.5)*exp(-0.5*x*x)
    def N(x):
        b0,b1,b2,b3,b4,b5=0.2316419,0.31938153,-0.356563782,1.781477937,-1.821255978,1.330274429
        t=1.0/(1.0+b0*abs(x)); poly=((((b5*t+b4)*t+b3)*t+b2)*t+b1)
        nd=1.0-_phi(abs(x))*poly*t
        return nd if x>=0 else 1.0-nd

# PRICING
def bs_price(kind:OptionType,p:OptionParams)->float:
    if p.T==0: return max(0.0,p.S-p.K) if kind=="call" else max(0.0,p.K-p.S)
    d1=(log(p.S/p.K)+(p.r-p.q+0.5*p.sigma**2)*p.T)/(p.sigma*sqrt(p.T)); d2=d1-p.sigma*sqrt(p.T)
    df_r,df_q=exp(-p.r*p.T),exp(-p.q*p.T)
    return df_q*p.S*N(d1)-df_r*p.K*N(d2) if kind=="call" else df_r*p.K*N(-d2)-df_q*p.S*N(-d1)

# --- IMPLIED VOL (ATM) calibration helpers -------------------------------

def _mid_price_from_chain_row(row):
    """Robust mid price from a yfinance option chain row."""
    try:
        bid = float(row.get("bid", np.nan))
        ask = float(row.get("ask", np.nan))
    except Exception:
        bid, ask = np.nan, np.nan

    if np.isfinite(bid) and np.isfinite(ask) and ask > 0 and bid >= 0:
        mid = 0.5 * (bid + ask)
        if mid > 0:
            return float(mid)

    # fallback: lastPrice
    try:
        lp = float(row.get("lastPrice", np.nan))
        if np.isfinite(lp) and lp > 0:
            return float(lp)
    except Exception:
        pass

    return None


def implied_vol_bs(kind: OptionType, price: float, p: OptionParams,
                   lo: float = 1e-6, hi: float = 5.0, it: int = 80) -> Optional[float]:
    """
    Bisection solve for European BS implied vol.
    Returns None if price is invalid or inversion fails.
    """
    try:
        price = float(price)
    except Exception:
        return None
    if not np.isfinite(price) or price <= 0:
        return None
    if p.T <= 0 or p.S <= 0 or p.K <= 0:
        return None

    # Basic no-arbitrage bounds (loose, just to avoid nonsense)
    df_r = exp(-p.r * p.T)
    df_q = exp(-p.q * p.T)
    if kind == "call":
        lower = max(0.0, df_q * p.S - df_r * p.K)
        upper = df_q * p.S
    else:
        lower = max(0.0, df_r * p.K - df_q * p.S)
        upper = df_r * p.K
    if price < lower - 1e-10 or price > upper + 1e-10:
        return None

    f_lo = bs_price(kind, OptionParams(p.S, p.K, p.T, p.r, p.q, lo)) - price
    f_hi = bs_price(kind, OptionParams(p.S, p.K, p.T, p.r, p.q, hi)) - price

    # If not bracketed, try expanding hi a bit
    if f_lo * f_hi > 0:
        for hi2 in (8.0, 12.0):
            f_hi = bs_price(kind, OptionParams(p.S, p.K, p.T, p.r, p.q, hi2)) - price
            if f_lo * f_hi <= 0:
                hi = hi2
                break
        else:
            return None

    a, b = lo, hi
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = bs_price(kind, OptionParams(p.S, p.K, p.T, p.r, p.q, m)) - price
        if fm == 0:
            return float(m)
        if f_lo * fm <= 0:
            b = m
            f_hi = fm
        else:
            a = m
            f_lo = fm
    return float(0.5 * (a + b))


def get_atm_iv_yf(ticker: str, S: float, T: float, r: float, q: float,
                  prefer_days_range=(20, 120)) -> Optional[float]:
    """
    ATM implied vol from yfinance option chain:
    - pick expiry closest to T (prefer window prefer_days_range)
    - pick CALL with strike closest to spot S
    - return impliedVolatility if available else invert from mid/last via BS
    """
    if not _HAVE_YF:
        return None
    try:
        tk = yf.Ticker(ticker)
        exps = list(getattr(tk, "options", []) or [])
        if not exps:
            return None

        today = dt.datetime.now(dt.UTC).date()
        target_days = int(round(T * 365))

        parsed = []
        for e in exps:
            try:
                ed = dt.datetime.strptime(e, "%Y-%m-%d").date()
                d = (ed - today).days
                if d <= 0:
                    continue
                parsed.append((abs(d - target_days), d, e))
            except Exception:
                continue
        if not parsed:
            return None

        lo_d, hi_d = prefer_days_range
        inside = [x for x in parsed if lo_d <= x[1] <= hi_d]
        _, d_days, expiry = min(inside or parsed, key=lambda t: t[0])
        T_use = d_days / 365.0

        chain = tk.option_chain(expiry)
        calls = getattr(chain, "calls", None)
        if calls is None or calls.empty or "strike" not in calls.columns:
            return None

        calls = calls.copy()
        calls["dist"] = (calls["strike"].astype(float) - float(S)).abs()
        row = calls.sort_values("dist").iloc[0]

        if "impliedVolatility" in calls.columns:
            try:
                iv = float(row["impliedVolatility"])
                if np.isfinite(iv) and iv > 0:
                    return float(np.clip(iv, 0.03, 1.50))
            except Exception:
                pass

        mid = _mid_price_from_chain_row(row)
        if mid is None:
            return None

        p0 = OptionParams(S=float(S), K=float(row["strike"]), T=float(T_use), r=float(r), q=float(q), sigma=0.2)
        iv = implied_vol_bs("call", float(mid), p0)
        if iv is None:
            return None
        return float(np.clip(iv, 0.03, 1.50))

    except Exception:
        return None


def get_mkt_iv_yf(ticker: str, kind: OptionType, S: float, K: float, T: float, r: float, q: float,
                  prefer_days_range=(5, 365)) -> Optional[float]:
    """
    Implied vol "au strike K" depuis la chaîne yfinance :
    - pick expiry closest to T
    - pick CALL ou PUT selon kind, strike le plus proche de K
    - return impliedVolatility si dispo sinon inversion BS depuis mid/last
    """
    if not _HAVE_YF:
        return None
    try:
        tk = yf.Ticker(ticker)
        exps = list(getattr(tk, "options", []) or [])
        if not exps:
            return None

        today = dt.datetime.now(dt.UTC).date()
        target_days = int(round(T * 365))

        parsed = []
        for e in exps:
            try:
                ed = dt.datetime.strptime(e, "%Y-%m-%d").date()
                d = (ed - today).days
                if d <= 0:
                    continue
                parsed.append((abs(d - target_days), d, e))
            except Exception:
                continue
        if not parsed:
            return None

        lo_d, hi_d = prefer_days_range
        inside = [x for x in parsed if lo_d <= x[1] <= hi_d]
        _, d_days, expiry = min(inside or parsed, key=lambda t: t[0])
        T_use = d_days / 365.0

        chain = tk.option_chain(expiry)
        df = chain.calls if kind == "call" else chain.puts
        if df is None or df.empty or "strike" not in df.columns:
            return None

        df = df.copy()
        df["dist"] = (df["strike"].astype(float) - float(K)).abs()
        row = df.sort_values("dist").iloc[0]
        K_use = float(row["strike"])

        if "impliedVolatility" in df.columns:
            try:
                iv = float(row["impliedVolatility"])
                if np.isfinite(iv) and iv > 0:
                    return float(np.clip(iv, 0.03, 1.50))
            except Exception:
                pass

        mid = _mid_price_from_chain_row(row)
        if mid is None:
            return None

        p0 = OptionParams(S=float(S), K=float(K_use), T=float(T_use), r=float(r), q=float(q), sigma=0.2)
        iv = implied_vol_bs(kind, float(mid), p0)
        if iv is None:
            return None
        return float(np.clip(iv, 0.03, 1.50))

    except Exception:
        return None
def build_iv_surface_from_yf(ticker: str,
                             S0: float,
                             T_target: float,
                             r: float,
                             q: float,
                             max_expiries: int = 6,
                             moneyness=(0.7, 1.3),
                             prefer_days_range=(20, 365),
                            min_strikes: int = 9) -> "Optional[ImpliedVolSurface]":
    """
    Build ImpliedVolSurface(Tgrid, Kgrid, IVgrid) from yfinance option_chain.
    - Picks up to max_expiries expiries close-ish to T_target (and within prefer_days_range)
    - Uses calls; IV from 'impliedVolatility' if present else invert BS from mid/last
    - Builds a common strike grid (intersection if possible, else union + interpolation)
    """
    if not _HAVE_YF:
        return None

    try:
        import pandas as pd  # yfinance uses pandas; ensure available
    except Exception:
        return None

    try:
        tk = yf.Ticker(ticker)
        exps = list(getattr(tk, "options", []) or [])
        if not exps:
            return None

        today = dt.datetime.now(dt.UTC).date()
        target_days = int(round(T_target * 365))
        lo_d, hi_d = prefer_days_range

        parsed = []
        for e in exps:
            try:
                ed = dt.datetime.strptime(e, "%Y-%m-%d").date()
                d = (ed - today).days
                if d <= 0:
                    continue
                # keep a reasonable window
                if d < lo_d or d > hi_d:
                    continue
                parsed.append((abs(d - target_days), d, e))
            except Exception:
                continue

        if not parsed:
            # if none in window, relax and take closest expiry overall
            parsed = []
            for e in exps:
                try:
                    ed = dt.datetime.strptime(e, "%Y-%m-%d").date()
                    d = (ed - today).days
                    if d <= 0:
                        continue
                    parsed.append((abs(d - target_days), d, e))
                except Exception:
                    continue
            if not parsed:
                return None

        # choose expiries: closest to target, but return in increasing maturity
        parsed.sort(key=lambda x: x[0])
        chosen = parsed[:max_expiries]
        chosen.sort(key=lambda x: x[1])  # sort by days

        Tgrid = []
        perT = []  # list of dict K->iv

        Kmin, Kmax = moneyness[0] * float(S0), moneyness[1] * float(S0)

        for _, d, e in chosen:
            T_use = d / 365.0
            try:
                chain = tk.option_chain(e)
                calls = getattr(chain, "calls", None)
                if calls is None or calls.empty or "strike" not in calls.columns:
                    continue
                calls = calls.copy()
                calls = calls[(calls["strike"] >= Kmin) & (calls["strike"] <= Kmax)]
                if calls.empty:
                    continue

                k2iv = {}

                # If impliedVolatility exists, use it when finite
                has_iv = ("impliedVolatility" in calls.columns)

                for _, row in calls.iterrows():
                    K = float(row["strike"])

                    iv = None
                    if has_iv:
                        try:
                            v = float(row["impliedVolatility"])
                            if np.isfinite(v) and v > 0:
                                iv = v
                        except Exception:
                            iv = None

                    if iv is None:
                        mid = _mid_price_from_chain_row(row)
                        if mid is None:
                            continue
                        p0 = OptionParams(S=float(S0), K=float(K), T=float(T_use),
                                          r=float(r), q=float(q), sigma=0.2)
                        iv = implied_vol_bs("call", float(mid), p0)
                        if iv is None:
                            continue

                    iv = float(np.clip(iv, 0.03, 1.50))
                    k2iv[K] = iv

                if len(k2iv) < min_strikes:
                    continue

                Tgrid.append(float(T_use))
                perT.append(k2iv)

            except Exception:
                continue

        if len(Tgrid) < 2:
            return None

        # Build common K grid
        strike_sets = [set(d.keys()) for d in perT]
        common = sorted(set.intersection(*strike_sets)) if strike_sets else []
        if len(common) >= min_strikes:
            Kgrid = np.array(common, float)
            IV = []
            for k2iv in perT:
                IV.append([k2iv[float(K)] for K in Kgrid])
            print("[iv] source = yahoo option_chain (intersection grid)")
            return ImpliedVolSurface(Tgrid, Kgrid, IV)

        # Fallback: union grid + interpolation
        union = sorted(set.union(*strike_sets))
        if len(union) < min_strikes:
            return None
        Kgrid = np.array(union, float)

        IV = []
        for k2iv in perT:
            Ks = np.array(sorted(k2iv.keys()), float)
            Vs = np.array([k2iv[k] for k in Ks], float)
            row = np.interp(np.clip(Kgrid, Ks[0], Ks[-1]), Ks, Vs)
            IV.append(row.tolist())

        print("[iv] source = yahoo option_chain (union+interp grid)")
        return ImpliedVolSurface(Tgrid, Kgrid, IV)

    except Exception:
        return None
    
class ImpliedVolSurface:
    def __init__(self,Tgrid,Kgrid,IV): self.T=np.array(Tgrid); self.K=np.array(Kgrid); self.IV=np.array(IV,float)
    def _interp1d(self,x,xp,fp): return float(np.interp(np.clip(x,xp[0],xp[-1]),xp,fp))
    def iv(self,K,T):
        row=np.array([self._interp1d(T,self.T,self.IV[:,j]) for j in range(len(self.K))])
        return self._interp1d(K,self.K,row)

class LocalVolSurface:
    """
    Local volatility surface σ_loc(K,T) computed from Dupire.
    We'll evaluate it at K=S_node when simulating local vol dynamics.
    """
    def __init__(self, Tgrid, Kgrid, LV):
        self.T = np.array(Tgrid, float)
        self.K = np.array(Kgrid, float)
        self.LV = np.array(LV, float)

    def _interp1d(self, x, xp, fp):
        return float(np.interp(np.clip(x, xp[0], xp[-1]), xp, fp))

    def lv(self, K, T):
        # Interpolate in T for each strike column, then in K
        row = np.array([self._interp1d(T, self.T, self.LV[:, j]) for j in range(len(self.K))])
        return self._interp1d(K, self.K, row)


def build_dupire_local_vol_surface(iv_surf: ImpliedVolSurface,
                                  S0: float,
                                  r_fn,
                                  q_fn,
                                  clip=(0.03, 1.50),
                                  eps=1e-12) -> LocalVolSurface:
    """
    Build σ_loc(K,T) using Dupire formula from a call price surface C(K,T).

    Dupire:
    σ_loc^2(K,T) = [∂C/∂T + (r-q)K ∂C/∂K + q C] / [0.5 K^2 ∂^2C/∂K^2]

    We create C(K,T) from BS with implied vol σ_impl(K,T) = iv_surf.IV grid.
    Robustness: if num/den invalid, fallback to σ_impl.
    """
    T = np.array(iv_surf.T, float)
    K = np.array(iv_surf.K, float)
    IV = np.array(iv_surf.IV, float)

    nT, nK = IV.shape
    if nT < 3 or nK < 3:
        # Not enough points for stable 2nd derivatives
        return LocalVolSurface(T, K, np.clip(IV, clip[0], clip[1]))

    # 1) Build call price surface C(T_i, K_j) using BS + implied vols
    C = np.zeros((nT, nK), float)
    for i in range(nT):
        Ti = float(T[i])
        r = float(r_fn(Ti))
        q = float(q_fn(Ti))
        for j in range(nK):
            sig = float(max(1e-8, IV[i, j]))
            C[i, j] = bs_price("call", OptionParams(S=float(S0), K=float(K[j]), T=Ti, r=r, q=q, sigma=sig))

    # 2) Derivatives (use numpy.gradient to handle non-uniform grids)
    # dC/dT along axis=0 (T dimension)
    dC_dT = np.gradient(C, T, axis=0, edge_order=2)

    # dC/dK and d2C/dK2 along axis=1 (K dimension)
    dC_dK = np.gradient(C, K, axis=1, edge_order=2)
    d2C_dK2 = np.gradient(dC_dK, K, axis=1, edge_order=2)

    # 3) Apply Dupire pointwise
    LV = np.zeros_like(IV)
    for i in range(nT):
        Ti = float(T[i])
        r = float(r_fn(Ti))
        q = float(q_fn(Ti))
        for j in range(nK):
            Kj = float(K[j])

            num = dC_dT[i, j] + (r - q) * Kj * dC_dK[i, j] + q * C[i, j]
            den = 0.5 * (Kj ** 2) * d2C_dK2[i, j]

            sigma2 = None
            if np.isfinite(num) and np.isfinite(den) and abs(den) > eps:
                sigma2 = num / den

            # Robust fallback to implied vol if invalid
            if (sigma2 is None) or (not np.isfinite(sigma2)) or (sigma2 <= 0):
                lv = float(max(1e-8, IV[i, j]))
            else:
                lv = float(np.sqrt(max(sigma2, 1e-12)))

            LV[i, j] = float(np.clip(lv, clip[0], clip[1]))

    return LocalVolSurface(T, K, LV)

def build_iv_surface_synth(S:float,sigma_ref:float,
                           T=(0.0833,0.25,0.5,1.0), M=(0.7,0.85,0.9,1.0,1.1,1.2,1.3),
                           beta=0.15, skew=-0.25, smile=0.20):
    T=list(float(t) for t in T if t>0); K=[S*m for m in M]; IV=[]
    for t in T:
        base=max(1e-4,float(sigma_ref)*(t**float(beta)))
        row=[]; 
        for k in K:
            x=k/S-1.0; v=base*(1.0+skew*x+smile*x*x); row.append(float(np.clip(v,0.03,1.50)))
        IV.append(row)
    return ImpliedVolSurface(T,K,IV)

class SigmaProvider:
    def __init__(self, mode: str, const_sigma: float, S0: float, r_fn, q_fn,
                 iv_surf: Optional[ImpliedVolSurface] = None,
                 lv_surf: Optional[LocalVolSurface] = None):
        self.mode = mode
        self.const = max(1e-8, float(const_sigma))
        self.S0 = float(S0)
        self.r_fn = r_fn
        self.q_fn = q_fn
        self.iv_surf = iv_surf
        self.lv_surf = lv_surf

    def sigma(self, S, t):
        t = max(1e-8, float(t))

        if self.mode == "constant" or (self.iv_surf is None and self.lv_surf is None):
            return self.const

        # term: use implied vol on forward as you already do
        if self.mode == "term":
            if self.iv_surf is None:
                return self.const
            r, q = self.r_fn(t), self.q_fn(t)
            F = self.S0 * exp((r - q) * t)
            return max(1e-8, float(self.iv_surf.iv(F, t)))

        # local (your old heuristic): sigma = IV(S,t)
        if self.mode == "local":
            if self.iv_surf is None:
                return self.const
            return max(1e-8, float(self.iv_surf.iv(S, t)))

        # dupire local vol: sigma = sigma_loc(S,t) by evaluating σ_loc(K=S, T=t)
        if self.mode == "dupire":
            if self.lv_surf is None:
                return self.const
            return max(1e-8, float(self.lv_surf.lv(S, t)))

        raise ValueError(f"Unknown vol mode: {self.mode}")

def binomial_tree(kind:OptionType,p:OptionParams,steps:int,american:bool,sp:Optional[SigmaProvider]=None):
    if steps<1: raise ValueError("steps>=1")
    if p.T==0.0:
        v=max(0.0,p.S-p.K) if kind=="call" else max(0.0,p.K-p.S)
        return v,[[p.S]],[[v]],[[False]]
    mode=sp.mode if sp else "constant"; dt=p.T/steps

    # constant
    if mode=="constant":
        u=exp(p.sigma*sqrt(dt)); d=1.0/u; disc=exp(-p.r*dt)
        prob=(exp((p.r-p.q)*dt)-d)/(u-d); prob=min(max(prob,0.0),1.0)
        S=[[p.S*(u**j)*(d**(i-j)) for j in range(i+1)] for i in range(steps+1)]
        V=[[] for _ in range(steps+1)]; EX=[[False]*(i+1) for i in range(steps+1)]
        last=[max(0.0,s-p.K) if kind=="call" else max(0.0,p.K-s) for s in S[-1]]; V[-1]=last
        for i in range(steps-1,-1,-1):
            row=[]
            for j in range(i+1):
                cont=disc*(prob*V[i+1][j+1]+(1.0-prob)*V[i+1][j])
                if american:
                    s=S[i][j]; intr=max(0.0,s-p.K) if kind=="call" else max(0.0,p.K-s)
                    if intr>cont: EX[i][j]=True; row.append(intr)
                    else: row.append(cont)
                else: row.append(cont)
            V[i]=row
        return V[0][0],S,V,EX

    # term (recombinant)
    if mode=="term":
        u=np.zeros(steps); d=np.zeros(steps); prob=np.zeros(steps); disc=np.zeros(steps)
        for i in range(steps):
            t=max(1e-8,i*dt); sig=sp.sigma(p.S,t); u[i]=exp(sig*sqrt(dt)); d[i]=1.0/u[i]
            a=exp((sp.r_fn(t)-sp.q_fn(t))*dt); prob[i]=min(max((a-d[i])/(u[i]-d[i]),0.0),1.0)
            disc[i]=exp(-sp.r_fn(t)*dt)
        S=[[0.0]*(i+1) for i in range(steps+1)]; S[0][0]=p.S
        for i in range(1,steps+1):
            S[i][0]=S[i-1][0]*d[i-1]
            for j in range(1,i+1): S[i][j]=S[i-1][j-1]*u[i-1]
        V=[max(0.0,s-p.K) if kind=="call" else max(0.0,p.K-s) for s in S[-1]]
        EX=[[False]*(i+1) for i in range(steps+1)]; Vlat=[[] for _ in range(steps+1)]; Vlat[-1]=V[:]
        for i in range(steps-1,-1,-1):
            nxt=np.zeros(i+1)
            for j in range(i+1):
                cont=disc[i]*(prob[i]*V[j+1]+(1.0-prob[i])*V[j])
                if american:
                    intr=max(0.0,S[i][j]-p.K) if kind=="call" else max(0.0,p.K-S[i][j])
                    nxt[j]=intr if intr>cont else cont; EX[i][j]=intr>cont
                else: nxt[j]=cont
            V=nxt.tolist(); Vlat[i]=V
        return float(V[0]),S,Vlat,EX

    # local (non-recombinant)
    S=[[p.S]]; prob=[]; disc=[]
    for i in range(steps):
        t=max(1e-8,i*dt); disc.append(exp(-sp.r_fn(t)*dt))
        u_i=[]; d_i=[]; p_i=[]
        for j in range(i+1):
            sig=sp.sigma(S[i][j],t); u=exp(sig*sqrt(dt)); d=1.0/u
            a=exp((sp.r_fn(t)-sp.q_fn(t))*dt); p_star=min(max((a-d)/(u-d),0.0),1.0)
            u_i.append(u); d_i.append(d); p_i.append(p_star)
        prob.append(p_i)
        nxt=[0.0]*(i+2); nxt[0]=S[i][0]*d_i[0]
        for j in range(1,i+1): nxt[j]=S[i][j-1]*u_i[j-1]
        nxt[i+1]=S[i][i]*u_i[i]; S.append(nxt)
    V=[max(0.0,s-p.K) if kind=="call" else max(0.0,p.K-s) for s in S[-1]]
    EX=[[False]*(i+1) for i in range(steps+1)]; Vlat=[[] for _ in range(steps+1)]; Vlat[-1]=V[:]
    for i in range(steps-1,-1,-1):
        nxt=np.zeros(i+1)
        for j in range(i+1):
            cont=disc[i]*(prob[i][j]*V[j+1]+(1.0-prob[i][j])*V[j])
            intr=max(0.0,S[i][j]-p.K) if kind=="call" else max(0.0,p.K-S[i][j])
            if american and intr>cont: EX[i][j]=True; nxt[j]=intr
            else: nxt[j]=cont
        V=nxt.tolist(); Vlat[i]=V
    return float(V[0]),S,Vlat,EX

# PLOT 
# import flags

def setup_matplotlib_backend():
    """
    Select an interactive backend for notebooks, or a GUI backend for plain .py scripts.
    Returns the chosen backend name.
    """
    if not _HAVE_MPL:
        return "none"

    import matplotlib

    # Detect if running inside IPython / Jupyter
    in_ipython = False
    try:
        from IPython import get_ipython
        in_ipython = get_ipython() is not None
    except Exception:
        in_ipython = False

    if in_ipython:
        # Try widget first, then notebook, otherwise non-interactive fallback
        try:
            ip = get_ipython()
            ip.run_line_magic("matplotlib", "widget")
            return "widget"
        except Exception:
            try:
                ip = get_ipython()
                ip.run_line_magic("matplotlib", "notebook")
                return "notebook"
            except Exception:
                matplotlib.use("Agg")
                return "Agg"
    else:
        # Plain .py script: use a GUI backend if possible
        try:
            import matplotlib
            matplotlib.use("Qt5Agg")
            return "Qt5Agg"
        except Exception:
            try:
                matplotlib.use("TkAgg")
                return "TkAgg"
            except Exception:
                import matplotlib

                matplotlib.use("Agg")
                return "Agg"

def _is_interactive_backend():
    if not _HAVE_MPL:
        return False
    import matplotlib
    return matplotlib.get_backend().lower() not in ("agg",)

def plot_tree(S, V, title, ex_mask=None, currency_symbol="$"):
    if not _HAVE_MPL:
        print("Install matplotlib to view the tree plot.")
        return

    n = len(S) - 1
    fig, ax = plt.subplots(figsize=(max(10, 0.2*n + 10), max(6, 0.12*n + 6)))
    fig.suptitle(title)

    # edges
    for i in range(n):
        for j in range(i + 1):
            ax.plot([i, i+1], [j, j],   lw=0.5, alpha=0.6)
            ax.plot([i, i+1], [j, j+1], lw=0.5, alpha=0.6)

    # # nodes (and buffers for tooltips)
    xs, ys, cols = [], [], []
    I, J, Sf, Vf, Ef = [], [], [], [], []
    for i in range(n + 1):
        for j in range(i + 1):
            xs.append(i); ys.append(j)
            early = bool(ex_mask and ex_mask[i][j])
            cols.append("red" if early else "C0")
            I.append(i); J.append(j)
            Sf.append(S[i][j]); Vf.append(V[i][j]); Ef.append(early)

    sc = ax.scatter(xs, ys, s=14, c=cols)
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(-0.5, n + 0.5)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Node index")

    # Tooltips si backend interactif + mplcursors
    if _is_interactive_backend() and _HAVE_MPLCURSORS:
        cursor = mplcursors.cursor(sc, hover=True)

        @cursor.connect("add")
        def _on(sel):
            k = sel.index
            i, j = I[k], J[k]
            up, down = j, i - j
            txt = (f"step i={i}, node j={j}\n"
                   f"up={up}, down={down}\n"
                   f"S={currency_symbol}{Sf[k]:.4f}\n"
                   f"V={currency_symbol}{Vf[k]:.4f}")
            if Ef[k]:
                txt += "\n(EARLY EX.)"
            sel.annotation.set_text(txt)
            sel.annotation.get_bbox_patch().set(alpha=0.95)
    else:

        if not _is_interactive_backend():
            print("Tooltips disabled: non-interactive backend (use Jupyter + `%matplotlib widget`).")
        elif not _HAVE_MPLCURSORS:
            print("Tooltips disabled: install mplcursors → pip install mplcursors")

    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.show()

#  APPLICATION 
print_missing_deps()
def main():
    setup_matplotlib_backend()  # choose an appropriate matplotlib backend
    print(" Smart Option Pricer (compact) ")
    print("Data source: Yahoo Finance with robust stooq/manual fallback.")

    # 1) Contract inputs before fetching market data
    ticker = input("Ticker (ex: AAPL, MSFT, NVDA )): ").strip()
    kind   = input("Option type ('call'/'put'): ").strip().lower()
    days   = int(input("Time to maturity (days): ").strip())
    T      = days / 365.0

    # Exercise style = contract feature (not ticker-based)
    raw = (input("Exercise style (contract) [A]=American / [E]=European (default: A): ").strip().lower() or "a")
    if raw in ("a", "am", "american"):
        exercise_style = "american"
    elif raw in ("e", "eu", "european"):
        exercise_style = "european"
    else:
        exercise_style = "american"

    is_american_contract = (exercise_style == "american")

    # 2) Fetch market data (yfinance / stooq / manual) 
    S, q, sigma, src = fetch_market_params(ticker)

    # 3) Market region / currency / risk-free rate
    region = detect_market_region(ticker)
    cur_code, cur_sym = detect_currency(ticker)

    r_auto = get_risk_free_rate(region, T)
    print(f"Auto risk-free (cont.): {r_auto:.6f}")
    r_in = input(f"Risk-free r (cont.) [{r_auto:.6f}] - press ENTER to keep: ").strip()
    r = float(r_in) if r_in else float(r_auto)

    # 4) Choose sigma source (historical vs ATM implied vol) BEFORE asking for K
    print(f"\nS0={cur_sym}{S:.4f} | q={q:.4f} | sigma_hist={sigma:.4f}  [src: {src}]")
    K = float(input(f"Strike K (empty=ATM={S:.4f}): ").strip() or S)

    sigma_choice = (input(
        "Volatility input [hist=Historical | atm=ATM Implied | mkt=Implied at strike K] (default: hist): "
    ).strip() or "hist").lower()

    # petit support pour raccourcis
    if sigma_choice in ("atm", "a"):
        sigma_choice = "atm_iv"
    if sigma_choice in ("mkt", "m", "k"):
        sigma_choice = "mkt_iv_at_k"
    if sigma_choice in ("h",):
        sigma_choice = "hist"

    sigma_used = sigma  # default = historique

    if sigma_choice == "atm_iv":
        iv = get_atm_iv_yf(ticker, S=S, T=T, r=r, q=q)
        if iv is not None:
            sigma_used = iv
            print(f"[calib] ATM implied vol used: sigma={sigma_used:.4f} (yfinance chain)")
        else:
            print("[calib] ATM implied vol unavailable → fallback to historical sigma.")

    elif sigma_choice == "mkt_iv_at_k":
        iv_k = get_mkt_iv_yf(ticker, kind=kind, S=S, K=K, T=T, r=r, q=q)
        if iv_k is not None:
            sigma_used = iv_k
            print(f"[calib] Market IV(K) used: sigma={sigma_used:.4f} (from option chain)")
        else:
            print("[calib] Market IV(K) unavailable → fallback to historical sigma.")

    else:
        # hist
        pass

    # 5) Binomial steps & volatility mode
    steps    = int(input("Binomial steps [ex: 10]: ").strip() or "10")
    vol_mode = (input("Vol mode 'constant'|'term'|'local'|'dupire' [constant]: ").strip() or "constant").lower()

    if vol_mode not in ("constant","term","local","dupire"):
        vol_mode = "constant"

    if vol_mode in ("local","dupire") and steps > 200:
        print("[warn] local/dupire is non-recombining (O(n^2)). Consider steps <= 200.")
    # 6) Pricing
    r_fn = lambda t, _r=r: _r
    q_fn = lambda t, _q=q: _q

    iv_surface = None
    lv_surface = None

    # Ask IV surface source only when needed
    if vol_mode in ("term", "local", "dupire"):
        iv_src = (input("IV surface source 'yahoo'|'synth' [yahoo]: ").strip() or "yahoo").lower()

        # Build IV surface
        if iv_src == "yahoo":
            iv_surface = build_iv_surface_from_yf(ticker, S0=S, T_target=T, r=r, q=q)
            if iv_surface is None:
                print("[iv] yahoo chain unavailable → fallback to synthetic surface.")
                iv_surface = build_iv_surface_synth(S, sigma_used)
        else:
            iv_surface = build_iv_surface_synth(S, sigma_used)

        # Build Dupire local vol surface if requested
        if vol_mode == "dupire":
            lv_surface = build_dupire_local_vol_surface(iv_surface, S0=S, r_fn=r_fn, q_fn=q_fn)
            print("[dupire] Local vol surface built from IV surface (Dupire).")

    sp = SigmaProvider(vol_mode, sigma_used, S, r_fn, q_fn, iv_surf=iv_surface, lv_surf=lv_surface)
    p  = OptionParams(S=S, K=K, T=T, r=r, q=q, sigma=sigma_used)

    eu, euS, euV, euEX = binomial_tree(kind, p, steps, american=False, sp=sp)

    # Exercise style logic
    american_flag = bool(is_american_contract)

    note = ""
    if american_flag and kind == "call" and q <= 1e-10:
        american_flag = False
        note = " (no early-exercise benefit for call when q≈0)"

    contract_label  = "AMERICAN" if is_american_contract else "EUROPEAN"
    effective_label = "AMERICAN" if american_flag else "EUROPEAN"

    # Recruiter-friendly label for plot / summary
    if contract_label != effective_label:
        display_style = f"{contract_label.title()} (European-equivalent)"
    else:
        display_style = contract_label.title()

    print(f"Exercise: Contract={contract_label} | Priced as={effective_label}{note} (kind={kind}, q={q:.6f})")
    if american_flag:
        us, usS, usV, usEX = binomial_tree(kind, p, steps, american=True, sp=sp)
    else:
        us, usS, usV, usEX = eu, euS, euV, euEX

    bs    = bs_price(kind, p)
    prem  = us - eu
    early = sum(1 for row in usEX for m in row if m)

    print(f"\nMarket region: {region.upper()} | CCY: {cur_code}")
    print(f"European {kind}: Binomial={eu:.6f} | Black–Scholes={bs:.6f}")
    print(f"American  {kind}: Binomial={us:.6f} | Premium={prem:.6f}")
    if kind == "put":
        print(f"Early-ex nodes: {early}")

    # Plot tree 
    title = (
        f"Binomial Tree  {display_style} {kind} | "
        f"{steps} steps | vol={vol_mode}"
    )

    if note:
        title += f"\nReason: {note.strip()}"    
    if american_flag:
        plot_tree(usS, usV, title, ex_mask=usEX, currency_symbol=cur_sym)
    else:
        plot_tree(euS, euV, title, ex_mask=None, currency_symbol=cur_sym)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")