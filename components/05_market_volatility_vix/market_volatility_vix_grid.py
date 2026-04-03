"""
market_volatility_vix_grid.py
==============================
Grid search: různé formule pro VIX × normalizační metody × okna
CNN říká: VIX vs. 50-day moving average
Vysoký VIX = strach → inverse normalizace

Formule:
  - Raw VIX (absolutní hodnota)
  - VIX / MA(n)  — poměr k MA (různá n)
  - VIX - MA(n)  — rozdíl od MA (různá n)
  - VIX pct_change(n) — n-denní změna v %

Celkem: 13 formulí × 3 metody × 12 oken = 468 kombinací
Výsledky se NEUKLÁDAJÍ — jen výpis do konzole.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

_dir = Path(__file__).resolve().parent
CNN_CSV = _dir / '../../data/fear_greed_historical.csv'

WINDOWS = [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512]
MA_WINDOWS = [20, 50, 125, 252]

# ── Data ──────────────────────────────────────────────────────────────────────
vix_csv = _dir / 'vix_daily_data.csv'

if vix_csv.exists():
    try:
        vix_df = pd.read_csv(vix_csv, skiprows=2)
        vix_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        vix_df['Date'] = pd.to_datetime(vix_df['Date'])
        vix = vix_df.set_index('Date')['Close'].astype(float).dropna()
    except Exception:
        vix_df = pd.read_csv(vix_csv, parse_dates=['Date'], index_col='Date')
        vix = vix_df.iloc[:, 0].astype(float).dropna()
    print(f"Nacteno z CSV: {len(vix)} dni  ({vix.index[0].date()} → {vix.index[-1].date()})")
else:
    print("Stahuji VIX (^VIX) z Yahoo Finance...")
    raw = yf.download('^VIX', start='1990-01-01', end='2026-03-20', progress=False)
    vix = raw['Close'].iloc[:, 0] if isinstance(raw['Close'], pd.DataFrame) else raw['Close']
    vix.index = pd.to_datetime(vix.index)
    raw.to_csv(vix_csv)
    print(f"  Ulozeno {len(vix)} dni")

# ── CNN FGI ───────────────────────────────────────────────────────────────────
cnn = pd.read_csv(CNN_CSV, parse_dates=['date'], index_col='date')['value']
cnn.index.name = 'Date'

# ── Formule raw signálu ───────────────────────────────────────────────────────
formulas = {}

# Raw VIX
formulas['VIX_raw'] = vix.copy()

# VIX / MA(n) a VIX - MA(n)
for n in MA_WINDOWS:
    ma = vix.rolling(n).mean()
    formulas[f'VIX/MA{n}']  = (vix / ma).dropna()
    formulas[f'VIX-MA{n}']  = (vix - ma).dropna()

# VIX n-denní změna
for n in [5, 10, 20]:
    formulas[f'VIX_pct{n}d'] = vix.pct_change(n).dropna()

# ── Normalizace ───────────────────────────────────────────────────────────────
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std()
    z = (series - rm) / rs
    if inverse: z = -z
    return (z * 25 + 50).clip(0, 100)

def rolling_percentile(series, window, inverse=False):
    def rank_last(x):
        return (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100
    r = series.rolling(window, min_periods=window // 2).apply(rank_last, raw=True).clip(0, 100)
    return (100 - r) if inverse else r

def rolling_minmax(series, window, inverse=False):
    rmin = series.rolling(window, min_periods=window // 2).min()
    rmax = series.rolling(window, min_periods=window // 2).max()
    z = (series - rmin) / (rmax - rmin)
    if inverse: z = 1 - z
    return (z * 100).clip(0, 100)

NORM_METHODS = {
    'zscore':     rolling_zscore,
    'minmax':     rolling_minmax,
    'percentile': rolling_percentile,
}

# ── Grid search ───────────────────────────────────────────────────────────────
print(f"\n── Grid search (best method+window per formule) ────────────────────────")
print(f"  {'Formule':<20} {'r':>7}  {'w':>6}  {'Metoda':<12}  {'Od':>12}")
print(f"  {'-'*65}")

results = []

for fname, raw in formulas.items():
    raw = raw.dropna()
    if len(raw) < 100:
        continue

    overlap = raw.index.intersection(cnn.index)
    best_r, best_w, best_method = 0, None, None

    for method_name, norm_fn in NORM_METHODS.items():
        for w in WINDOWS:
            try:
                norm = norm_fn(raw, w, inverse=True)
                valid = overlap[norm.loc[overlap].notna() & cnn.loc[overlap].notna()]
                if len(valid) < 200:
                    continue
                r, _ = pearsonr(norm.loc[valid], cnn.loc[valid])
                if r > best_r:
                    best_r, best_w, best_method = r, w, method_name
            except Exception:
                continue

    if best_r > 0:
        covers = raw.index[0].year <= 1998
        marker = " ✓1998" if covers else "      "
        results.append({
            'formula': fname, 'r': best_r, 'w': best_w,
            'method': best_method, 'start': raw.index[0].date(), 'covers': covers
        })
        print(f"  {fname:<20} {best_r:>7.3f}  {best_w:>5}d  {best_method:<12}  {str(raw.index[0].date()):>12}{marker}")

# ── Výsledek ──────────────────────────────────────────────────────────────────
results.sort(key=lambda x: x['r'], reverse=True)
print(f"\n  Původní verze (VIX/MA50, Z-score 1512d): r=0.644")

if results:
    best = results[0]
    print(f"\n  --> NEJLEPŠÍ: {best['formula']}  r={best['r']:.3f}  w={best['w']}d  {best['method']}  (od {best['start']})")

    print(f"\n  Top 5:")
    for res in results[:5]:
        covers = "✓1998" if res['covers'] else "     "
        print(f"    {res['formula']:<20} r={res['r']:.3f}  w={res['w']:4d}d  {res['method']:<12}  {covers}")
