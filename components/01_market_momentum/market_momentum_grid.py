"""
market_momentum_grid.py
=======================
Grid search: typ signálu × normalizační metody × normalizační okna
CNN metodika: "S&P 500 above/below its 125-day moving average" — MA125 fixní

Dvě formulace raw signálu (obě s MA125):
  pct:  (SP500 - MA125) / MA125 × 100  — procentuální odchylka
  diff: SP500 - MA125                  — absolutní rozdíl

Dimenze grid searche:
  - Typy signálu: pct, diff (2 hodnoty)
  - Normalizační metody: Z-score, Min-Max, Percentile (3 metody)
  - Normalizační okna: [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512] (12 oken)
  Celkem: 2 × 3 × 12 = 72 kombinací

Výsledky se NEUKLÁDAJÍ — jen výpis do konzole.

Author: Petr Amler (AML0005)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import pearsonr

_dir    = Path(__file__).resolve().parent
CNN_CSV = _dir / '../../data/fear_greed_historical.csv'

NORM_WINDOWS = [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512]
MA_WINDOW    = 125  # CNN metodika — fixní

SIGNAL_TYPES = {
    'pct':  lambda sp, ma: (sp - ma) / ma * 100,  # procentuální odchylka (aktuální produkce)
    'diff': lambda sp, ma: sp - ma,                # absolutní rozdíl
}

# ── ČÁST 1: Data ──────────────────────────────────────────────────────────────
print("Stahuji ^GSPC (S&P 500 price index)...")
sp500_raw = yf.download('^GSPC', start='1995-01-01', end='2026-03-20', progress=False)['Close']
sp500_raw.index = pd.to_datetime(sp500_raw.index)
sp500 = sp500_raw.iloc[:, 0] if isinstance(sp500_raw, pd.DataFrame) else sp500_raw

cnn = pd.read_csv(CNN_CSV, parse_dates=['date'], index_col='date')['value']
cnn.index.name = 'Date'

print(f"^GSPC: {sp500.index[0].date()} → {sp500.index[-1].date()}  ({len(sp500)} dní)")
print(f"CNN FGI: {cnn.index[0].date()} → {cnn.index[-1].date()}  ({len(cnn)} dní)")

# ── ČÁST 2: Normalizační metody ───────────────────────────────────────────────
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std(ddof=0)
    z  = (series - rm) / rs
    if inverse:
        z = -z
    return (z * 25 + 50).clip(0, 100)

def rolling_percentile(series, window, inverse=False):
    def rank_last(x):
        return (x[:-1] < x[-1]).sum() / (len(x) - 1) * 100
    r = series.rolling(window, min_periods=window // 2).apply(rank_last, raw=True).clip(0, 100)
    return (100 - r) if inverse else r

def rolling_minmax(series, window, inverse=False):
    rmin = series.rolling(window, min_periods=window // 2).min()
    rmax = series.rolling(window, min_periods=window // 2).max()
    z    = (series - rmin) / (rmax - rmin)
    if inverse:
        z = 1 - z
    return (z * 100).clip(0, 100)

NORM_METHODS = {
    'zscore':     rolling_zscore,
    'minmax':     rolling_minmax,
    'percentile': rolling_percentile,
}

# ── ČÁST 3: Grid search ───────────────────────────────────────────────────────
print(f"\n── Grid search: typ signálu × normalizační metoda × norm. okno (MA125 fixní) ──")
print(f"  {'Signál':<6}  {'Norm. metoda':<12}  {'Norm. okno':>10}  {'r':>7}  {'n':>5}")
print(f"  {'-'*52}")

ma      = sp500.rolling(MA_WINDOW).mean()
results = []

for sig_name, sig_fn in SIGNAL_TYPES.items():
    raw     = sig_fn(sp500, ma).dropna()
    overlap = raw.index.intersection(cnn.index)

    for method_name, norm_fn in NORM_METHODS.items():
        best_r_method = 0
        for nw in NORM_WINDOWS:
            try:
                norm  = norm_fn(raw, nw)
                valid = overlap[norm.loc[overlap].notna() & cnn.loc[overlap].notna()]
                if len(valid) < 100:
                    continue
                r, _ = pearsonr(norm.loc[valid], cnn.loc[valid])
                results.append({
                    'signal':      sig_name,
                    'norm_method': method_name,
                    'norm_window': nw,
                    'r':           r,
                    'n':           len(valid),
                })
                marker = " ← NEJLEPSI" if r > best_r_method else ""
                if r > best_r_method:
                    best_r_method = r
                print(f"  {sig_name:<6}  {method_name:<12}  {nw:>10}  {r:>7.3f}  {len(valid):>5}{marker}")
            except Exception:
                continue

# ── ČÁST 4: Výsledek ─────────────────────────────────────────────────────────
results.sort(key=lambda x: x['r'], reverse=True)
print(f"\n{'─'*52}")
print(f"  Produkční verze (diff, MA125, Z-score w=63d):  r=0.832")
print(f"\n  Top 5 kombinací:")
print(f"  {'Signál':<6}  {'Norm. metoda':<12}  {'Norm. okno':>10}  {'r':>7}  {'n':>5}")
print(f"  {'-'*52}")
for res in results[:5]:
    print(f"  {res['signal']:<6}  {res['norm_method']:<12}  {res['norm_window']:>10}  {res['r']:>7.3f}  {res['n']:>5}")

if results:
    best = results[0]
    print(f"\n  --> NEJLEPŠÍ: signal={best['signal']}  MA125  {best['norm_method']}  w={best['norm_window']}d  r={best['r']:.3f}")
