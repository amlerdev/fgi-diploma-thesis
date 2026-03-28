"""
stock_price_breadth_grid.py
===========================
Grid search: normalizační metody × normalizační okna pro NYSE McClellan Summation Index
Indikátor: !VMCSUMNYA (NYSE Volume McClellan Summation Index)

Dimenze grid searche:
  - Normalizační metody: Z-score, Min-Max, Percentile (3 metody)
  - Normalizační okna: [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512] (12 oken)
  Celkem: 3 × 12 = 36 kombinací

Výsledky se NEUKLÁDAJÍ — jen výpis do konzole.

Author: Petr Amler (AML0005)
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

_dir    = Path(__file__).resolve().parent
CNN_CSV = _dir / '../../data/fear_greed_historical.csv'
NYSI_CSV = _dir / 'nysi_data.csv'

NORM_WINDOWS = [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512]

# ── ČÁST 1: Data ──────────────────────────────────────────────────────────────
if not NYSI_CSV.exists():
    raise FileNotFoundError(f"Nenalezeno: {NYSI_CSV}\nNejprve spusť stock_price_breadth.py pro stažení dat.")

nysi_df = pd.read_csv(NYSI_CSV, parse_dates=['Date'])
nysi_df['Date'] = nysi_df['Date'].dt.normalize()
raw_signal = nysi_df.groupby('Date')['VMCSUMNYA'].last().dropna()
raw_signal.name = 'VMCSUMNYA'

cnn = pd.read_csv(CNN_CSV, parse_dates=['date'], index_col='date')['value']
cnn.index.name = 'Date'

overlap = raw_signal.index.intersection(cnn.index)

print(f"!VMCSUMNYA: {raw_signal.index[0].date()} → {raw_signal.index[-1].date()}  ({len(raw_signal)} dní)")
print(f"CNN FGI:    {cnn.index[0].date()} → {cnn.index[-1].date()}  ({len(cnn)} dní)")
print(f"Overlap:    {len(overlap)} dní")

# ── ČÁST 2: Normalizační metody ───────────────────────────────────────────────
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std()
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
print(f"\n── Grid search: normalizační metoda × normalizační okno ────────────────────")
print(f"  {'Norm. metoda':<12}  {'Norm. okno':>10}  {'r':>7}  {'n':>5}")
print(f"  {'-'*40}")

results = []

for method_name, norm_fn in NORM_METHODS.items():
    best_r_method = 0
    for nw in NORM_WINDOWS:
        try:
            norm  = norm_fn(raw_signal, nw)
            valid = overlap[norm.loc[overlap].notna() & cnn.loc[overlap].notna()]
            if len(valid) < 100:
                continue
            r, _ = pearsonr(norm.loc[valid], cnn.loc[valid])
            results.append({
                'norm_method': method_name,
                'norm_window': nw,
                'r':           r,
                'n':           len(valid),
            })
            marker = " ← NEJLEPSI" if r > best_r_method else ""
            if r > best_r_method:
                best_r_method = r
            print(f"  {method_name:<12}  {nw:>10}  {r:>7.3f}  {len(valid):>5}{marker}")
        except Exception:
            continue

# ── ČÁST 4: Výsledek ─────────────────────────────────────────────────────────
results.sort(key=lambda x: x['r'], reverse=True)
print(f"\n{'─'*50}")
print(f"  Produkční verze (Z-score w=126d):  r=0.816")
print(f"\n  Top 5 kombinací:")
print(f"  {'Norm. metoda':<12}  {'Norm. okno':>10}  {'r':>7}  {'n':>5}")
print(f"  {'-'*40}")
for res in results[:5]:
    print(f"  {res['norm_method']:<12}  {res['norm_window']:>10}  {res['r']:>7.3f}  {res['n']:>5}")

if results:
    best = results[0]
    print(f"\n  --> NEJLEPŠÍ: {best['norm_method']}  w={best['norm_window']}d  r={best['r']:.3f}")
