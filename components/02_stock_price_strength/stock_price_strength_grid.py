"""
stock_price_strength_grid.py
=============================
Grid search: různé formule × normalizační metody × okna
CNN říká: "Number of stocks hitting 52-week highs vs lows on NYSE"
Hledáme nejlepší kombinaci pro Stock Price Strength komponentu.
Výsledky se NEUKLÁDAJÍ — jen výpis do konzole.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

_dir = Path(__file__).resolve().parent
CNN_CSV = _dir / '../../data/fear_greed_historical.csv'

# ── Data ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(_dir / 'stock_price_strength_1980_2026.csv', parse_dates=['Date'], index_col='Date')
df.index = df.index.normalize()

# Timestamp fix pro případ že index má čas
nyhgh = df['NYHGH'].dropna()
nylow = df['NYLOW'].dropna()

common = nyhgh.index.intersection(nylow.index)
nyhgh = nyhgh.loc[common]
nylow = nylow.loc[common]

print(f"Data: {common[0].date()} → {common[-1].date()}  ({len(common)} dní)")

# ── CNN FGI ───────────────────────────────────────────────────────────────────
cnn = pd.read_csv(CNN_CSV, parse_dates=['date'], index_col='date')['value']
cnn.index.name = 'Date'

# ── Formule raw signálu ───────────────────────────────────────────────────────
# Všechny formule: vyšší hodnota = více highs = greed = vysoké F&G skóre

formulas = {
    'Net (NYHGH-NYLOW)':               nyhgh - nylow,
    'Ratio (NYHGH-NYLOW)/(NYHGH+NYLOW)': (nyhgh - nylow) / (nyhgh + nylow).replace(0, np.nan),
    'Pct_Highs NYHGH/(NYHGH+NYLOW)':   nyhgh / (nyhgh + nylow).replace(0, np.nan),
    'Log_Ratio log(NYHGH/NYLOW)':       np.log((nyhgh / nylow.replace(0, np.nan)).replace(0, np.nan)),
    'NYHGH_only':                       nyhgh,
    'NYLOW_only (inv)':                 nylow,  # inverse — více low = strach
    # MA-smoothed verze (5d a 20d)
    'Net_MA5':                          (nyhgh - nylow).rolling(5).mean(),
    'Net_MA20':                         (nyhgh - nylow).rolling(20).mean(),
    'Ratio_MA5':                        ((nyhgh - nylow) / (nyhgh + nylow).replace(0, np.nan)).rolling(5).mean(),
    'Ratio_MA20':                       ((nyhgh - nylow) / (nyhgh + nylow).replace(0, np.nan)).rolling(20).mean(),
}

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

WINDOWS = [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512]
NORM_METHODS = {
    'zscore':     lambda s, w, inv: rolling_zscore(s, w, inv),
    'minmax':     lambda s, w, inv: rolling_minmax(s, w, inv),
    'percentile': lambda s, w, inv: rolling_percentile(s, w, inv),
}

# ── Grid search ───────────────────────────────────────────────────────────────
print(f"\n── Grid search (best window+method per formula) ────────────────────────")
print(f"  {'Formula':<42} {'r':>7}  {'w':>6}  {'Metoda':<12}  {'Od':>12}")
print(f"  {'-'*85}")

results = []

for fname, raw in formulas.items():
    raw = raw.dropna()
    if len(raw) < 100:
        continue

    # NYLOW je inverse
    is_inverse = 'inv' in fname.lower() or 'nylow_only' in fname.lower()

    best_r, best_w, best_method = 0, None, None
    overlap = raw.index.intersection(cnn.index)

    for method_name, norm_fn in NORM_METHODS.items():
        for w in WINDOWS:
            try:
                norm = norm_fn(raw, w, is_inverse)
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
        print(f"  {fname:<42} {best_r:>7.3f}  {best_w:>5}d  {best_method:<12}  {str(raw.index[0].date()):>12}{marker}")

# ── Výsledek ──────────────────────────────────────────────────────────────────
results.sort(key=lambda x: x['r'], reverse=True)
print(f"\n  Stará verze (Ratio, Z-score 504d): r=0.699")

if results:
    best = results[0]
    print(f"  --> NEJLEPŠÍ: {best['formula']}  r={best['r']:.3f}  w={best['w']}d  {best['method']}  (od {best['start']})")

    print(f"\n  Top 5:")
    for res in results[:5]:
        covers = "✓1998" if res['covers'] else "     "
        print(f"    {res['formula']:<42} r={res['r']:.3f}  w={res['w']:4d}d  {res['method']:<12}  {covers}")
