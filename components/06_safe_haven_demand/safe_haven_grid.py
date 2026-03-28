"""
safe_haven_grid.py
==================
Grid search: různé kombinace stock index × bond instrument × normalizační metody × okna
Hledáme nejlepší kombinaci pro Safe Haven Demand komponentu.
Výsledky se NEUKLÁDAJÍ — jen výpis do konzole.

Celkem: 5 stock × 5 bond × 3 metody × 12 oken = 900 kombinací
"""

from pathlib import Path
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

_dir = Path(__file__).resolve().parent
CNN_CSV = _dir / '../../data/fear_greed_historical.csv'

# ── Konfigurace: co testovat ─────────────────────────────────────────────────

STOCK_INDICES = {
    '^GSPC'  : 'S&P 500',
    '^DJI'   : 'Dow Jones 30',
    '^IXIC'  : 'NASDAQ Composite',
    '^NYA'   : 'NYSE Composite',
    '^RUT'   : 'Russell 2000',
}

BOND_INSTRUMENTS = {
    'VUSTX'  : 'Treasury 20+ yr (Vanguard)',
    'VFITX'  : 'Treasury 5-10 yr (Vanguard)',
    'VFISX'  : 'Treasury 1-3 yr (Vanguard)',
    'VBMFX'  : 'Total Bond Market (Vanguard)',
    'VWESX'  : 'Long-Term Bond (Vanguard)',
}

WINDOWS = [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512]

# ── Download dat ─────────────────────────────────────────────────────────────
print("Stahuji data...")

stock_data = {}
for ticker, name in STOCK_INDICES.items():
    try:
        raw = yf.download(ticker, start='1997-01-01', end='2026-12-31', progress=False)['Close']
        raw.index = pd.to_datetime(raw.index)
        s = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
        s = s.dropna()
        stock_data[ticker] = s
        print(f"  ✓ {ticker:<8} {name:<30} {s.index[0].date()} → {s.index[-1].date()}  ({len(s)} dní)")
    except Exception as e:
        print(f"  ✗ {ticker:<8} {name:<30} CHYBA: {e}")

print()
bond_data = {}
for ticker, name in BOND_INSTRUMENTS.items():
    try:
        raw = yf.download(ticker, start='1997-01-01', end='2026-12-31', progress=False)['Close']
        raw.index = pd.to_datetime(raw.index)
        b = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
        b = b.dropna()
        bond_data[ticker] = b
        print(f"  ✓ {ticker:<8} {name:<30} {b.index[0].date()} → {b.index[-1].date()}  ({len(b)} dní)")
    except Exception as e:
        print(f"  ✗ {ticker:<8} {name:<30} CHYBA: {e}")

# ── CNN FGI ──────────────────────────────────────────────────────────────────
cnn = pd.read_csv(CNN_CSV, parse_dates=['date'], index_col='date')['value']
cnn.index.name = 'Date'

# ── Normalizace ──────────────────────────────────────────────────────────────
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

# ── Grid search ──────────────────────────────────────────────────────────────
print(f"\n── Grid search (best window+method per stock×bond kombinace) ────────────")
print(f"  {'Stock':<8} {'Bond':<8} {'r':>7}  {'w':>6}  {'Metoda':<12}  {'Od':>12}")
print(f"  {'-'*65}")

results = []

for s_ticker, s_series in stock_data.items():
    for b_ticker, b_series in bond_data.items():
        try:
            s_ret = s_series.pct_change(periods=20) * 100
            b_ret = b_series.pct_change(periods=20) * 100

            common = s_ret.index.intersection(b_ret.index)
            raw = (s_ret.loc[common] - b_ret.loc[common]).dropna()

            if len(raw) < 100:
                continue

            overlap = raw.index.intersection(cnn.index)
            best_r, best_w, best_method = 0, None, None

            for method_name, norm_fn in NORM_METHODS.items():
                for w in WINDOWS:
                    try:
                        norm = norm_fn(raw, w)
                        valid = overlap[norm.loc[overlap].notna() & cnn.loc[overlap].notna()]
                        if len(valid) < 200:
                            continue
                        r, _ = pearsonr(norm.loc[valid], cnn.loc[valid])
                        if r > best_r:
                            best_r, best_w, best_method = r, w, method_name
                    except Exception:
                        continue

            if best_r > 0:
                start_date = raw.index[0].date()
                covers_1998 = raw.index[0].year <= 1998
                results.append({
                    'stock': s_ticker, 'bond': b_ticker,
                    'r': best_r, 'w': best_w, 'method': best_method,
                    'start': start_date, 'covers_1998': covers_1998,
                })

        except Exception as e:
            print(f"  CHYBA {s_ticker}+{b_ticker}: {e}")

# ── Výsledky ─────────────────────────────────────────────────────────────────
results.sort(key=lambda x: x['r'], reverse=True)

for res in results:
    marker = " ✓1998" if res['covers_1998'] else "      "
    print(f"  {res['stock']:<8} {res['bond']:<8} {res['r']:>7.3f}  {res['w']:>5}d  {res['method']:<12}  {str(res['start']):>12}{marker}")

print(f"\n  Stará verze (yield proxy DGS20, Z-score 504d): r=0.671")

if results:
    best = results[0]
    print(f"\n  --> NEJLEPŠÍ: {best['stock']} + {best['bond']}  r={best['r']:.3f}  w={best['w']}d  {best['method']}  (od {best['start']})")

    print(f"\n  Top 5:")
    for res in results[:5]:
        marker = "✓1998" if res['covers_1998'] else "     "
        print(f"    {res['stock']:<8} {res['bond']:<8} r={res['r']:.3f}  w={res['w']:4d}d  {res['method']:<12}  {marker}")
