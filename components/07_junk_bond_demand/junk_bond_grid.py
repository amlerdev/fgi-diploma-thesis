"""
junk_bond_grid.py
=================
Grid search: různé kombinace junk bond × investment grade spreadu × normalizační metody × okna
CNN říká: "Yield spread: junk bonds vs. investment grade"
Raw signal = junk_yield - ig_yield
Menší spread = greed, větší spread = fear → inverse normalizace

Celkem: 13 raw signálů × 3 metody × 12 oken = 468 kombinací
Výsledky se NEUKLÁDAJÍ — jen výpis do konzole.
"""

from pathlib import Path
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas_datareader.data as web
    HAS_PDR = True
except ImportError:
    HAS_PDR = False
    print("WARN: pandas_datareader není nainstalován — FRED data přeskočena")

_dir = Path(__file__).resolve().parent
CNN_CSV = _dir / '../../data/fear_greed_historical.csv'

WINDOWS = [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512]

# ── ČÁST 1: Download ETF dat (yfinance) ──────────────────────────────────────
print("Stahuji ETF data...")

ETF_TICKERS = {
    'HYG'   : 'iShares HY Corporate (2007+)',
    'JNK'   : 'SPDR HY Bond (2007+)',
    'VWEHX' : 'Vanguard HY Corporate (1978+)',
    'LQD'   : 'iShares IG Corporate (2002+)',
    'VWESX' : 'Vanguard LT Investment Grade (1973+)',
    'VBMFX' : 'Vanguard Total Bond Market (1986+)',
    'VCSH'  : 'Vanguard Short-Term Corp Bond (2009+)',
}

etf_data = {}
for ticker, name in ETF_TICKERS.items():
    try:
        raw = yf.download(ticker, start='1995-01-01', end='2026-03-20', progress=False)['Close']
        raw.index = pd.to_datetime(raw.index)
        s = raw.iloc[:, 0] if isinstance(raw, pd.DataFrame) else raw
        s = s.dropna()
        etf_data[ticker] = s
        print(f"  ✓ {ticker:<8} {name:<40} {s.index[0].date()} → {s.index[-1].date()}")
    except Exception as e:
        print(f"  ✗ {ticker:<8} {name:<40} CHYBA: {e}")

# ── ČÁST 2: Download FRED OAS dat ────────────────────────────────────────────
print("\nStahuji FRED OAS data...")

fred_data = {}
if HAS_PDR:
    FRED_SERIES = {
        'BAMLH0A0HYM2'  : 'ICE BofA HY OAS (vs Treasury)',
        'BAMLH0A1HYBB'  : 'ICE BofA BB OAS',
        'BAMLC0A4CBBB'  : 'ICE BofA BBB OAS',
        'BAMLC0A3CA'    : 'ICE BofA A OAS',
    }
    for series_id, name in FRED_SERIES.items():
        try:
            df = web.DataReader(series_id, 'fred', '1995-01-01', '2026-03-20')
            s = df.iloc[:, 0].dropna()
            s.index = pd.to_datetime(s.index)
            s.index.name = 'Date'
            fred_data[series_id] = s
            print(f"  ✓ {series_id:<20} {name:<35} {s.index[0].date()} → {s.index[-1].date()}")
        except Exception as e:
            print(f"  ✗ {series_id:<20} {name:<35} CHYBA: {e}")

# ── ČÁST 3: CNN FGI ──────────────────────────────────────────────────────────
cnn = pd.read_csv(CNN_CSV, parse_dates=['date'], index_col='date')['value']
cnn.index.name = 'Date'

# ── ČÁST 4: Normalizace ──────────────────────────────────────────────────────
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std(ddof=0)
    z = (series - rm) / rs
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
    z = (series - rmin) / (rmax - rmin)
    if inverse:
        z = 1 - z
    return (z * 100).clip(0, 100)

NORM_METHODS = {
    'zscore':     rolling_zscore,
    'minmax':     rolling_minmax,
    'percentile': rolling_percentile,
}

def best_r_over_windows(raw, inverse=True):
    """Najde nejlepší r, window a metodu přes všechny kombinace."""
    from scipy.stats import pearsonr
    best_r, best_w, best_method = 0, None, None
    overlap = raw.index.intersection(cnn.index)
    for method_name, norm_fn in NORM_METHODS.items():
        for w in WINDOWS:
            try:
                norm = norm_fn(raw, w, inverse=inverse)
                valid = overlap[norm.loc[overlap].notna() & cnn.loc[overlap].notna()]
                if len(valid) < 200:
                    continue
                r, _ = pearsonr(norm.loc[valid], cnn.loc[valid])
                if r > best_r:
                    best_r, best_w, best_method = r, w, method_name
            except Exception:
                continue
    return best_r, best_w, best_method

# ── ČÁST 5: Grid search ──────────────────────────────────────────────────────
print(f"\n── Výsledky grid search (best window+method per spread) ────────────────────")
print(f"  {'Spread':<45} {'r':>7}  {'w':>6}  {'Metoda':<12}  {'Od':>12}")
print(f"  {'-'*85}")

results = []

# A) FRED OAS spreads (přímé — jeden instrument)
for series_id, s in fred_data.items():
    try:
        raw = s.copy()
        r, w, method = best_r_over_windows(raw, inverse=True)
        if r == 0:
            continue
        covers = raw.index[0].year <= 1998
        marker = " ✓1998" if covers else "      "
        overlap = raw.index.intersection(cnn.index)
        n = len(overlap[cnn.loc[overlap].notna()])
        results.append({'label': series_id, 'r': r, 'w': w, 'method': method, 'n': n,
                        'start': raw.index[0].date(), 'covers': covers})
        print(f"  {series_id:<45} {r:>7.3f}  {w:>5}d  {method:<12}  {str(raw.index[0].date()):>12}{marker}")
    except Exception as e:
        pass

# B) FRED OAS diferenciály (junk - IG)
fred_pairs = [
    ('BAMLH0A0HYM2', 'BAMLC0A4CBBB', 'HY_OAS - BBB_OAS'),
    ('BAMLH0A1HYBB',  'BAMLC0A4CBBB', 'BB_OAS - BBB_OAS'),
    ('BAMLH0A0HYM2', 'BAMLC0A3CA',   'HY_OAS - A_OAS'),
    ('BAMLH0A1HYBB',  'BAMLC0A3CA',   'BB_OAS - A_OAS'),
]
for junk_id, ig_id, label in fred_pairs:
    if junk_id not in fred_data or ig_id not in fred_data:
        continue
    try:
        common = fred_data[junk_id].index.intersection(fred_data[ig_id].index)
        raw = (fred_data[junk_id].loc[common] - fred_data[ig_id].loc[common]).dropna()
        r, w, method = best_r_over_windows(raw, inverse=True)
        if r == 0:
            continue
        covers = raw.index[0].year <= 1998
        marker = " ✓1998" if covers else "      "
        overlap = raw.index.intersection(cnn.index)
        n = len(overlap[cnn.loc[overlap].notna()])
        results.append({'label': label, 'r': r, 'w': w, 'method': method, 'n': n,
                        'start': raw.index[0].date(), 'covers': covers})
        print(f"  {label:<45} {r:>7.3f}  {w:>5}d  {method:<12}  {str(raw.index[0].date()):>12}{marker}")
    except Exception as e:
        pass

# C) ETF price return diferenciály
etf_pairs = [
    ('HYG',   'LQD',   'HYG_ret - LQD_ret'),
    ('JNK',   'LQD',   'JNK_ret - LQD_ret'),
    ('VWEHX', 'VWESX', 'VWEHX_ret - VWESX_ret'),
    ('VWEHX', 'VBMFX', 'VWEHX_ret - VBMFX_ret'),
    ('HYG',   'VBMFX', 'HYG_ret - VBMFX_ret'),
]
for junk_t, ig_t, label in etf_pairs:
    if junk_t not in etf_data or ig_t not in etf_data:
        continue
    try:
        jr = etf_data[junk_t].pct_change(20) * 100
        ir = etf_data[ig_t].pct_change(20) * 100
        common = jr.index.intersection(ir.index)
        raw = (jr.loc[common] - ir.loc[common]).dropna()
        r, w, method = best_r_over_windows(raw, inverse=False)  # junk outperforms = greed = vysoké skóre
        if r == 0:
            continue
        covers = raw.index[0].year <= 1998
        marker = " ✓1998" if covers else "      "
        overlap = raw.index.intersection(cnn.index)
        n = len(overlap[cnn.loc[overlap].notna()])
        results.append({'label': label, 'r': r, 'w': w, 'method': method, 'n': n,
                        'start': raw.index[0].date(), 'covers': covers})
        print(f"  {label:<45} {r:>7.3f}  {w:>5}d  {method:<12}  {str(raw.index[0].date()):>12}{marker}")
    except Exception as e:
        pass

# ── Výsledek ─────────────────────────────────────────────────────────────────
results.sort(key=lambda x: x['r'], reverse=True)

if results:
    best = results[0]
    print(f"  --> NEJLEPŠÍ: {best['label']}  r={best['r']:.3f}  w={best['w']}d  {best['method']}  (od {best['start']})")

    print(f"\n  Top 5:")
    for res in results[:5]:
        covers = "✓1998" if res['covers'] else "     "
        print(f"    {res['label']:<45} r={res['r']:.3f}  w={res['w']:4d}d  {res['method']:<12}  {covers}")
