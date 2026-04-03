"""
put_call_grid.py
================
Grid search: různé put/call indikátory × MA okna × normalizační metody × okna
CNN říká: "Put/Call Ratio — bearish vs bullish options"
Vysoký ratio = více put opcí = strach → inverse normalizace

Indikátory:
  $CPC   — CBOE Total Put/Call Ratio (equity + index options), dostupný od ~1995

Celkem: 1 indikátor × 5 MA oken × 3 metody × 12 oken = 180 kombinací
Výsledky se NEUKLÁDAJÍ — jen výpis do konzole.
"""

from pathlib import Path
import requests
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

_dir = Path(__file__).resolve().parent
CNN_CSV = _dir / '../../data/fear_greed_historical.csv'

WINDOWS = [21, 42, 63, 126, 252, 378, 504, 630, 756, 1008, 1260, 1512]
MA_WINDOWS = [1, 3, 5, 10, 20]  # 1 = žádná MA (raw)

# ── Download ──────────────────────────────────────────────────────────────────
def download_quotebrain(symbol, save_path):
    url = "https://stockcharts.com/quotebrain/historyandquote/d"
    params = {
        'symbol': symbol,
        'start': '19950101',
        'end': '20260320',
        'windowid': 'main', 'chartid': 'main',
        'fromde': 'false', 'numCharts': '1', 'numWindows': '1',
        'appv': '1.91', 'z': 'true', 'extended': 'true',
    }
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Accept': '*/*',
        'Referer': 'https://stockcharts.com/acp/',
    }
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    intervals = data['history']['intervals']
    dates  = [pd.to_datetime(i['start']['time']) for i in intervals]
    closes = [i['close'] for i in intervals]
    df = pd.DataFrame({'Date': dates, 'Close': closes}).sort_values('Date').reset_index(drop=True)
    df['Date'] = df['Date'].dt.normalize()
    df.to_csv(save_path, index=False)
    print(f"  Stazeno {len(df)} dni  ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    return df

# ── Načtení / download indikátoru ─────────────────────────────────────────────
INDICATORS = {
    '$CPC': 'put_call_cpc_1995_2026.csv',
}

raw_data = {}
print("Načítám/stahuji indikátory...")
for symbol, fname in INDICATORS.items():
    csv_path = _dir / fname
    try:
        if csv_path.exists():
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
            s = df.set_index('Date')['Close'].dropna()
        else:
            print(f"  Stahuji {symbol}...")
            df = download_quotebrain(symbol, csv_path)
            s = df.set_index('Date')['Close'].dropna()
        raw_data[symbol] = s
        print(f"  ✓ {symbol:<8} {s.index[0].date()} → {s.index[-1].date()}  ({len(s)} dní)")
    except Exception as e:
        print(f"  ✗ {symbol:<8} CHYBA: {e}")

# ── CNN FGI ───────────────────────────────────────────────────────────────────
cnn = pd.read_csv(CNN_CSV, parse_dates=['date'], index_col='date')['value']
cnn.index.name = 'Date'

# ── Normalizace ───────────────────────────────────────────────────────────────
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std(ddof=0)
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
print(f"\n── Grid search (best MA+method+window per indikátor) ───────────────────")
print(f"  {'Indikátor':<8} {'MA':>4}  {'r':>7}  {'w':>6}  {'Metoda':<12}  {'Od':>12}")
print(f"  {'-'*60}")

results = []

for symbol, series in raw_data.items():
    best_r, best_ma, best_w, best_method = 0, None, None, None
    overlap = series.index.intersection(cnn.index)

    for ma in MA_WINDOWS:
        raw = series.rolling(ma).mean().dropna() if ma > 1 else series.copy()
        raw_overlap = raw.index.intersection(cnn.index)

        for method_name, norm_fn in NORM_METHODS.items():
            for w in WINDOWS:
                try:
                    norm = norm_fn(raw, w, inverse=True)
                    valid = raw_overlap[norm.loc[raw_overlap].notna() & cnn.loc[raw_overlap].notna()]
                    if len(valid) < 200:
                        continue
                    r, _ = pearsonr(norm.loc[valid], cnn.loc[valid])
                    if r > best_r:
                        best_r, best_ma, best_w, best_method = r, ma, w, method_name
                except Exception:
                    continue

    if best_r > 0:
        covers = series.index[0].year <= 1998
        marker = " ✓1998" if covers else "      "
        ma_str = f"{best_ma}d" if best_ma > 1 else "raw"
        results.append({
            'symbol': symbol, 'r': best_r, 'ma': best_ma, 'w': best_w,
            'method': best_method, 'start': series.index[0].date(), 'covers': covers
        })
        print(f"  {symbol:<8} {ma_str:>4}  {best_r:>7.3f}  {best_w:>5}d  {best_method:<12}  {str(series.index[0].date()):>12}{marker}")

# ── Výsledek ──────────────────────────────────────────────────────────────────
results.sort(key=lambda x: x['r'], reverse=True)

if results:
    best = results[0]
    ma_str = f"MA{best['ma']}d" if best['ma'] > 1 else "raw"
    print(f"\n  --> NEJLEPŠÍ: {best['symbol']}  {ma_str}  r={best['r']:.3f}  w={best['w']}d  {best['method']}  (od {best['start']})")
