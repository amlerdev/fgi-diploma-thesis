"""
put_call_ratio.py
=================
Component #4: Put/Call Ratio
Indicator: CBOE Total Put/Call Ratio ($CPC), 5-day MA, INVERSE normalization
Normalization: Rolling Z-score (inverse), window=252d (~1 year)
Correlation with CNN FGI: r=0.641 (2011-2026, n=3826)
Data source: StockCharts QuoteBrain API (FREE) — data v CSV

Grid search: viz put_call_grid.py
"""

from pathlib import Path
import requests
import pandas as pd
from datetime import datetime

_dir = Path(__file__).resolve().parent
BEST_WINDOW = 252  # ~1 rok — nejlepší korelace r=0.641

# ── ČÁST 1: Download / načtení dat ───────────────────────────────────────────
def download_quotebrain(symbol, save_path):
    """Stáhne historická data z StockCharts QuoteBrain API."""
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
    df = pd.DataFrame({'Date': dates, 'CPC': closes}).sort_values('Date').reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"  Stazeno {len(df)} dni  ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    return df

pc_csv = _dir / 'put_call_ratio_1995_2026.csv'

if pc_csv.exists():
    pc_df = pd.read_csv(pc_csv, parse_dates=['Date'])
    print(f"Nacteno z CSV: {len(pc_df)} zaznamu")
else:
    print("Stahuji $CPC z StockCharts QuoteBrain API...")
    pc_df = download_quotebrain('$CPC', pc_csv)

# ── ČÁST 2: Výpočet surového signálu ─────────────────────────────────────────
if 'PC_Ratio_5D' in pc_df.columns:
    pc_df = pc_df.set_index('Date')[['PC_Ratio_5D']].dropna()
    raw_signal = pc_df['PC_Ratio_5D']
elif 'CPC' in pc_df.columns:
    pc_df = pc_df.set_index('Date')
    raw_signal = pc_df['CPC'].rolling(5).mean().dropna()
    pc_df['PC_Ratio_5D'] = raw_signal
    pc_df.reset_index().to_csv(pc_csv, index=False)
else:
    raise ValueError(f"Neocekavane sloupce v {pc_csv}: {pc_df.columns.tolist()}")

raw_signal = raw_signal.dropna()
print(f"$CPC: {raw_signal.index[0].date()} → {raw_signal.index[-1].date()}  ({len(raw_signal)} dní)")

# ── ČÁST 3: Normalize + Save ──────────────────────────────────────────────────
# Vysoký put/call = strach = nízké skóre → inverse=True
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std(ddof=0)
    z  = (series - rm) / rs
    if inverse:
        z = -z
    return (z * 25 + 50).clip(0, 100)

pc_norm = rolling_zscore(raw_signal, BEST_WINDOW, inverse=True)
pc_norm.name = 'PutCall_Norm'

out = pc_norm.dropna().reset_index()
out.columns = ['Date', 'PutCall_Norm']
output_file = _dir / 'put_call_ratio_normalized.csv'
out.to_csv(output_file, index=False)
print(f"\nUlozeno: {output_file}")
print(f"Radku: {len(out)}  ({out['Date'].iloc[0].date()} -> {out['Date'].iloc[-1].date()})")
print(f"Mean: {out['PutCall_Norm'].mean():.1f}  (ocekavano ~50)")
