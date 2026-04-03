"""
stock_price_strength.py
=======================
Component #2: Stock Price Strength
Formula: 20-day MA of (NYHGH - NYLOW) net difference
  MA20 smoothing odstraní denní šum, zachová trend
Normalization: Rolling Z-score, window=126d (~6 měsíců)
Correlation with CNN FGI: r=0.760 (2011-2026, n=3826)
Data source: StockCharts QuoteBrain API (FREE) — data v CSV, od 1980

Grid search: viz stock_price_strength_grid.py
Změna 2026-04-02: Net_MA20+zscore 126d (r=0.760) místo Ratio_MA20+zscore 126d (r=0.696)
"""

from pathlib import Path
import requests
import pandas as pd
import numpy as np
from datetime import datetime

_dir = Path(__file__).resolve().parent
BEST_WINDOW = 126   # ~6 měsíců — nejlepší korelace r=0.760 (Net_MA20 + zscore)
MA_WINDOW   = 20    # vyhlazení denního šumu

# ── ČÁST 1: Download / načtení dat ───────────────────────────────────────────
def download_quotebrain(symbol, save_path):
    url = "https://stockcharts.com/quotebrain/historyandquote/d"
    params = {
        'symbol': symbol,
        'start': '19800101',
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
    df.to_csv(save_path, index=False)
    print(f"  Stazeno {len(df)} dni  ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    return df

nyhgh_csv = _dir / 'nyhgh_1980_2026.csv'
nylow_csv = _dir / 'nylow_1980_2026.csv'

if nyhgh_csv.exists() and nylow_csv.exists():
    highs = pd.read_csv(nyhgh_csv, parse_dates=['Date'])
    lows  = pd.read_csv(nylow_csv, parse_dates=['Date'])
    print(f"Nacteno z CSV: NYHGH {len(highs)} dni, NYLOW {len(lows)} dni")
else:
    print("Stahuji $NYHGH z StockCharts...")
    highs = download_quotebrain('$NYHGH', nyhgh_csv)
    print("Stahuji $NYLOW z StockCharts...")
    lows  = download_quotebrain('$NYLOW', nylow_csv)

# ── ČÁST 2: Výpočet surového signálu ─────────────────────────────────────────
df = pd.merge(
    highs[['Date', 'Close']].rename(columns={'Close': 'NYHGH'}),
    lows[['Date', 'Close']].rename(columns={'Close': 'NYLOW'}),
    on='Date', how='inner'
).set_index('Date').sort_index()

df.index = df.index.normalize()

raw_signal = (df['NYHGH'] - df['NYLOW']).rolling(MA_WINDOW).mean().dropna()
raw_signal.name = 'SPS_Net_MA20'

print(f"Raw signal: {raw_signal.index[0].date()} → {raw_signal.index[-1].date()}  ({len(raw_signal)} dní)")

# ── ČÁST 3: Normalize + Save ──────────────────────────────────────────────────
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std()
    z  = (series - rm) / rs
    if inverse:
        z = -z
    return (z * 25 + 50).clip(0, 100)

sps_norm = rolling_zscore(raw_signal, BEST_WINDOW)
sps_norm.name = 'Strength_Norm'

out = sps_norm.dropna().reset_index()
out.columns = ['Date', 'Strength_Norm']
output_file = _dir / 'stock_price_strength_normalized.csv'
out.to_csv(output_file, index=False)
print(f"\nUlozeno: {output_file}")
print(f"Radku: {len(out)}  ({out['Date'].iloc[0].date()} -> {out['Date'].iloc[-1].date()})")
print(f"Mean: {out['Strength_Norm'].mean():.1f}  (ocekavano ~50)")
