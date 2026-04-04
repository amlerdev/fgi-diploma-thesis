"""
stock_price_breadth.py
======================
Component #3: Stock Price Breadth
Indicator: NYSE Volume McClellan Summation Index (!VMCSUMNYA)
Normalization: Rolling Z-score, window=126d (~6 months)
Correlation with CNN FGI: r=0.816 (2011-2026, n=3826)
Data source: StockCharts QuoteBrain API (FREE) — data v CSV

Grid search: viz stock_price_breadth_grid.py
"""

from pathlib import Path
import requests
import pandas as pd
from datetime import datetime

_dir = Path(__file__).resolve().parent
BEST_WINDOW = 126  # ~6 měsíců

# ── ČÁST 1: Download / načtení dat ───────────────────────────────────────────
def download_vmcsumnya(save_path):
    """Stáhne !VMCSUMNYA z StockCharts QuoteBrain API."""
    url = "https://stockcharts.com/quotebrain/historyandquote/d"
    params = {
        'symbol': '!VMCSUMNYA',
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
    df = pd.DataFrame({'Date': dates, 'VMCSUMNYA': closes}).sort_values('Date').reset_index(drop=True)
    df.to_csv(save_path, index=False)
    print(f"  Stazeno {len(df)} dni  ({df['Date'].min().date()} -> {df['Date'].max().date()})")
    return df

nysi_csv = _dir / 'nysi_data.csv'

if nysi_csv.exists():
    nysi_df = pd.read_csv(nysi_csv, parse_dates=['Date'])
    print(f"Nacteno z CSV: {len(nysi_df)} zaznamu")
else:
    print("Stahuji !VMCSUMNYA z StockCharts QuoteBrain API...")
    nysi_df = download_vmcsumnya(nysi_csv)

# ── ČÁST 2: Výpočet surového signálu ─────────────────────────────────────────
nysi_df['Date'] = nysi_df['Date'].dt.normalize()
raw_signal = nysi_df.groupby('Date')['VMCSUMNYA'].last().dropna()
raw_signal.name = 'VMCSUMNYA'

print(f"!VMCSUMNYA: {raw_signal.index[0].date()} → {raw_signal.index[-1].date()}  ({len(raw_signal)} dní)")

# ── ČÁST 3: Normalize + Save ──────────────────────────────────────────────────
def rolling_zscore(series, window):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std(ddof=0)
    return ((series - rm) / rs * 25 + 50).clip(0, 100)

nysi_norm = rolling_zscore(raw_signal, BEST_WINDOW)
nysi_norm.name = 'Breadth_Norm'

out = nysi_norm.dropna().reset_index()
out.columns = ['Date', 'Breadth_Norm']
output_file = _dir / 'stock_price_breadth_normalized.csv'
out.to_csv(output_file, index=False)
print(f"\nUlozeno: {output_file}")
print(f"Radku: {len(out)}  ({out['Date'].iloc[0].date()} -> {out['Date'].iloc[-1].date()})")
print(f"Mean: {out['Breadth_Norm'].mean():.1f}  (ocekavano ~50)")
