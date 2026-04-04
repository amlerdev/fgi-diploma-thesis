"""
01_merge_components.py
======================
Krok 1/3: Merguje 7 normalizovaných komponent s S&P 500 a CNN daty.

Vstupy:  code/components/*/  (*_normalized.csv)
         code/data/fear_greed_historical.csv
         Yahoo Finance (^SP500TR)
Výstup:  code/data/fgi_dataset.csv

Author: Petr Amler (AML0005)
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

BASE      = Path(__file__).resolve().parent.parent
COMP_DIR  = BASE / 'components'
DATA_DIR  = BASE / 'data'
INDEX_DIR = Path(__file__).resolve().parent
OUTPUT    = INDEX_DIR / 'fgi_dataset.csv'

START_DATE = '1998-01-01'
END_DATE   = '2026-12-31'

COMPONENTS = [
    ('01_market_momentum',       'market_momentum_normalized.csv',       'Mom_Norm'),
    ('02_stock_price_strength',  'stock_price_strength_normalized.csv',  'Strength_Norm'),
    ('03_stock_price_breadth',   'stock_price_breadth_normalized.csv',   'Breadth_Norm'),
    ('04_put_call_ratio',        'put_call_ratio_normalized.csv',        'PutCall_Norm'),
    ('05_market_volatility_vix', 'market_volatility_vix_normalized.csv', 'VIX_Norm'),
    ('06_safe_haven_demand',     'safe_haven_normalized.csv',            'SafeHaven_Norm'),
    ('07_junk_bond_demand',      'junk_bond_normalized.csv',             'JunkBond_Norm'),
]

print("=" * 65)
print("KROK 1: MERGE KOMPONENT")
print("=" * 65)

# ── S&P 500 ───────────────────────────────────────────────────────────────────
print("\n[1/9] Stahuji S&P 500 Total Return (^SP500TR)...")
sp_raw = yf.download('^SP500TR', start='1997-01-01', end=END_DATE, progress=False)['Close']
sp_raw.index = pd.to_datetime(sp_raw.index)
sp500 = sp_raw.iloc[:, 0] if isinstance(sp_raw, pd.DataFrame) else sp_raw
sp500.name = 'SP500_Close'
print(f"   {len(sp500)} dní  ({sp500.index[0].date()} → {sp500.index[-1].date()})")

df = pd.DataFrame(sp500)
df = df.loc[START_DATE:]

# ── 7 komponent ───────────────────────────────────────────────────────────────
for i, (folder, filename, col) in enumerate(COMPONENTS, start=2):
    path = COMP_DIR / folder / filename
    print(f"\n[{i}/9] {col}  ←  {folder}/{filename}")
    series = pd.read_csv(path, parse_dates=['Date'], index_col='Date')[col]
    series.index = series.index.normalize()
    df = df.join(series, how='left')
    valid = series.dropna()
    print(f"   {len(valid)} platných hodnot  ({valid.index[0].date()} → {valid.index[-1].date()})")

# ── CNN Fear & Greed ──────────────────────────────────────────────────────────
print("\n[9/9] CNN Fear & Greed historical...")
cnn_path = DATA_DIR / 'fear_greed_historical.csv'
cnn = pd.read_csv(cnn_path, parse_dates=['date'], index_col='date')['value']
cnn.index.name = 'Date'
df['CNN_FearGreed'] = cnn
cnn_valid = cnn.dropna()
print(f"   {len(cnn_valid)} platných hodnot  ({cnn_valid.index[0].date()} → {cnn_valid.index[-1].date()})")

# ── Uložení ───────────────────────────────────────────────────────────────────
df.index.name = 'Date'
df.to_csv(OUTPUT)

print("\n" + "=" * 65)
print(f"Uloženo: {OUTPUT}")
print(f"Shape: {df.shape}")
print(f"Období: {df.index[0].date()} → {df.index[-1].date()}")

COMP_COLS = [c[2] for c in COMPONENTS]
print("\nPlatné hodnoty per sloupec:")
for col in ['SP500_Close'] + COMP_COLS + ['CNN_FearGreed']:
    n     = df[col].notna().sum()
    first = df[col].dropna().index[0].date() if n > 0 else 'N/A'
    print(f"   {col:<22} {n:>5} řádků  (od {first})")

print("\nHotovo — spusť 02_compute_weights.py")
