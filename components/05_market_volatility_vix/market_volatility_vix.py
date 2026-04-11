"""
market_volatility_vix.py
========================
Component #5: Market Volatility (VIX)
Indicator: VIX / MA50 ratio, INVERSE normalization
Normalization: Rolling Z-score (inverse), window=1512d (~6 years)
Correlation with CNN FGI: r=0.653 (2011-2026, n=3826)
Data source: Yahoo Finance (^VIX) — FREE

Grid search: viz market_volatility_vix_grid.py
"""

from pathlib import Path
import pandas as pd

from vix_data import END_DATE, load_vix_series

_dir = Path(__file__).resolve().parent
BEST_WINDOW = 1512  # ~6 let — nejlepší korelace r=0.653; VIX spiky potřebují dlouhý kontext
VIX_CSV = _dir / 'vix_daily_data.csv'

# ── ČÁST 1: Download dat ──────────────────────────────────────────────────────
print(f"Stahuji VIX (^VIX) z Yahoo Finance a aktualizuji {VIX_CSV.name}...")
vix_raw = load_vix_series(VIX_CSV, refresh=True)

print(f"^VIX: {vix_raw.index[0].date()} → {vix_raw.index[-1].date()}  ({len(vix_raw)} dní)")
print(f"Raw CSV: {VIX_CSV}  (do {END_DATE})")

# ── ČÁST 2: Výpočet surového signálu ─────────────────────────────────────────
vix_ma50   = vix_raw.rolling(50).mean()
raw_signal = (vix_raw / vix_ma50).dropna()
raw_signal.name = 'VIX_Ratio'

# ── ČÁST 3: Normalize + Save ──────────────────────────────────────────────────
# Vysoký VIX = strach = nízké skóre → inverse=True
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std(ddof=0)
    z  = (series - rm) / rs
    if inverse:
        z = -z
    return (z * 25 + 50).clip(0, 100)

vix_norm = rolling_zscore(raw_signal, BEST_WINDOW, inverse=True)
vix_norm.name = 'VIX_Norm'

out = vix_norm.dropna().reset_index()
out.columns = ['Date', 'VIX_Norm']
output_file = _dir / 'market_volatility_vix_normalized.csv'
out.to_csv(output_file, index=False)
print(f"\nUlozeno: {output_file}")
print(f"Radku: {len(out)}  ({out['Date'].iloc[0].date()} -> {out['Date'].iloc[-1].date()})")
print(f"Mean: {out['VIX_Norm'].mean():.1f}  (ocekavano ~50)")
