"""
market_volatility_vix.py
========================
Component #5: Market Volatility (VIX)
Indicator: VIX / MA50 ratio, INVERSE normalization
Normalization: Rolling Z-score (inverse), window=1512d (~6 years)
Correlation with CNN FGI: r=0.644 (2011-2026, n=3826)
Data source: Yahoo Finance (^VIX) — FREE

Grid search: viz market_volatility_vix_grid.py
"""

from pathlib import Path
import pandas as pd
import yfinance as yf

_dir = Path(__file__).resolve().parent
BEST_WINDOW = 1512  # ~6 let — nejlepší korelace r=0.644; VIX spiky potřebují dlouhý kontext

# ── ČÁST 1: Download dat ──────────────────────────────────────────────────────
print("Stahuji VIX (^VIX) z Yahoo Finance...")
vix_dl  = yf.download('^VIX', start='1990-01-01', end='2026-03-20', progress=False)
vix_raw = vix_dl['Close'].iloc[:, 0] if isinstance(vix_dl['Close'], pd.DataFrame) else vix_dl['Close']
vix_raw.index = pd.to_datetime(vix_raw.index)

print(f"^VIX: {vix_raw.index[0].date()} → {vix_raw.index[-1].date()}  ({len(vix_raw)} dní)")

# ── ČÁST 2: Výpočet surového signálu ─────────────────────────────────────────
vix_ma50   = vix_raw.rolling(50).mean()
raw_signal = (vix_raw / vix_ma50).dropna()
raw_signal.name = 'VIX_Ratio'

# ── ČÁST 3: Normalize + Save ──────────────────────────────────────────────────
# Vysoký VIX = strach = nízké skóre → inverse=True
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std()
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
