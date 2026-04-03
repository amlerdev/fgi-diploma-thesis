"""
safe_haven_demand.py
====================
Component #6: Safe Haven Demand
Formula: ^IXIC 20-day return MINUS VFITX 20-day return
  CNN metodika: "Difference in 20-day stock and bond returns"
  Stock = ^IXIC (NASDAQ Composite)
  Bond  = VFITX (Vanguard Intermediate-Term Treasury Fund, 5-10yr, od 1997)
Normalization: Rolling Z-score, window=504d (~2 roky)
Correlation with CNN FGI: r=0.730 (grid search, zscore top1)
Data sources: Yahoo Finance (^IXIC + VFITX) — FREE, od 1997

Změna 2026-04-03: ^GSPC+VFISX+zscore (r=0.672) → ^IXIC+VFITX+zscore (r=0.730)

Validace raw signálu vs CNN referenčních hodnot:
  Nov 25, 2025: -2.45%  |  Dec 31, 2025: +0.50%  |  Jan 20, 2026: +0.29%
  Jan 30, 2026: +1.43%  |  Feb 13, 2026: -2.47%  |  Mar 10, 2026: -2.05%

Grid search: viz safe_haven_grid.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

_dir = Path(__file__).resolve().parent
BEST_WINDOW = 504  # ~2 roky — nejlepší korelace r=0.730 (zscore)

# ── ČÁST 1: Download dat ──────────────────────────────────────────────────────
print("Stahuji ^IXIC (NASDAQ Composite)...")
spy_raw = yf.download('^IXIC', start='1997-01-01', end='2026-03-20', progress=False)['Close']
spy_raw.index = pd.to_datetime(spy_raw.index)
spy = spy_raw.iloc[:, 0] if isinstance(spy_raw, pd.DataFrame) else spy_raw

print("Stahuji VFITX (Vanguard Intermediate-Term Treasury, 5-10yr)...")
vustx_raw = yf.download('VFITX', start='1997-01-01', end='2026-03-20', progress=False)['Close']
vustx_raw.index = pd.to_datetime(vustx_raw.index)
vustx = vustx_raw.iloc[:, 0] if isinstance(vustx_raw, pd.DataFrame) else vustx_raw

print(f"^IXIC: {spy.index[0].date()} → {spy.index[-1].date()}  ({len(spy)} dní)")
print(f"VFITX: {vustx.index[0].date()} → {vustx.index[-1].date()}  ({len(vustx)} dní)")

# ── ČÁST 2: Výpočet surového signálu ─────────────────────────────────────────
spy_ret   = spy.pct_change(periods=20) * 100
vustx_ret = vustx.pct_change(periods=20) * 100

common     = spy_ret.index.intersection(vustx_ret.index)
raw_signal = (spy_ret.loc[common] - vustx_ret.loc[common]).dropna()
raw_signal.name = 'SafeHaven_Raw'

print(f"\nRaw signal: {raw_signal.index[0].date()} → {raw_signal.index[-1].date()}")
print(f"Range: {raw_signal.min():.2f}% to {raw_signal.max():.2f}%")

# ── ČÁST 3: Normalize + Save ──────────────────────────────────────────────────
# Akcie > dluhopisy = greed = vysoké skóre → NO inverse
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std()
    z  = (series - rm) / rs
    if inverse:
        z = -z
    return (z * 25 + 50).clip(0, 100)

sh_norm = rolling_zscore(raw_signal, BEST_WINDOW, inverse=False)
sh_norm.name = 'SafeHaven_Norm'

out = sh_norm.dropna().reset_index()
out.columns = ['Date', 'SafeHaven_Norm']
output_file = _dir / 'safe_haven_normalized.csv'
out.to_csv(output_file, index=False)
print(f"\nUlozeno: {output_file}")
print(f"Radku: {len(out)}  ({out['Date'].iloc[0].date()} -> {out['Date'].iloc[-1].date()})")
print(f"Mean: {out['SafeHaven_Norm'].mean():.1f}  (ocekavano ~50)")
