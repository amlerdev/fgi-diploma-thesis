"""
market_momentum.py
==================
Component #1: Market Momentum
Formula: SP500 - MA125  (absolutní rozdíl od 125-denního průměru)
CNN metodika: "S&P 500 above/below its 125-day moving average" — přesný vzorec nespecifikován
Normalization: Rolling Z-score, window=63d (~3 months)
Correlation with CNN FGI: r=0.832 (grid search, diff+zscore 63d)
Data source: Yahoo Finance (^GSPC) — FREE

Grid search: viz market_momentum_grid.py
"""

from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

_dir = Path(__file__).resolve().parent
BEST_WINDOW = 63   # ~3 měsíce

# ── ČÁST 1: Download dat ──────────────────────────────────────────────────────
print("Stahuji S&P 500 (^GSPC) od 1995...")
sp500_raw = yf.download('^GSPC', start='1995-01-01', end='2026-03-20', progress=False)['Close']
sp500_raw.index = pd.to_datetime(sp500_raw.index)
sp500 = sp500_raw.iloc[:, 0] if isinstance(sp500_raw, pd.DataFrame) else sp500_raw

print(f"^GSPC: {sp500.index[0].date()} → {sp500.index[-1].date()}  ({len(sp500)} dní)")

# ── ČÁST 2: Výpočet surového signálu ─────────────────────────────────────────
ma125 = sp500.rolling(125).mean()
raw_signal = (sp500 - ma125).dropna()
raw_signal.name = 'Momentum_Diff'

# ── ČÁST 3: Normalize + Save ──────────────────────────────────────────────────
def rolling_zscore(series, window):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std(ddof=0)
    return ((series - rm) / rs * 25 + 50).clip(0, 100)

mom_norm = rolling_zscore(raw_signal, BEST_WINDOW)
mom_norm.name = 'Mom_Norm'

out = mom_norm.dropna().reset_index()
out.columns = ['Date', 'Mom_Norm']
output_file = _dir / 'market_momentum_normalized.csv'
out.to_csv(output_file, index=False)
print(f"\nUlozeno: {output_file}")
print(f"Radku: {len(out)}  ({out['Date'].iloc[0].date()} -> {out['Date'].iloc[-1].date()})")
print(f"Mean: {out['Mom_Norm'].mean():.1f}  (ocekavano ~50)")
