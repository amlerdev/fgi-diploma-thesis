"""
junk_bond_demand.py
===================
Component #7: Junk Bond Demand
Indicator: ICE BofA US High Yield OAS (BAMLH0A0HYM2), INVERSE normalization
  Wider spread = more fear, tighter spread = more greed
Normalization: Rolling Z-score (inverse), window=63d (~3 měsíce)
Correlation with CNN FGI: r=0.768 (grid search 2026-04-03)
Data source: FRED (BAMLH0A0HYM2) — FREE, od 1996

Grid search: viz junk_bond_grid.py
"""

from pathlib import Path
import pandas as pd

_dir = Path(__file__).resolve().parent
BEST_WINDOW = 63  # ~3 měsíce — nejlepší korelace r=0.768

# ── ČÁST 1: Download / načtení dat ───────────────────────────────────────────
hy_csv = _dir / 'hy_oas_data.csv'

if hy_csv.exists():
    jb_df = pd.read_csv(hy_csv, parse_dates=['Date'], index_col='Date')
    print(f"Nacteno z CSV: {len(jb_df)} zaznamu  ({jb_df.index[0].date()} → {jb_df.index[-1].date()})")
else:
    print("Stahuji ICE BofA HY OAS (BAMLH0A0HYM2) z FRED...")
    import pandas_datareader.data as web
    hy = web.DataReader('BAMLH0A0HYM2', 'fred', '1996-01-01', '2026-12-31')
    hy.columns = ['HY_OAS']
    hy.index.name = 'Date'
    hy = hy.dropna()
    hy.to_csv(hy_csv)
    jb_df = hy
    print(f"  Ulozeno {len(jb_df)} zaznamu")

# ── ČÁST 2: Výpočet surového signálu ─────────────────────────────────────────
# HY OAS: širší spread = vyšší rizikové prémium = strach = nízké F&G skóre
raw_signal = jb_df['HY_OAS'].dropna()
raw_signal.name = 'HY_OAS'

print(f"HY OAS: {raw_signal.index[0].date()} → {raw_signal.index[-1].date()}  ({len(raw_signal)} dní)")
print(f"Range: {raw_signal.min():.2f}% — {raw_signal.max():.2f}%")

# ── ČÁST 3: Normalize + Save ──────────────────────────────────────────────────
# Širší HY spread = strach = nízké skóre → inverse=True
def rolling_zscore(series, window, inverse=False):
    rm = series.rolling(window, min_periods=window // 2).mean()
    rs = series.rolling(window, min_periods=window // 2).std(ddof=0)
    z  = (series - rm) / rs
    if inverse:
        z = -z
    return (z * 25 + 50).clip(0, 100)

jb_norm = rolling_zscore(raw_signal, BEST_WINDOW, inverse=True)
jb_norm.name = 'JunkBond_Norm'

out = jb_norm.dropna().reset_index()
out.columns = ['Date', 'JunkBond_Norm']
output_file = _dir / 'junk_bond_normalized.csv'
out.to_csv(output_file, index=False)
print(f"\nUlozeno: {output_file}")
print(f"Radku: {len(out)}  ({out['Date'].iloc[0].date()} -> {out['Date'].iloc[-1].date()})")
print(f"Mean: {out['JunkBond_Norm'].mean():.1f}  (ocekavano ~50)")
