"""
03_build_index.py
=================
Krok 3/3: Aplikuje váhy a sestaví finální index se dvěma variantami.

  FGI_Equal  — průměr 7 komponent (1/7 každá)
  FGI_OLS    — OLS koeficienty + intercept, ořezáno na [0, 100]

Vstup:   code/data/fgi_dataset.csv
         code/data/fgi_weights.csv
Výstup:  code/data/fgi_index_final.csv

Author: Petr Amler (AML0005)
"""

from pathlib import Path
import pandas as pd

INDEX_DIR = Path(__file__).resolve().parent
INPUT_DS  = INDEX_DIR / 'fgi_dataset.csv'
INPUT_W   = INDEX_DIR / 'fgi_weights.csv'
OUTPUT    = INDEX_DIR / 'fgi_index_final.csv'

COMP_COLS = ['Mom_Norm', 'Strength_Norm', 'Breadth_Norm',
             'PutCall_Norm', 'VIX_Norm', 'SafeHaven_Norm', 'JunkBond_Norm']

print("=" * 65)
print("KROK 3: SESTAVENÍ INDEXU")
print("=" * 65)

# ── Načtení ───────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_DS, parse_dates=['Date'], index_col='Date')
w  = pd.read_csv(INPUT_W,  index_col='method')

print(f"\nDataset: {df.shape[0]} řádků  ({df.index[0].date()} → {df.index[-1].date()})")

X = df[COMP_COLS]

# ── FGI_Equal ──────────────────────────────────────────────────────────────────
equal_weights = w.loc['Equal', COMP_COLS].values
df['FGI_Equal'] = X.dot(equal_weights)

# ── FGI_OLS ────────────────────────────────────────────────────────────────────
ols_weights   = w.loc['OLS', COMP_COLS].values
ols_intercept = w.loc['OLS', 'intercept']
df['FGI_OLS']  = (X.dot(ols_weights) + ols_intercept).clip(0, 100)

# ── Přehled ───────────────────────────────────────────────────────────────────
print("\nPlatné hodnoty per index:")
for col in ['FGI_Equal', 'FGI_OLS']:
    valid = df[col].dropna()
    print(f"   {col:<12} {len(valid):>5} řádků  "
          f"({valid.index[0].date()} → {valid.index[-1].date()})  "
          f"mean={valid.mean():.1f}  min={valid.min():.1f}  max={valid.max():.1f}")

overlap_mask = df['CNN_FearGreed'].notna()
print(f"\nOverlap s CNN ({overlap_mask.sum()} dní):")
print(f"   {'Varianta':<12} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
print(f"   {'-'*45}")
print(f"   {'CNN':<12} {df.loc[overlap_mask,'CNN_FearGreed'].mean():>8.1f} "
      f"{df.loc[overlap_mask,'CNN_FearGreed'].std():>8.1f} "
      f"{df.loc[overlap_mask,'CNN_FearGreed'].min():>8.1f} "
      f"{df.loc[overlap_mask,'CNN_FearGreed'].max():>8.1f}")
for col in ['FGI_Equal', 'FGI_OLS']:
    sub = df.loc[overlap_mask, col].dropna()
    print(f"   {col:<12} {sub.mean():>8.1f} {sub.std():>8.1f} "
          f"{sub.min():>8.1f} {sub.max():>8.1f}")

# ── Uložení ───────────────────────────────────────────────────────────────────
out_cols = ['SP500_Close'] + COMP_COLS + ['FGI_Equal', 'FGI_OLS', 'CNN_FearGreed']
df[out_cols].to_csv(OUTPUT)

print(f"\nUloženo: {OUTPUT}")
print(f"Sloupce: {out_cols}")
print("\nHotovo — spust validate_index.py")
