"""
02_compute_weights.py
=====================
Krok 2/3: Výpočet vah pro dvě varianty indexu.

  Equal   — 1/7 pro každou komponentu (CNN metodika)
  OLS     — klasická OLS regrese (CNN FGI ~ 7 komponent), celý overlap 2011–2026

Vstup:   code/data/fgi_dataset.csv
Výstup:  code/data/fgi_weights.csv

Author: Petr Amler (AML0005)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm

INDEX_DIR = Path(__file__).resolve().parent
INPUT     = INDEX_DIR / 'fgi_dataset.csv'
OUTPUT    = INDEX_DIR / 'fgi_weights.csv'

COMP_COLS = ['Mom_Norm', 'Strength_Norm', 'Breadth_Norm',
             'PutCall_Norm', 'VIX_Norm', 'SafeHaven_Norm', 'JunkBond_Norm']

print("=" * 65)
print("KROK 2: VÝPOČET VAH")
print("=" * 65)

# ── Načtení dat ───────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT, parse_dates=['Date'], index_col='Date')

# Pouze řádky kde máme CNN i všechny komponenty (overlap 2011–2026)
mask   = df['CNN_FearGreed'].notna() & df[COMP_COLS].notna().all(axis=1)
df_fit = df[mask].copy()

print(f"\nFitovací období: {df_fit.index[0].date()} → {df_fit.index[-1].date()}")
print(f"Počet dní: {len(df_fit):,}")

X = df_fit[COMP_COLS].values
y = df_fit['CNN_FearGreed'].values

# ── 1) Equal weights ──────────────────────────────────────────────────────────
print("\n[1/2] Equal weights (1/7)...")
equal_w = np.full(len(COMP_COLS), 1 / len(COMP_COLS))
print(f"   Váha každé komponenty: {equal_w[0]:.6f}")

# ── 2) OLS ────────────────────────────────────────────────────────────────────
print("\n[2/2] OLS regrese...")
X_ols  = sm.add_constant(X)
model  = sm.OLS(y, X_ols).fit()
ols_w  = model.params[1:]   # bez interceptu
ols_b0 = model.params[0]    # intercept

print(f"\n   OLS výsledky:")
print(f"   {'Komponenta':<22} {'Koeficient':>12} {'p-value':>10} {'Signif.':>8}")
print(f"   {'-'*55}")
for name, coef, pval in zip(COMP_COLS, model.params[1:], model.pvalues[1:]):
    sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
    print(f"   {name:<22} {coef:>12.6f} {pval:>10.4f} {sig:>8}")
print(f"   {'Intercept':<22} {ols_b0:>12.6f} {model.pvalues[0]:>10.4f}")
print(f"\n   R² = {model.rsquared:.4f}  |  Adj. R² = {model.rsquared_adj:.4f}")
print("\n" + model.summary().as_text())

# ── Tabulka vah ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SROVNÁNÍ VAH")
print("=" * 65)
print(f"\n   {'Komponenta':<22} {'Equal':>10} {'OLS':>10}")
print(f"   {'-'*44}")
for name, ew, ow in zip(COMP_COLS, equal_w, ols_w):
    print(f"   {name:<22} {ew:>10.4f} {ow:>10.4f}")
print(f"   {'Intercept':<22} {'0.0000':>10} {ols_b0:>10.4f}")

# ── Uložení ───────────────────────────────────────────────────────────────────
rows = []
rows.append({'method': 'Equal', **dict(zip(COMP_COLS, equal_w)), 'intercept': 0.0})
rows.append({'method': 'OLS',   **dict(zip(COMP_COLS, ols_w)),   'intercept': ols_b0})

weights_df = pd.DataFrame(rows).set_index('method')
weights_df.to_csv(OUTPUT)

print(f"\nUloženo: {OUTPUT}")
