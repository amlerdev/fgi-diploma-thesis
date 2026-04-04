"""
01_grid_search.py
======================
Unified Grid Search — všechny strategie v jednom běhu (in-sample 1998–2015).

Strategie:
  mr_long     — Mean Reversion Long        (entry < exit)
  mr_short    — Mean Reversion Short       (entry < exit)
  mom_long    — Momentum Long              (entry > exit)
  mom_short   — Momentum Short             (entry > exit)
  mr_long_ma  — MR Long + MA filtr na FGI  (entry < exit, fast_ma < slow_ma)
  mom_long_ma — Mom Long + MA filtr na FGI (entry > exit, fast_ma < slow_ma)

Parametry:
  2D (mr/mom):     entry, exit ∈ [1, 99], krok 1
  4D (ma):
    mr_long_ma:    entry ∈ range(5, 50, 2), exit ∈ range(50, 96, 2)
    mom_long_ma:   entry ∈ range(50, 96, 2), exit ∈ range(5, 50, 2)
    fast_ma ∈ range(5, 35, 1), slow_ma ∈ range(20, 200, 5)

MA filtr (vstup):  MA_fast[i] > MA_slow[i]  (uptrend na FGI indexu)
MA filtr (výstup): bez filtru — FG > exit je dostatečná podmínka

FG varianty: FGI_Equal, FGI_OLS
Výstup: grid_results.csv

Author: Petr Amler (AML0005)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed

# ── Konfigurace ───────────────────────────────────────────────────────────────
STRATEGY_DIR = Path(__file__).resolve().parent
INPUT        = STRATEGY_DIR.parent / 'index' / 'fgi_index_final.csv'
OUTPUT       = STRATEGY_DIR / 'grid_results.csv'

IS_START = '1998-01-01'
IS_END   = '2015-12-31'

INITIAL  = 10_000
FEE      = 0.001   # 0.1% per trade (buy + sell = 0.2% round-trip)
FGI_COLS  = ['FGI_Equal', 'FGI_OLS']

# 2D rozsahy
GRID_VALS = list(range(1, 100))

# 4D MA rozsahy
ENTRY_MR    = list(range(2,  50, 2))   # 23 hodnot
EXIT_MR     = list(range(50, 96, 2))   # 23 hodnot
ENTRY_MOM   = list(range(50, 96, 2))   # 23 hodnot
EXIT_MOM    = list(range(2,  50, 2))   # 23 hodnot
FAST_RANGE  = list(range(5,  35, 1))   # 30 hodnot
SLOW_RANGE  = list(range(20, 200, 5))  # 36 hodnot

# ── Načtení dat ───────────────────────────────────────────────────────────────
print("=" * 70)
print("UNIFIED GRID SEARCH — VŠECHNY STRATEGIE (IN-SAMPLE 1998–2015)")
print("=" * 70)

df    = pd.read_csv(INPUT, parse_dates=['Date'], index_col='Date')
df_is = df.loc[IS_START:IS_END].dropna(subset=['SP500_Close'])

print(f"\nIn-sample: {df_is.index[0].date()} → {df_is.index[-1].date()}")
print(f"Počet dní: {len(df_is):,}")

prices = df_is['SP500_Close'].values

bh_return = (prices[-1] - prices[0]) / prices[0] * 100
n_days    = len(prices)
bh_cagr   = ((prices[-1] / prices[0]) ** (252 / n_days) - 1) * 100
print(f"\nBuy & Hold (IS): Return={bh_return:.1f}%  CAGR={bh_cagr:.1f}%")

# ── Metriky ───────────────────────────────────────────────────────────────────
def compute_metrics(equity: np.ndarray, trades: int) -> dict:
    n        = len(equity)
    total_r  = (equity[-1] - INITIAL) / INITIAL * 100
    cagr     = ((equity[-1] / INITIAL) ** (252 / n) - 1) * 100 if equity[-1] > 0 else -100.0
    daily_r  = np.diff(equity) / equity[:-1]
    sharpe   = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else 0.0
    roll_max = np.maximum.accumulate(equity)
    max_dd   = ((equity - roll_max) / roll_max * 100).min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0.0
    return {
        'total_return': round(total_r, 4),
        'cagr':         round(cagr,    4),
        'sharpe':       round(sharpe,  4),
        'max_dd':       round(max_dd,  4),
        'calmar':       round(calmar,  4),
        'trades':       trades,
    }

# ── Backtestery ───────────────────────────────────────────────────────────────
def mr_long(prices, fg, entry, exit_):
    equity = np.empty(len(prices))
    cash = float(INITIAL); shares = 0.0; trades = 0
    for i in range(len(prices) - 1):
        if fg[i] < entry and shares == 0 and cash > 0:
            shares = cash * (1 - FEE) / prices[i+1]; cash = 0.0; trades += 1
        elif fg[i] > exit_ and shares > 0:
            cash = shares * prices[i+1] * (1 - FEE); shares = 0.0; trades += 1
        equity[i] = cash + shares * prices[i]
    if shares > 0: cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity, trades

def mr_short(prices, fg, entry, exit_):
    equity = np.empty(len(prices))
    cash = float(INITIAL); short_sh = 0.0; entry_px = 0.0; trades = 0
    for i in range(len(prices) - 1):
        if fg[i] > exit_ and short_sh == 0 and cash > 0:
            short_sh = cash * (1 - FEE) / prices[i+1]; entry_px = prices[i+1]; trades += 1
        elif fg[i] < entry and short_sh > 0:
            cash += (entry_px - prices[i+1]) * short_sh - FEE * prices[i+1] * short_sh
            short_sh = 0.0; trades += 1
        equity[i] = cash + ((entry_px - prices[i]) * short_sh if short_sh > 0 else 0.0)
    if short_sh > 0:
        cash += (entry_px - prices[-1]) * short_sh - FEE * prices[-1] * short_sh
    equity[-1] = cash
    return equity, trades

def mom_long(prices, fg, entry, exit_):
    equity = np.empty(len(prices))
    cash = float(INITIAL); shares = 0.0; trades = 0
    for i in range(len(prices) - 1):
        if fg[i] > entry and shares == 0 and cash > 0:
            shares = cash * (1 - FEE) / prices[i+1]; cash = 0.0; trades += 1
        elif fg[i] < exit_ and shares > 0:
            cash = shares * prices[i+1] * (1 - FEE); shares = 0.0; trades += 1
        equity[i] = cash + shares * prices[i]
    if shares > 0: cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity, trades

def mom_short(prices, fg, entry, exit_):
    equity = np.empty(len(prices))
    cash = float(INITIAL); short_sh = 0.0; entry_px = 0.0; trades = 0
    for i in range(len(prices) - 1):
        if fg[i] < entry and short_sh == 0 and cash > 0:
            short_sh = cash * (1 - FEE) / prices[i+1]; entry_px = prices[i+1]; trades += 1
        elif fg[i] > exit_ and short_sh > 0:
            cash += (entry_px - prices[i+1]) * short_sh - FEE * prices[i+1] * short_sh
            short_sh = 0.0; trades += 1
        equity[i] = cash + ((entry_px - prices[i]) * short_sh if short_sh > 0 else 0.0)
    if short_sh > 0:
        cash += (entry_px - prices[-1]) * short_sh - FEE * prices[-1] * short_sh
    equity[-1] = cash
    return equity, trades

def mr_long_ma(prices, fg, entry, exit_, ma_fast, ma_slow):
    equity = np.empty(len(prices))
    cash = float(INITIAL); shares = 0.0; trades = 0
    for i in range(len(prices) - 1):
        if (fg[i] < entry and ma_fast[i] > ma_slow[i]
                and shares == 0 and cash > 0):
            shares = cash * (1 - FEE) / prices[i+1]; cash = 0.0; trades += 1
        elif fg[i] > exit_ and shares > 0:
            cash = shares * prices[i+1] * (1 - FEE); shares = 0.0; trades += 1
        equity[i] = cash + shares * prices[i]
    if shares > 0: cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity, trades

def mom_long_ma(prices, fg, entry, exit_, ma_fast, ma_slow):
    """Mom Long s MA filtrem: vstup jen v uptrendu FGI indexu."""
    equity = np.empty(len(prices))
    cash = float(INITIAL); shares = 0.0; trades = 0
    for i in range(len(prices) - 1):
        if (fg[i] > entry and ma_fast[i] > ma_slow[i]
                and shares == 0 and cash > 0):
            shares = cash * (1 - FEE) / prices[i+1]; cash = 0.0; trades += 1
        elif fg[i] < exit_ and shares > 0:
            cash = shares * prices[i+1] * (1 - FEE); shares = 0.0; trades += 1
        equity[i] = cash + shares * prices[i]
    if shares > 0: cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity, trades

STRATEGY_FNS_2D = {
    'mr_long':   mr_long,
    'mr_short':  mr_short,
    'mom_long':  mom_long,
    'mom_short': mom_short,
}

# ── Tasky pro paralelní pool ───────────────────────────────────────────────────
def run_2d(entry, exit_, strategy, fgi_variant, prices, fg):
    equity, trades = STRATEGY_FNS_2D[strategy](prices, fg, entry, exit_)
    m = compute_metrics(equity, trades)
    return {
        'strategy':   strategy,
        'fgi_variant': fgi_variant,
        'entry':      entry,
        'exit':       exit_,
        'fast_ma':    None,
        'slow_ma':    None,
        **m,
    }

def run_ma(entry, exit_, fast, slow, strategy, fgi_variant,
           prices, fg, ma_fast, ma_slow):
    if strategy == 'mr_long_ma':
        equity, trades = mr_long_ma(prices, fg, entry, exit_, ma_fast, ma_slow)
    else:
        equity, trades = mom_long_ma(prices, fg, entry, exit_, ma_fast, ma_slow)
    m = compute_metrics(equity, trades)
    return {
        'strategy':   strategy,
        'fgi_variant': fgi_variant,
        'entry':      entry,
        'exit':       exit_,
        'fast_ma':    fast,
        'slow_ma':    slow,
        **m,
    }

# ── Sestavení tasků ───────────────────────────────────────────────────────────
all_tasks = []   # (type, args...)

for fgi_var in FGI_COLS:
    if fgi_var not in df_is.columns:
        print(f"   VAROVÁNÍ: {fgi_var} chybí, přeskakuji.")
        continue

    fg_arr    = df_is[fgi_var].ffill().values
    fg_series = df_is[fgi_var].ffill()

    # -- 2D tasky --
    for strategy in STRATEGY_FNS_2D:
        for entry in GRID_VALS:
            for exit_ in GRID_VALS:
                if strategy in ('mr_long', 'mr_short') and entry >= exit_:
                    continue
                if strategy in ('mom_long', 'mom_short') and entry <= exit_:
                    continue
                all_tasks.append(('2d', entry, exit_, strategy, fgi_var, prices, fg_arr))

    # -- 4D MA tasky — předpočítej MA cache --
    ma_cache = {}
    for w in set(FAST_RANGE) | set(SLOW_RANGE):
        ma_cache[w] = fg_series.rolling(w, min_periods=w).mean().values

    valid_ma_pairs = [(f, s) for f in FAST_RANGE for s in SLOW_RANGE if f < s]

    # MR Long + MA
    for entry in ENTRY_MR:
        for exit_ in EXIT_MR:
            # entry < exit garantováno rozsahy (max entry=49 < min exit=50)
            for fast, slow in valid_ma_pairs:
                all_tasks.append(('ma', entry, exit_, fast, slow,
                                  'mr_long_ma', fgi_var, prices, fg_arr,
                                  ma_cache[fast], ma_cache[slow]))

    # Mom Long + MA
    for entry in ENTRY_MOM:
        for exit_ in EXIT_MOM:
            # entry > exit garantováno rozsahy (min entry=50 > max exit=49)
            for fast, slow in valid_ma_pairs:
                all_tasks.append(('ma', entry, exit_, fast, slow,
                                  'mom_long_ma', fgi_var, prices, fg_arr,
                                  ma_cache[fast], ma_cache[slow]))

print(f"\nCelkem tasků: {len(all_tasks):,}")
print(f"  z toho 2D:  {sum(1 for t in all_tasks if t[0] == '2d'):,}")
print(f"  z toho MA:  {sum(1 for t in all_tasks if t[0] == 'ma'):,}")

# ── Paralelní běh ─────────────────────────────────────────────────────────────
def dispatch(task):
    if task[0] == '2d':
        _, entry, exit_, strategy, fgi_var, pr, fg = task
        return run_2d(entry, exit_, strategy, fgi_var, pr, fg)
    else:
        _, entry, exit_, fast, slow, strategy, fgi_var, pr, fg, maf, mas = task
        return run_ma(entry, exit_, fast, slow, strategy, fgi_var, pr, fg, maf, mas)

print(f"\nSpouštím paralelní výpočet (n_jobs=-1)...")
t0      = datetime.now()
results = Parallel(n_jobs=-1, verbose=0)(
    delayed(dispatch)(task) for task in all_tasks
)
elapsed = (datetime.now() - t0).total_seconds()
print(f"Hotovo za {elapsed:.1f}s  ({len(results):,} výsledků)")

# ── Uložení ───────────────────────────────────────────────────────────────────
df_res = pd.DataFrame(results)
before = len(df_res)
df_res = df_res[df_res['trades'] >= 10]
print(f"\nFiltr min. obchodů (≥10): {before:,} → {len(df_res):,} kombinací"
      f" (vyřazeno {before - len(df_res):,})")
df_res.to_csv(OUTPUT, index=False)
print(f"Uloženo: {OUTPUT}")
print(f"Řádků celkem: {len(df_res):,}")

# ── TOP výsledky ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TOP 1 per strategie × FGI varianta (podle Total Return)")
print("=" * 70)
print(f"\nBuy & Hold (IS): Return={bh_return:.1f}%  CAGR={bh_cagr:.1f}%")

for fgi_var in FGI_COLS:
    print(f"\n{'─'*70}")
    print(f"  {fgi_var}")
    print(f"{'─'*70}")
    print(f"  {'Strategie':<15} {'entry':>5} {'exit':>5} {'fast':>5} {'slow':>5}"
          f"  {'Return':>9} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Trades':>7}")
    sub = df_res[df_res['fgi_variant'] == fgi_var]
    for strat in ['mr_long', 'mr_short', 'mom_long', 'mom_short',
                  'mr_long_ma', 'mom_long_ma']:
        s = sub[sub['strategy'] == strat]
        if s.empty:
            continue
        top = s.nlargest(1, 'total_return').iloc[0]
        fast = int(top['fast_ma']) if pd.notna(top['fast_ma']) else 0
        slow = int(top['slow_ma']) if pd.notna(top['slow_ma']) else 0
        print(f"  {strat:<15} {int(top['entry']):>5} {int(top['exit']):>5}"
              f" {fast:>5} {slow:>5}"
              f"  {top['total_return']:>8.1f}%"
              f" {top['cagr']:>6.1f}%"
              f"  {top['sharpe']:>7.2f}"
              f" {top['max_dd']:>7.1f}%"
              f"  {int(top['trades']):>7}")

print("\nHotovo — spust 02_out_of_sample.py pro out-of-sample validaci")
