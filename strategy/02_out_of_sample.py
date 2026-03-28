"""
02_out_of_sample.py
==============
Out-of-sample validace nejlepších parametrů z grid_results.csv.

Selekce: top 1 per strategie × FG varianta (podle Total Return IS)
Strategie: mr_long, mr_short, mom_long, mom_short, mr_long_ma, mom_long_ma

IS:  1998-01-01 → 2015-12-31
OOS: 2016-01-01 → 2026-12-31

Výstup: oos_results.csv

Author: Petr Amler (AML0005)
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Konfigurace ───────────────────────────────────────────────────────────────
STRATEGY_DIR = Path(__file__).resolve().parent
INPUT        = STRATEGY_DIR.parent / 'index' / 'fg_index_final.csv'
GRID         = STRATEGY_DIR / 'grid_results.csv'
OUTPUT       = STRATEGY_DIR / 'oos_results.csv'

IS_START  = '1998-01-01'
IS_END    = '2015-12-31'
OOS_START = '2016-01-01'
OOS_END   = '2026-12-31'

INITIAL = 10_000
FEE     = 0.001   # 0.1% per trade (buy + sell = 0.2% round-trip)

MA_STRATEGIES = {'mr_long_ma', 'mom_long_ma'}

# ── Načtení dat ───────────────────────────────────────────────────────────────
print("=" * 70)
print("UNIFIED OUT-OF-SAMPLE VALIDACE (2016–2026)")
print("=" * 70)

df     = pd.read_csv(INPUT, parse_dates=['Date'], index_col='Date')
df_is  = df.loc[IS_START:IS_END].dropna(subset=['SP500_Close'])
df_oos = df.loc[OOS_START:OOS_END].dropna(subset=['SP500_Close'])

print(f"\nIS:  {df_is.index[0].date()} → {df_is.index[-1].date()}  ({len(df_is):,} dní)")
print(f"OOS: {df_oos.index[0].date()} → {df_oos.index[-1].date()}  ({len(df_oos):,} dní)")

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
        'total_return': round(total_r, 2),
        'cagr':         round(cagr,    2),
        'sharpe':       round(sharpe,  3),
        'max_dd':       round(max_dd,  2),
        'calmar':       round(calmar,  3),
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

STRATEGY_FNS = {
    'mr_long':  mr_long,  'mr_short':  mr_short,
    'mom_long': mom_long, 'mom_short': mom_short,
}

# ── Pomocná funkce: spusť backtest na daném období ────────────────────────────
def run_period(df_period, row):
    pr       = df_period['SP500_Close'].values
    fgv      = row['fg_variant']
    fg       = df_period[fgv].ffill().values
    e, x     = int(row['entry']), int(row['exit'])
    strategy = row['strategy']

    if strategy in MA_STRATEGIES:
        fg_series = df_period[fgv].ffill()
        maf = fg_series.rolling(int(row['fast_ma']), min_periods=1).mean().values
        mas = fg_series.rolling(int(row['slow_ma']), min_periods=1).mean().values
        if strategy == 'mr_long_ma':
            eq, tr = mr_long_ma(pr, fg, e, x, maf, mas)
        else:
            eq, tr = mom_long_ma(pr, fg, e, x, maf, mas)
    else:
        eq, tr = STRATEGY_FNS[strategy](pr, fg, e, x)

    return compute_metrics(eq, tr)

# ── Selekce top 1 per strategie × FG varianta ────────────────────────────────
print("\nNačítám grid_results.csv...")
df_grid = pd.read_csv(GRID)
print(f"   {len(df_grid):,} řádků celkem")

all_strategies = ['mr_long', 'mr_short', 'mom_long', 'mom_short',
                  'mr_long_ma', 'mom_long_ma']

top = (df_grid[df_grid['strategy'].isin(all_strategies)]
       .sort_values('total_return', ascending=False)
       .groupby(['fg_variant', 'strategy'])
       .first()
       .reset_index())

print(f"   Vybrané konfigurace: {len(top)} (top 1 per strategie × FG varianta)")

# ── Výpočet IS + OOS ──────────────────────────────────────────────────────────
print("\nPočítám IS a OOS výsledky...")
rows = []
for _, row in top.iterrows():
    is_m  = run_period(df_is,  row)
    oos_m = run_period(df_oos, row)
    fast  = int(row['fast_ma']) if pd.notna(row['fast_ma']) else None
    slow  = int(row['slow_ma']) if pd.notna(row['slow_ma']) else None
    rows.append({
        'fg_variant': row['fg_variant'],
        'strategy':   row['strategy'],
        'entry':      int(row['entry']),
        'exit':       int(row['exit']),
        'fast_ma':    fast,
        'slow_ma':    slow,
        **{f'is_{k}':  v for k, v in is_m.items()},
        **{f'oos_{k}': v for k, v in oos_m.items()},
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(OUTPUT, index=False)
print(f"Uloženo: {OUTPUT}")

# ── Buy & Hold benchmark ──────────────────────────────────────────────────────
def bh_metrics(df_period):
    pr = df_period['SP500_Close'].values
    eq = INITIAL * pr / pr[0]
    return compute_metrics(eq, 0)

bh_is  = bh_metrics(df_is)
bh_oos = bh_metrics(df_oos)

# ── Výpis výsledků ────────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("VÝSLEDKY: IN-SAMPLE vs OUT-OF-SAMPLE")
print("=" * 75)

header = (f"\n  {'Strategie':<18} {'entry':>5} {'exit':>5} {'fast':>5} {'slow':>5}  "
          f"{'IS Ret':>8} {'IS Shr':>6} {'IS DD':>7}  "
          f"{'OOS Ret':>8} {'OOS Shr':>6} {'OOS DD':>7}")
sep = "  " + "─" * 73

print(f"\n{'─'*75}")
print(f"  BUY & HOLD (S&P 500 Total Return)")
print(f"{'─'*75}")
print(header)
print(sep)
print(f"  {'buy_and_hold':<18} {'—':>5} {'—':>5} {'—':>5} {'—':>5}  "
      f"{bh_is['total_return']:>7.1f}% {bh_is['sharpe']:>6.2f} {bh_is['max_dd']:>6.1f}%  "
      f"{bh_oos['total_return']:>7.1f}% {bh_oos['sharpe']:>6.2f} {bh_oos['max_dd']:>6.1f}%")

for fg_var in ['FG_Equal', 'FG_OLS']:
    sub = df_out[df_out['fg_variant'] == fg_var]
    print(f"\n{'─'*75}")
    print(f"  {fg_var}")
    print(f"{'─'*75}")
    print(header)
    print(sep)
    for strat in all_strategies:
        r = sub[sub['strategy'] == strat]
        if r.empty:
            continue
        r = r.iloc[0]
        fast = int(r['fast_ma']) if pd.notna(r['fast_ma']) else 0
        slow = int(r['slow_ma']) if pd.notna(r['slow_ma']) else 0
        print(f"  {strat:<18} {int(r['entry']):>5} {int(r['exit']):>5}"
              f" {fast:>5} {slow:>5}  "
              f"{r['is_total_return']:>7.1f}% {r['is_sharpe']:>6.2f} {r['is_max_dd']:>6.1f}%  "
              f"{r['oos_total_return']:>7.1f}% {r['oos_sharpe']:>6.2f} {r['oos_max_dd']:>6.1f}%")

print("\nHotovo — spust 03_analysis.py pro detailní grafy")
