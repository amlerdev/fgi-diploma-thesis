"""
03_analysis.py
===================
Detailní analýza a vizualizace výsledků — unified grid search.

Strategie vybrány podle nejlepšího OOS Total Return:
  - Buy & Hold (benchmark, S&P 500 Total Return)
  - MR Long+MA  FG_Equal  entry=28, exit=92, fast=16, slow=100  (OOS 231%)
  - MR Long     FG_OLS    entry=48, exit=79                      (OOS 222%)
  - Mom Long    FG_Equal  entry=39, exit=32                      (OOS 167%, Sharpe 0.87)
  - Mom Long    FG_OLS    entry=85, exit=7                       (OOS 120%, DD -18.8%)

Primární hodnocení: OOS výsledky (2016–2026)

Author: Petr Amler (AML0005)
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Konfigurace ───────────────────────────────────────────────────────────────
STRATEGY_DIR = Path(__file__).resolve().parent
INPUT        = STRATEGY_DIR.parent / 'index' / 'fg_index_final.csv'

IS_START  = '1998-01-01'
IS_END    = '2015-12-31'
OOS_START = '2016-01-01'
INITIAL   = 10_000
FEE       = 0.001   # 0.1% per trade

CRISES = {
    'Dot-com\n(2000–2002)':      ('2000-03-01', '2002-10-31'),
    'Fin. krize\n(2007–2009)':   ('2007-10-01', '2009-03-31'),
    'COVID crash\n(Feb–Mar 20)': ('2020-02-01', '2020-03-31'),
    'COVID recovery\n(2020)':    ('2020-03-31', '2020-12-31'),
}

COLORS = {
    'Buy & Hold':        '#000000',
    'MR+MA FG_Equal':   '#4CAF50',
    'MR Long FG_OLS':   '#2196F3',
    'Mom Long FG_Equal': '#FF9800',
    'Mom Long FG_OLS':  '#E91E63',
}

# ── Načtení dat ───────────────────────────────────────────────────────────────
print("=" * 65)
print("UNIFIED ANALÝZA STRATEGIÍ — OOS 2016–2026 (primární, poplatky 0.1%)")
print("=" * 65)

df     = pd.read_csv(INPUT, parse_dates=['Date'], index_col='Date')
df     = df.dropna(subset=['SP500_Close'])
prices = df['SP500_Close'].values
dates  = df.index

fg_eq  = df['FG_Equal'].ffill().values
fg_ols = df['FG_OLS'].ffill().values

# MA pro MR Long+MA FG_Equal (fast=16, slow=100)
fg_eq_s = df['FG_Equal'].ffill()
ma16    = fg_eq_s.rolling(16,  min_periods=1).mean().values
ma100   = fg_eq_s.rolling(100, min_periods=1).mean().values

is_mask  = dates <= IS_END
oos_mask = dates >= OOS_START

print(f"\nIS:  {dates[is_mask][0].date()} → {dates[is_mask][-1].date()}  ({is_mask.sum():,} dní)")
print(f"OOS: {dates[oos_mask][0].date()} → {dates[oos_mask][-1].date()}  ({oos_mask.sum():,} dní)")

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

def mr_long_ma(prices, fg, entry, exit_, maf, mas):
    equity = np.empty(len(prices))
    cash = float(INITIAL); shares = 0.0; trades = 0
    for i in range(len(prices) - 1):
        if (fg[i] < entry and maf[i] > mas[i]
                and shares == 0 and cash > 0):
            shares = cash * (1 - FEE) / prices[i+1]; cash = 0.0; trades += 1
        elif fg[i] > exit_ and shares > 0:
            cash = shares * prices[i+1] * (1 - FEE); shares = 0.0; trades += 1
        equity[i] = cash + shares * prices[i]
    if shares > 0: cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity, trades

def metrics(equity, trades, base=None):
    base    = base if base is not None else equity[0]
    n       = len(equity)
    total_r = (equity[-1] - base) / base * 100
    cagr    = ((equity[-1] / base) ** (252 / n) - 1) * 100 if equity[-1] > 0 else -100.0
    daily_r = np.diff(equity) / equity[:-1]
    sharpe  = daily_r.mean() / daily_r.std() * np.sqrt(252) if daily_r.std() > 0 else 0.0
    roll_max = np.maximum.accumulate(equity)
    max_dd  = ((equity - roll_max) / roll_max * 100).min()
    return {'total_return': total_r, 'cagr': cagr, 'sharpe': sharpe,
            'max_dd': max_dd, 'trades': trades}

# ── Výpočet equity křivek ─────────────────────────────────────────────────────
strategies = {
    'Buy & Hold':        (INITIAL * prices / prices[0],                         0),
    'MR+MA FG_Equal':   mr_long_ma(prices, fg_eq,  28, 92, ma16, ma100),
    'MR Long FG_OLS':   mr_long(prices,    fg_ols, 48, 79),
    'Mom Long FG_Equal': mom_long(prices,  fg_eq,  39, 32),
    'Mom Long FG_OLS':  mom_long(prices,   fg_ols, 85,  7),
}

# ── Tisk souhrnné tabulky ─────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f"  {'Strategie':<22} {'IS Ret':>8} {'IS Shr':>7} {'IS DD':>7}  "
      f"{'OOS Ret':>8} {'OOS Shr':>7} {'OOS DD':>7}")
print(f"{'─'*70}")
for name, (eq, tr) in strategies.items():
    m_is  = metrics(eq[is_mask],  0, base=eq[is_mask][0])
    m_oos = metrics(eq[oos_mask], 0, base=eq[oos_mask][0])
    print(f"  {name:<22} "
          f"{m_is['total_return']:>7.1f}% {m_is['sharpe']:>7.2f} {m_is['max_dd']:>6.1f}%  "
          f"{m_oos['total_return']:>7.1f}% {m_oos['sharpe']:>7.2f} {m_oos['max_dd']:>6.1f}%")

# ── VIZUALIZACE ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 24))
fig.suptitle('Fear & Greed Index — Unified Grid Search\n'
             'Primární hodnocení: OOS období 2016–2026  |  S&P 500 Total Return  |  Poplatky: 0.1% per trade',
             fontsize=13, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

# Panel 1: OOS equity křivky (plná šířka) — PRIMÁRNÍ
ax1 = fig.add_subplot(gs[0, :])
for name, (eq, _) in strategies.items():
    eq_oos = eq[oos_mask]
    ax1.plot(dates[oos_mask], eq_oos / eq_oos[0] * 100,
             color=COLORS[name],
             linewidth=1.6 if name == 'Buy & Hold' else 1.1,
             alpha=0.9, label=name)
ax1.set_ylabel('Hodnota portfolia (% počátku OOS)')
ax1.set_title('OOS Equity křivky (2016–2026) — rebázováno na 100  ★ PRIMÁRNÍ VÝSLEDEK',
              fontweight='bold')
ax1.legend(fontsize=9, loc='upper left')
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3)

# Panel 2: Celé období s IS/OOS dělítkem
ax2 = fig.add_subplot(gs[1, :])
for name, (eq, _) in strategies.items():
    ax2.plot(dates, eq / INITIAL * 100, color=COLORS[name],
             linewidth=1.4 if name == 'Buy & Hold' else 0.9,
             alpha=0.85, label=name)
ax2.axvline(pd.Timestamp(OOS_START), color='red', linewidth=1.5,
            linestyle='--', alpha=0.8, label='IS/OOS dělítko (2016)')
ax2.axvspan(pd.Timestamp(OOS_START), dates[-1], alpha=0.04, color='red')
ax2.set_ylabel('Hodnota portfolia (%)')
ax2.set_title('Celé období 1998–2026 (levá část = IS, pravá = OOS)')
ax2.legend(fontsize=9, loc='upper left')
ax2.set_ylim(bottom=0)
ax2.grid(True, alpha=0.3)

# Panely 3–4: Drawdown IS a OOS
for col, (mask, title) in enumerate([(is_mask,  'Drawdown — IS (1998–2015)'),
                                      (oos_mask, 'Drawdown — OOS (2016–2026)')]):
    ax = fig.add_subplot(gs[2, col])
    for name, (eq, _) in strategies.items():
        eq_s     = eq[mask]
        roll_max = np.maximum.accumulate(eq_s)
        dd       = (eq_s - roll_max) / roll_max * 100
        ax.plot(dates[mask], dd, color=COLORS[name], linewidth=0.9,
                alpha=0.85, label=name)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

# Panel 5: Krizové periody (plná šířka)
ax5 = fig.add_subplot(gs[3, :])
strat_names = list(strategies.keys())
x     = np.arange(len(CRISES))
width = 0.16

for i, (name, (eq, _)) in enumerate(strategies.items()):
    crisis_returns = []
    for start, end in CRISES.values():
        mask_c = (dates >= start) & (dates <= end)
        if mask_c.sum() < 2:
            crisis_returns.append(0.0)
            continue
        eq_c = eq[mask_c]
        crisis_returns.append((eq_c[-1] - eq_c[0]) / eq_c[0] * 100)
    offset = (i - len(strat_names) / 2 + 0.5) * width
    bars = ax5.bar(x + offset, crisis_returns, width,
                   color=COLORS[name], alpha=0.85, label=name)
    for bar, val in zip(bars, crisis_returns):
        ypos = bar.get_height() + 0.5 if val >= 0 else bar.get_height() - 4.5
        ax5.text(bar.get_x() + bar.get_width() / 2, ypos,
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=6.5)

ax5.axhline(0, color='black', linewidth=1.0)
ax5.set_xticks(x)
ax5.set_xticklabels(list(CRISES.keys()), fontsize=9)
ax5.set_ylabel('Výnos v období (%)')
ax5.set_title('Výkonnost v krizových a recovery obdobích')
ax5.legend(fontsize=8, loc='upper right')
ax5.grid(axis='y', alpha=0.3)

out = STRATEGY_DIR / 'analysis_chart.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nGraf uložen: {out}")

# ── TABULKA VÝSLEDKŮ ──────────────────────────────────────────────────────────
df_oos = pd.read_csv(STRATEGY_DIR / 'oos_results.csv')

bh_is_m  = metrics(strategies['Buy & Hold'][0][is_mask],  0, base=strategies['Buy & Hold'][0][is_mask][0])
bh_oos_m = metrics(strategies['Buy & Hold'][0][oos_mask], 0, base=strategies['Buy & Hold'][0][oos_mask][0])

# Řádky tabulky
STRAT_ORDER = ['mr_long', 'mr_short', 'mom_long', 'mom_short', 'mr_long_ma', 'mom_long_ma']
STRAT_LABELS = {
    'mr_long':    'MR Long',    'mr_short':   'MR Short',
    'mom_long':   'Mom Long',   'mom_short':  'Mom Short',
    'mr_long_ma': 'MR Long+MA', 'mom_long_ma':'Mom Long+MA',
}
FG_LABELS = {'FG_Equal': 'Equal', 'FG_OLS': 'OLS'}

col_labels = ['Strategie', 'FG', 'Entry', 'Exit', 'Fast', 'Slow', 'Trades',
              'IS Ret', 'IS Sharpe', 'IS MaxDD',
              'OOS Ret', 'OOS Sharpe', 'OOS MaxDD']

table_rows = []
row_colors = []

# Buy & Hold řádek
bh_fast = int(bh_is_m['total_return'] * 0) if False else ''
table_rows.append([
    'Buy & Hold', 'TR Index', '—', '—', '—', '—', '—',
    f"{bh_is_m['total_return']:.1f}%",  f"{bh_is_m['sharpe']:.2f}",  f"{bh_is_m['max_dd']:.1f}%",
    f"{bh_oos_m['total_return']:.1f}%", f"{bh_oos_m['sharpe']:.2f}", f"{bh_oos_m['max_dd']:.1f}%",
])
row_colors.append(['#e8e8e8'] * 13)

prev_strat = None
for fg in ['FG_Equal', 'FG_OLS']:
    for strat in STRAT_ORDER:
        r = df_oos[(df_oos['strategy'] == strat) & (df_oos['fg_variant'] == fg)]
        if r.empty:
            continue
        r = r.iloc[0]
        fast = int(r['fast_ma']) if pd.notna(r['fast_ma']) else '—'
        slow = int(r['slow_ma']) if pd.notna(r['slow_ma']) else '—'

        oos_ret = r['oos_total_return']
        is_ret  = r['is_total_return']

        # Barva OOS return: zelená = kladné a nad B&H/2, červená = záporné
        if oos_ret < 0:
            c_oos = '#ffcccc'
        elif oos_ret < 50:
            c_oos = '#fff3cc'
        else:
            c_oos = '#ccffcc'

        c_is = '#e8f5e9' if is_ret > bh_is_m['total_return'] else '#ffffff'

        table_rows.append([
            STRAT_LABELS[strat], FG_LABELS[fg],
            int(r['entry']), int(r['exit']),
            fast if fast != '—' else '—',
            slow if slow != '—' else '—',
            int(r['is_trades']),
            f"{is_ret:.1f}%",         f"{r['is_sharpe']:.2f}",  f"{r['is_max_dd']:.1f}%",
            f"{oos_ret:.1f}%",        f"{r['oos_sharpe']:.2f}", f"{r['oos_max_dd']:.1f}%",
        ])
        row_colors.append(
            ['#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5', '#f5f5f5',
             c_is, '#f5f5f5', '#f5f5f5',
             c_oos, '#f5f5f5', '#f5f5f5']
        )

# Vykresli tabulku
fig2, ax_t = plt.subplots(figsize=(18, 7))
ax_t.axis('off')
fig2.suptitle('Přehled výsledků — IS vs OOS  |  S&P 500 Total Return  |  Poplatky: 0.1%/trade  |  IS: 1998–2015  |  OOS: 2016–2026',
              fontsize=12, fontweight='bold', y=0.98)

tbl = ax_t.table(
    cellText=table_rows,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    cellColours=row_colors,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)

# Záhlaví tučně
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#37474F')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

# IS/OOS skupiny — záhlaví sloupců barevně
for j in range(7, 10):
    tbl[0, j].set_facecolor('#1565C0')
for j in range(10, 13):
    tbl[0, j].set_facecolor('#2E7D32')

out2 = STRATEGY_DIR / 'results_table.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
print(f"Tabulka uložena: {out2}")
print("\nHotovo.")
