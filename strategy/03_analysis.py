"""
Analýza a vizualizace výsledků OOS validace.

Načte oos_results.csv a vygeneruje:
  1. Tabulku IS vs OOS (PNG + terminálový výpis)
  2. Graf OOS equity křivek (top konfigurace, rebázované na 100)
  3. Graf celého období 1998–2026 s IS/OOS dělítkem

B&H benchmark je konzistentně počítán přes compute_metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FGI_COLS, INITIAL, INPUT,
    IS_END, IS_START, OOS_END, OOS_START, STRATEGY_DIR,
)
from backtester import STRATEGIES, FEE, compute_metrics


# ---------------------------------------------------------------------------
# Paleta barev a styly pro přehledné grafy
# ---------------------------------------------------------------------------

_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
_LS = ['-', '--', '-.', ':']   # různé styly čar pro odlišení


# ---------------------------------------------------------------------------
# Pomocné funkce pro rekonstrukci equity křivek
# ---------------------------------------------------------------------------

def _build_equity(row: pd.Series, prices: np.ndarray, fg: np.ndarray) -> np.ndarray:
    """Rekonstruuje equity křivku pro jednu konfiguraci na daném price/fg segmentu."""
    strategy = row['strategy']
    fn       = STRATEGIES[strategy]

    if strategy in ('ma_combined', 'ma_long'):
        eq, _ = fn(prices, fg, fast=int(row['fast']), slow=int(row['slow']))
    else:
        eq, _ = fn(
            prices, fg,
            entry=int(row['entry']),
            exit=int(row['exit']),
        )
    return eq


def _build_equity_ma_oos(
    row:     pd.Series,
    df_full: pd.DataFrame,
    df_oos:  pd.DataFrame,
) -> np.ndarray:
    """
    Rekonstruuje equity křivku pro ma_combined na OOS datech.
    MA jsou spočítány z celého IS+OOS datasetu — správný warmup na začátku OOS.
    """
    fgi_col = row['fgi_col']
    fast    = int(row['fast'])
    slow    = int(row['slow'])

    fg_full   = df_full[fgi_col].to_numpy(dtype=float)
    oos_idx   = df_full.index.get_loc(df_oos.index[0])
    ma_fast   = pd.Series(fg_full).rolling(fast, min_periods=fast).mean().to_numpy()[oos_idx:]
    ma_slow   = pd.Series(fg_full).rolling(slow, min_periods=slow).mean().to_numpy()[oos_idx:]
    prices    = df_oos['SP500_Close'].to_numpy(dtype=float)

    cash        = float(INITIAL)
    shares      = 0.0    # počet akcií v long pozici
    invested    = 0.0    # investovaná částka v short pozici
    entry_price = 0.0    # vstupní cena short pozice
    equity      = np.empty(len(prices))

    for i in range(len(prices) - 1):
        # Aktuální hodnota portfolia před případnou exekucí signálu na i+1
        if shares > 0.0:
            equity[i] = shares * prices[i]
        elif invested > 0.0:
            equity[i] = invested * (entry_price / prices[i])   # inverzní ETF model
        else:
            equity[i] = cash

        is_last_execution = i == len(prices) - 2

        # Warmup — slow MA ještě nemá dostatek dat
        if np.isnan(ma_slow[i]):
            continue

        if ma_fast[i] > ma_slow[i]:
            # Fast MA nad slow MA — sentiment roste, chceme být LONG

            if invested > 0.0:
                # Přechod SHORT → LONG: uzavři short...
                cash        = invested * (entry_price / prices[i + 1]) * (1 - FEE)
                invested    = 0.0
                entry_price = 0.0
                # ...a nakup long
                if not is_last_execution:
                    shares = cash * (1 - FEE) / prices[i + 1]
                    cash   = 0.0

            elif shares == 0.0 and not is_last_execution:
                # První vstup do LONG
                shares = cash * (1 - FEE) / prices[i + 1]
                cash   = 0.0

        elif ma_fast[i] < ma_slow[i]:
            # Fast MA pod slow MA — sentiment klesá, chceme být SHORT

            if shares > 0.0:
                # Přechod LONG → SHORT: prodej long...
                cash   = shares * prices[i + 1] * (1 - FEE)
                shares = 0.0
                # ...a otevři short
                if not is_last_execution:
                    invested    = cash * (1 - FEE)
                    entry_price = prices[i + 1]
                    cash        = 0.0

            elif invested == 0.0 and not is_last_execution:
                # První vstup do SHORT
                invested    = cash * (1 - FEE)
                entry_price = prices[i + 1]
                cash        = 0.0

        if equity[i] <= 0.0:
            equity[i:] = 0.0
            return equity

    # Uzavři pozici na konci období
    if shares > 0.0:
        cash = shares * prices[-1] * (1 - FEE)
    elif invested > 0.0:
        cash = invested * (entry_price / prices[-1]) * (1 - FEE)
    equity[-1] = cash
    return equity


def _build_equity_ma_long_oos(
    row:     pd.Series,
    df_full: pd.DataFrame,
    df_oos:  pd.DataFrame,
) -> np.ndarray:
    """
    Rekonstruuje equity křivku pro ma_long na OOS datech.
    MA jsou spočítány z celého IS+OOS datasetu — správný warmup na začátku OOS.
    Bez short větve — pouze long nebo hotovost.
    """
    fgi_col = row['fgi_col']
    fast    = int(row['fast'])
    slow    = int(row['slow'])

    fg_full = df_full[fgi_col].to_numpy(dtype=float)
    oos_idx = df_full.index.get_loc(df_oos.index[0])
    ma_fast = pd.Series(fg_full).rolling(fast, min_periods=fast).mean().to_numpy()[oos_idx:]
    ma_slow = pd.Series(fg_full).rolling(slow, min_periods=slow).mean().to_numpy()[oos_idx:]
    prices  = df_oos['SP500_Close'].to_numpy(dtype=float)

    cash   = float(INITIAL)
    shares = 0.0
    equity = np.empty(len(prices))

    for i in range(len(prices) - 1):
        if shares > 0.0:
            equity[i] = shares * prices[i]
        else:
            equity[i] = cash

        is_last_execution = i == len(prices) - 2

        if np.isnan(ma_slow[i]):
            continue

        if ma_fast[i] > ma_slow[i]:
            if shares == 0.0 and not is_last_execution:
                shares = cash * (1 - FEE) / prices[i + 1]
                cash   = 0.0

        elif ma_fast[i] < ma_slow[i]:
            if shares > 0.0:
                cash   = shares * prices[i + 1] * (1 - FEE)
                shares = 0.0

        if equity[i] <= 0.0:
            equity[i:] = 0.0
            return equity

    if shares > 0.0:
        cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity


# ---------------------------------------------------------------------------
# Tabulka IS vs OOS (PNG)
# ---------------------------------------------------------------------------

def plot_results_table(df_oos: pd.DataFrame, bh_is: dict, bh_oos: dict) -> None:
    """Uloží přehlednou tabulku IS vs OOS jako PNG."""
    rows = []
    for _, r in df_oos.sort_values(['strategy', 'fgi_col']).iterrows():
        if r['strategy'] in ('ma_combined', 'ma_long'):
            pstr = f"fast={int(r['fast'])} slow={int(r['slow'])}"
        else:
            pstr = f"entry={int(r['entry'])} exit={int(r['exit'])}"
        rows.append([
            r['strategy'], r['fgi_col'], pstr,
            f"{r['is_total_return']:+.1f}%",
            f"{r['is_sharpe']:+.2f}",
            f"{r['is_max_dd']:+.1f}%",
            f"{r['oos_total_return']:+.1f}%",
            f"{r['oos_sharpe']:+.2f}",
            f"{r['oos_max_dd']:+.1f}%",
        ])
    # B&H řádky
    rows.append([
        'Buy & Hold', f'IS {IS_START[:4]}–{IS_END[:4]}', '—',
        f"{bh_is['total_return']:+.1f}%", f"{bh_is['sharpe']:+.2f}", f"{bh_is['max_dd']:+.1f}%",
        '—', '—', '—',
    ])
    rows.append([
        'Buy & Hold', f'OOS {OOS_START[:4]}–{OOS_END[:4]}', '—',
        '—', '—', '—',
        f"{bh_oos['total_return']:+.1f}%", f"{bh_oos['sharpe']:+.2f}", f"{bh_oos['max_dd']:+.1f}%",
    ])

    cols = [
        'Strategie', 'FGI', 'Parametry',
        'IS Return', 'IS Sharpe', 'IS MaxDD',
        'OOS Return', 'OOS Sharpe', 'OOS MaxDD',
    ]

    fig, ax = plt.subplots(figsize=(18, 0.45 * (len(rows) + 2)))
    ax.axis('off')

    tbl = ax.table(
        cellText=rows, colLabels=cols,
        loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width(list(range(len(cols))))

    # Záhlaví tučně + šedé pozadí
    for j in range(len(cols)):
        tbl[0, j].set_facecolor('#404040')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    # Střídavé řádky
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')

    # B&H řádky — zvýraznění
    for j in range(len(cols)):
        tbl[len(rows) - 1, j].set_facecolor('#fff3cd')
        tbl[len(rows),     j].set_facecolor('#fff3cd')

    plt.title('IS vs OOS — přehled výkonnosti strategií', fontsize=12,
              fontweight='bold', pad=10)
    out = STRATEGY_DIR / 'results_table.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Tabulka uložena: {out}')


# ---------------------------------------------------------------------------
# Graf 1 — OOS equity křivky (rebázované na 100)
# ---------------------------------------------------------------------------

def plot_oos_equity(
    df_oos_res: pd.DataFrame,
    df_full:    pd.DataFrame,
    df_oos:     pd.DataFrame,
) -> None:
    """
    Vykreslí OOS equity křivky všech konfigurací (rebázované na 100)
    včetně B&H benchmarku.
    """
    dates_oos  = df_oos.index
    prices_oos = df_oos['SP500_Close'].to_numpy(dtype=float)
    bh_eq      = INITIAL * prices_oos / prices_oos[0]

    fig, ax = plt.subplots(figsize=(14, 7))

    color_idx = 0
    for i, (_, row) in enumerate(df_oos_res.sort_values(['strategy', 'fgi_col']).iterrows()):
        fgi_col = row['fgi_col']
        fg_oos  = df_oos[fgi_col].to_numpy(dtype=float)

        if row['strategy'] == 'ma_combined':
            eq = _build_equity_ma_oos(row, df_full, df_oos)
        elif row['strategy'] == 'ma_long':
            eq = _build_equity_ma_long_oos(row, df_full, df_oos)
        else:
            eq = _build_equity(row, prices_oos, fg_oos)

        rebased = eq / eq[0] * 100.0

        if row['strategy'] in ('ma_combined', 'ma_long'):
            label = f"{row['strategy']} {fgi_col} (f={int(row['fast'])},s={int(row['slow'])})"
        else:
            label = f"{row['strategy']} {fgi_col} (e={int(row['entry'])},x={int(row['exit'])})"

        ls = _LS[i % len(_LS)]
        ax.plot(dates_oos, rebased, label=label,
                color=_COLORS[color_idx % len(_COLORS)],
                linestyle=ls, linewidth=1.4, alpha=0.85)
        color_idx += 1

    # B&H
    ax.plot(dates_oos, bh_eq / bh_eq[0] * 100.0,
            label='Buy & Hold', color='black', linewidth=2.0, linestyle='-')

    ax.axhline(100, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title('OOS equity křivky (2016–2026) — rebázováno na 100', fontsize=13)
    ax.set_xlabel('Datum')
    ax.set_ylabel('Hodnota portfolia (start = 100)')
    ax.legend(loc='upper left', fontsize=7.5, framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = STRATEGY_DIR / 'oos_equity.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'OOS equity graf uložen: {out}')


# ---------------------------------------------------------------------------
# Graf 2 — celé období 1998–2026 s IS/OOS dělítkem
# ---------------------------------------------------------------------------

def plot_full_period(
    df_oos_res: pd.DataFrame,
    df_full:    pd.DataFrame,
    df_is:      pd.DataFrame,
    df_oos:     pd.DataFrame,
) -> None:
    """
    Vykreslí equity křivky pro celé období 1998–2026 (IS + OOS).
    IS část je šedá (výsledek optimalizace), OOS část barevná (validace).
    Svislá čára odděluje IS a OOS periodu.
    """
    dates_is   = df_is.index
    dates_oos  = df_oos.index
    prices_is  = df_is['SP500_Close'].to_numpy(dtype=float)
    prices_oos = df_oos['SP500_Close'].to_numpy(dtype=float)

    bh_is_eq  = INITIAL * prices_is  / prices_is[0]
    bh_oos_eq = INITIAL * prices_oos / prices_oos[0]

    fig, ax = plt.subplots(figsize=(16, 7))

    # IS/OOS dělítko
    split_date = pd.Timestamp(OOS_START)
    ax.axvline(split_date, color='black', linewidth=1.5, linestyle='--', alpha=0.7)
    ax.text(split_date, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1,
            ' ← IS  |  OOS →', fontsize=9, va='bottom', ha='left', alpha=0.7)

    # Šedý B&H pro IS, barevný pro OOS
    ax.plot(dates_is,  bh_is_eq,  color='gray',  linewidth=1.5, linestyle='-',  alpha=0.5)
    ax.plot(dates_oos, bh_oos_eq, color='black', linewidth=2.0, linestyle='-',  label='Buy & Hold OOS')

    color_idx = 0
    for i, (_, row) in enumerate(df_oos_res.sort_values(['strategy', 'fgi_col']).iterrows()):
        fgi_col = row['fgi_col']
        fg_is   = df_is[fgi_col].to_numpy(dtype=float)
        fg_oos  = df_oos[fgi_col].to_numpy(dtype=float)

        # IS equity (šedě, průhledně)
        if row['strategy'] in ('ma_long', 'ma_combined'):
            eq_is, _ = STRATEGIES[row['strategy']](
                prices_is, fg_is,
                fast=int(row['fast']), slow=int(row['slow'])
            )
        else:
            eq_is, _ = STRATEGIES[row['strategy']](
                prices_is, fg_is,
                entry=int(row['entry']),
                exit=int(row['exit']),
            )

        # OOS equity (barevně)
        if row['strategy'] == 'ma_combined':
            eq_oos = _build_equity_ma_oos(row, df_full, df_oos)
        elif row['strategy'] == 'ma_long':
            eq_oos = _build_equity_ma_long_oos(row, df_full, df_oos)
        else:
            eq_oos = _build_equity(row, prices_oos, fg_oos)

        color = _COLORS[color_idx % len(_COLORS)]
        ls    = _LS[i % len(_LS)]

        label = f"{row['strategy']} {fgi_col}"

        ax.plot(dates_is,  eq_is,  color=color, linewidth=1.0, linestyle=ls, alpha=0.3)
        ax.plot(dates_oos, eq_oos, color=color, linewidth=1.5, linestyle=ls,
                alpha=0.9, label=label)
        color_idx += 1

    # Přidej dělítko label znovu (po vykreslení os)
    ymax = ax.get_ylim()[1]
    ax.text(split_date, ymax * 0.97,
            ' ← IS  |  OOS →', fontsize=9, va='top', ha='left', alpha=0.7)

    ax.set_title('Celé období 1998–2026 (IS šedě, OOS barevně) — kapitál v USD', fontsize=13)
    ax.set_xlabel('Datum')
    ax.set_ylabel('Hodnota portfolia (USD, start = 10 000)')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = STRATEGY_DIR / 'full_period.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Graf celého období uložen: {out}')


# ---------------------------------------------------------------------------
# Hlavní funkce
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Načtení dat -------------------------------------------------------
    df_full = pd.read_csv(INPUT, index_col='Date', parse_dates=True)
    df_is   = df_full.loc[IS_START:IS_END].copy()
    df_oos  = df_full.loc[OOS_START:OOS_END].copy()

    prices_is  = df_is['SP500_Close'].to_numpy(dtype=float)
    prices_oos = df_oos['SP500_Close'].to_numpy(dtype=float)

    bh_is  = compute_metrics(INITIAL * prices_is  / prices_is[0],  0)
    bh_oos = compute_metrics(INITIAL * prices_oos / prices_oos[0], 0)

    # ---- Načtení OOS výsledků ---------------------------------------------
    oos_path = STRATEGY_DIR / 'oos_results.csv'
    if not oos_path.exists():
        raise FileNotFoundError(
            f'{oos_path} nenalezen. Nejprve spusť 02_out_of_sample.py.'
        )
    df_oos_res = pd.read_csv(oos_path)
    print(f'Načteno {len(df_oos_res)} konfigurací z {oos_path.name}\n')

    # ---- Terminálový výpis (kopie z 02_out_of_sample pro přehled) ---------
    SEP  = '=' * 100
    SEP2 = '-' * 100
    print(SEP)
    print('IS vs OOS — přehled')
    print(SEP)
    hdr = (
        f'{"Strategie":<23}  {"FGI":<10}  {"Parametry":<20}'
        f'  {"IS Ret":>7}  {"IS Sh":>6}  {"IS DD":>7}'
        f'  {"OOS Ret":>8}  {"OOS Sh":>7}  {"OOS DD":>8}'
    )
    print(hdr)
    print(SEP2)
    for _, r in df_oos_res.sort_values(['strategy', 'fgi_col']).iterrows():
        pstr = (f"f={int(r['fast'])} s={int(r['slow'])}"
                if r['strategy'] in ('ma_combined', 'ma_long')
                else f"entry={int(r['entry'])} exit={int(r['exit'])}")
        print(
            f'{r["strategy"]:<23}  {r["fgi_col"]:<10}  {pstr:<20}'
            f'  {r["is_total_return"]:>+6.1f}%  {r["is_sharpe"]:>+5.2f}  {r["is_max_dd"]:>+6.1f}%'
            f'  {r["oos_total_return"]:>+7.1f}%  {r["oos_sharpe"]:>+6.2f}  {r["oos_max_dd"]:>+7.1f}%'
        )
    print(SEP2)
    print(
        f'{"Buy & Hold IS":<23}  {"":<10}  {"":<20}'
        f'  {bh_is["total_return"]:>+6.1f}%  {bh_is["sharpe"]:>+5.2f}  {bh_is["max_dd"]:>+6.1f}%'
        f'  {"":>8}  {"":>7}  {"":>8}'
    )
    print(
        f'{"Buy & Hold OOS":<23}  {"":<10}  {"":<20}'
        f'  {"":>7}  {"":>6}  {"":>7}'
        f'  {bh_oos["total_return"]:>+7.1f}%  {bh_oos["sharpe"]:>+6.2f}  {bh_oos["max_dd"]:>+7.1f}%'
    )
    print(SEP)
    print()

    # ---- Grafy -------------------------------------------------------------
    plot_results_table(df_oos_res, bh_is, bh_oos)
    plot_oos_equity(df_oos_res, df_full, df_oos)
    plot_full_period(df_oos_res, df_full, df_is, df_oos)

    print('\nAnalýza dokončena.')


if __name__ == '__main__':
    main()
