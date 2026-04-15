"""
Analýza a vizualizace výsledků OOS validace.

Načte oos_results.csv a vygeneruje:
  1. Tabulku IS vs OOS (PNG + terminálový výpis)
  2. Dva samostatné OOS grafy pro thesis-ready výstup:
     - oos_equity_equal.png
     - oos_equity_ols.png
  3. Graf celého období 1998–2026 s IS/OOS dělítkem

B&H benchmark je konzistentně počítán přes compute_metrics.
Všechny strategie jsou rekonstruovány přes stejný position-based execution model.
Short expozice je zjednodušená syntetická -1x denní návratnost bez borrow
a financing cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FGI_COLS, INITIAL, INPUT,
    IS_END, IS_START, OOS_END, OOS_START, STRATEGY_DIR,
)
from backtester import (
    STRATEGIES,
    _ma_combined_from_arrays,
    _ma_long_from_arrays,
    compute_metrics,
)


# ---------------------------------------------------------------------------
# Paleta barev a styly pro přehledné grafy
# ---------------------------------------------------------------------------

_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]
_LS = ['-', '--', '-.', ':']   # různé styly čar pro odlišení

_STRATEGY_ORDER = [
    'kontrarian_long',
    'kontrarian_combined',
    'trend_long',
    'trend_combined',
    'ma_long',
    'ma_combined',
]

_STRATEGY_LABELS = {
    'kontrarian_long': 'Kontrarian long',
    'kontrarian_combined': 'Kontrarian combined',
    'trend_long': 'Trend long',
    'trend_combined': 'Trend combined',
    'ma_long': 'MA long',
    'ma_combined': 'MA combined',
}

_STRATEGY_COLORS = {
    'kontrarian_long': '#1f77b4',
    'kontrarian_combined': '#ff7f0e',
    'trend_long': '#2ca02c',
    'trend_combined': '#d62728',
    'ma_long': '#9467bd',
    'ma_combined': '#8c564b',
}


# ---------------------------------------------------------------------------
# Pomocné funkce pro rekonstrukci equity křivek
# ---------------------------------------------------------------------------

def _normalize_trade_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zajistí kompatibilitu starších výstupů s názvem trades místo transactions.
    """
    rename_map = {}
    if 'is_transactions' not in df.columns and 'is_trades' in df.columns:
        rename_map['is_trades'] = 'is_transactions'
    if 'oos_transactions' not in df.columns and 'oos_trades' in df.columns:
        rename_map['oos_trades'] = 'oos_transactions'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


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
    Short expozice je modelována stejně jako v backtesteru:
    zjednodušená syntetická -1x denní návratnost bez borrow a financing cost.
    """
    fgi_col = row['fgi_col']
    fast    = int(row['fast'])
    slow    = int(row['slow'])

    fg_full   = df_full[fgi_col].to_numpy(dtype=float)
    oos_idx   = df_full.index.get_loc(df_oos.index[0])
    ma_fast   = pd.Series(fg_full).rolling(fast, min_periods=fast).mean().to_numpy()[oos_idx:]
    ma_slow   = pd.Series(fg_full).rolling(slow, min_periods=slow).mean().to_numpy()[oos_idx:]
    prices    = df_oos['SP500_Close'].to_numpy(dtype=float)

    equity, _ = _ma_combined_from_arrays(prices, ma_fast, ma_slow)
    return equity


def _build_equity_ma_long_oos(
    row:     pd.Series,
    df_full: pd.DataFrame,
    df_oos:  pd.DataFrame,
) -> np.ndarray:
    """
    Rekonstruuje equity křivku pro ma_long na OOS datech.
    MA jsou spočítány z celého IS+OOS datasetu — správný warmup na začátku OOS.
    Logika je identická s backtester.ma_long ve stejném unified
    position-based execution frameworku.
    """
    fgi_col = row['fgi_col']
    fast    = int(row['fast'])
    slow    = int(row['slow'])

    fg_full = df_full[fgi_col].to_numpy(dtype=float)
    oos_idx = df_full.index.get_loc(df_oos.index[0])
    ma_fast = pd.Series(fg_full).rolling(fast, min_periods=fast).mean().to_numpy()[oos_idx:]
    ma_slow = pd.Series(fg_full).rolling(slow, min_periods=slow).mean().to_numpy()[oos_idx:]
    prices  = df_oos['SP500_Close'].to_numpy(dtype=float)

    equity, _ = _ma_long_from_arrays(prices, ma_fast, ma_slow)
    return equity


def _select_best_oos_per_strategy(df_oos_res: pd.DataFrame) -> pd.DataFrame:
    """
    Vybere nejlepší OOS konfiguraci pro každou dvojici (fgi_col, strategy).
    Primární kritérium je OOS total return, sekundárně OOS Sharpe.
    """
    best = (
        df_oos_res.sort_values(
            ['fgi_col', 'strategy', 'oos_total_return', 'oos_sharpe'],
            ascending=[True, True, False, False],
        )
        .groupby(['fgi_col', 'strategy'], sort=False)
        .head(1)
        .copy()
    )
    best['strategy'] = pd.Categorical(
        best['strategy'],
        categories=_STRATEGY_ORDER,
        ordered=True,
    )
    return best.sort_values(['fgi_col', 'strategy']).reset_index(drop=True)


def _build_oos_equity(
    row: pd.Series,
    df_full: pd.DataFrame,
    df_oos: pd.DataFrame,
) -> np.ndarray:
    """Vrátí OOS equity křivku přes stejnou logiku jako současný backtester."""
    prices_oos = df_oos['SP500_Close'].to_numpy(dtype=float)
    fg_oos = df_oos[row['fgi_col']].to_numpy(dtype=float)

    if row['strategy'] == 'ma_combined':
        return _build_equity_ma_oos(row, df_full, df_oos)
    if row['strategy'] == 'ma_long':
        return _build_equity_ma_long_oos(row, df_full, df_oos)
    return _build_equity(row, prices_oos, fg_oos)


def _strategy_curve_label(row: pd.Series) -> str:
    """Sestaví čitelný label pro legendu grafu."""
    base = _STRATEGY_LABELS[row['strategy']]
    if row['strategy'] in ('ma_combined', 'ma_long'):
        return f"{base} ({int(row['fast'])}/{int(row['slow'])})"
    return f"{base} ({int(row['entry'])}/{int(row['exit'])})"


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
            f"{r['is_cagr']:+.1f}%",
            f"{r['is_sharpe']:+.2f}",
            f"{r['is_calmar']:+.2f}",
            f"{r['is_max_dd']:+.1f}%",
            f"{int(r['is_transactions'])}",
            f"{r['oos_total_return']:+.1f}%",
            f"{r['oos_cagr']:+.1f}%",
            f"{r['oos_sharpe']:+.2f}",
            f"{r['oos_calmar']:+.2f}",
            f"{r['oos_max_dd']:+.1f}%",
            f"{int(r['oos_transactions'])}",
        ])
    # B&H řádky
    rows.append([
        'Buy & Hold', f'IS {IS_START[:4]}–{IS_END[:4]}', '—',
        f"{bh_is['total_return']:+.1f}%", f"{bh_is['cagr']:+.1f}%", f"{bh_is['sharpe']:+.2f}", f"{bh_is['calmar']:+.2f}", f"{bh_is['max_dd']:+.1f}%",
        '0', '—', '—', '—', '—', '—', '—',
    ])
    rows.append([
        'Buy & Hold', f'OOS {OOS_START[:4]}–{OOS_END[:4]}', '—',
        '—', '—', '—', '—', '—', '—',
        f"{bh_oos['total_return']:+.1f}%", f"{bh_oos['cagr']:+.1f}%", f"{bh_oos['sharpe']:+.2f}", f"{bh_oos['calmar']:+.2f}", f"{bh_oos['max_dd']:+.1f}%", '0',
    ])

    cols = [
        'Strategie', 'FGI', 'Parametry',
        'IS Return', 'IS CAGR', 'IS Sharpe', 'IS Calmar', 'IS MaxDD', 'IS Trades',
        'OOS Return', 'OOS CAGR', 'OOS Sharpe', 'OOS Calmar', 'OOS MaxDD', 'OOS Trades',
    ]

    fig, ax = plt.subplots(figsize=(27, 0.45 * (len(rows) + 2)))
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
# Graf 1 — OOS equity křivky po FGI variantách
# ---------------------------------------------------------------------------

def _plot_oos_equity_by_fgi(
    best_oos: pd.DataFrame,
    df_full: pd.DataFrame,
    df_oos: pd.DataFrame,
    fgi_col: str,
    out_name: str,
    shared_ylim: tuple[float, float] | None = None,
) -> None:
    """
    Vykreslí jeden thesis-ready graf pro konkrétní FGI variantu.
    Zobrazuje pouze nejlepší OOS konfiguraci pro každou strategii + B&H.
    """
    selected = best_oos.loc[best_oos['fgi_col'] == fgi_col].copy()
    if selected.empty:
        raise ValueError(f'Pro {fgi_col} nebyly nalezeny žádné OOS konfigurace.')

    dates_oos = df_oos.index
    prices_oos = df_oos['SP500_Close'].to_numpy(dtype=float)
    bh_eq = INITIAL * prices_oos / prices_oos[0]

    fig, ax = plt.subplots(figsize=(14.2, 7.8))

    for _, row in selected.iterrows():
        equity = _build_oos_equity(row, df_full, df_oos)
        ax.plot(
            dates_oos,
            equity,
            label=_strategy_curve_label(row),
            color=_STRATEGY_COLORS[row['strategy']],
            linewidth=2.2,
            alpha=0.95,
            zorder=3,
        )

    ax.plot(
        dates_oos,
        bh_eq,
        label='Buy & Hold',
        color='black',
        linewidth=3.8,
        alpha=1.0,
        zorder=5,
    )

    ax.axhline(INITIAL, color='gray', linewidth=0.9, linestyle='--', alpha=0.6)
    if shared_ylim is not None:
        ax.set_ylim(shared_ylim)

    ax.set_title(fgi_col, fontsize=18, fontweight='bold')
    ax.set_xlabel('Datum', fontsize=14)
    ax.set_ylabel('Hodnota portfolia (USD)', fontsize=14)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, axis='y', alpha=0.25)
    ax.margins(x=0.01)
    legend = ax.legend(
        loc='upper left',
        bbox_to_anchor=(0.0, -0.14, 1.0, 0.1),
        fontsize=11.8,
        frameon=False,
        ncol=4,
        mode='expand',
        borderaxespad=0.0,
        handlelength=2.4,
        columnspacing=1.2,
        labelspacing=0.8,
    )
    legend._legend_box.align = 'left'
    fig.text(
        0.08, 0.015,
        f'OOS equity křivky, období {OOS_START} až {OOS_END}; legenda uvádí entry/exit, resp. fast/slow.',
        ha='left',
        va='bottom',
        fontsize=11,
        color='dimgray',
    )
    plt.tight_layout(rect=(0, 0.13, 1, 1))

    out = STRATEGY_DIR / out_name
    plt.savefig(out, dpi=260, bbox_inches='tight')
    plt.close()
    print(f'OOS equity graf uložen: {out}')


def plot_oos_equity_split(
    df_oos_res: pd.DataFrame,
    df_full: pd.DataFrame,
    df_oos: pd.DataFrame,
) -> pd.DataFrame:
    """
    Vygeneruje dva samostatné OOS grafy:
      - FGI_Equal + Buy & Hold
      - FGI_OLS + Buy & Hold
    """
    best_oos = _select_best_oos_per_strategy(df_oos_res)
    prices_oos = df_oos['SP500_Close'].to_numpy(dtype=float)
    bh_eq = INITIAL * prices_oos / prices_oos[0]

    y_min = float(np.nanmin(bh_eq))
    y_max = float(np.nanmax(bh_eq))
    for fgi_col in FGI_COLS:
        selected = best_oos.loc[best_oos['fgi_col'] == fgi_col]
        for _, row in selected.iterrows():
            equity = _build_oos_equity(row, df_full, df_oos)
            y_min = min(y_min, float(np.nanmin(equity)))
            y_max = max(y_max, float(np.nanmax(equity)))

    y_pad = (y_max - y_min) * 0.04
    shared_ylim = (
        max(0.0, np.floor((y_min - y_pad) / 1000.0) * 1000.0),
        np.ceil((y_max + y_pad) / 1000.0) * 1000.0,
    )

    output_map = {
        'FGI_Equal': 'oos_equity_equal.png',
        'FGI_OLS': 'oos_equity_ols.png',
    }
    for fgi_col in FGI_COLS:
        _plot_oos_equity_by_fgi(
            best_oos,
            df_full,
            df_oos,
            fgi_col,
            output_map[fgi_col],
            shared_ylim=shared_ylim,
        )

    return best_oos


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
    df_oos_res = _normalize_trade_columns(pd.read_csv(oos_path))
    print(f'Načteno {len(df_oos_res)} konfigurací z {oos_path.name}\n')

    # ---- Terminálový výpis (kopie z 02_out_of_sample pro přehled) ---------
    SEP  = '=' * 160
    SEP2 = '-' * 160
    print(SEP)
    print('IS vs OOS — přehled')
    print(SEP)
    hdr = (
        f'{"Strategie":<23}  {"FGI":<10}  {"Parametry":<20}'
        f'  {"IS Ret":>7}  {"IS CAGR":>8}  {"IS Sh":>6}  {"IS Cal":>7}  {"IS DD":>7}  {"IS Trades":>9}'
        f'  {"OOS Ret":>8}  {"OOS CAGR":>9}  {"OOS Sh":>7}  {"OOS Cal":>8}  {"OOS DD":>8}  {"OOS Trades":>10}'
    )
    print(hdr)
    print(SEP2)
    for _, r in df_oos_res.sort_values(['strategy', 'fgi_col']).iterrows():
        pstr = (f"f={int(r['fast'])} s={int(r['slow'])}"
                if r['strategy'] in ('ma_combined', 'ma_long')
                else f"entry={int(r['entry'])} exit={int(r['exit'])}")
        print(
            f'{r["strategy"]:<23}  {r["fgi_col"]:<10}  {pstr:<20}'
            f'  {r["is_total_return"]:>+6.1f}%  {r["is_cagr"]:>+7.1f}%  {r["is_sharpe"]:>+5.2f}  {r["is_calmar"]:>+6.2f}'
            f'  {r["is_max_dd"]:>+6.1f}%  {int(r["is_transactions"]):>9d}'
            f'  {r["oos_total_return"]:>+7.1f}%  {r["oos_cagr"]:>+8.1f}%  {r["oos_sharpe"]:>+6.2f}  {r["oos_calmar"]:>+7.2f}  {r["oos_max_dd"]:>+7.1f}%'
            f'  {int(r["oos_transactions"]):>10d}'
        )
    print(SEP2)
    print(
        f'{"Buy & Hold IS":<23}  {"":<10}  {"":<20}'
        f'  {bh_is["total_return"]:>+6.1f}%  {bh_is["cagr"]:>+7.1f}%  {bh_is["sharpe"]:>+5.2f}  {bh_is["calmar"]:>+6.2f}  {bh_is["max_dd"]:>+6.1f}%'
        f'  {0:>9d}  {"":>8}  {"":>9}  {"":>7}  {"":>8}  {"":>8}  {"":>10}'
    )
    print(
        f'{"Buy & Hold OOS":<23}  {"":<10}  {"":<20}'
        f'  {"":>7}  {"":>8}  {"":>6}  {"":>7}  {"":>7}  {"":>9}'
        f'  {bh_oos["total_return"]:>+7.1f}%  {bh_oos["cagr"]:>+8.1f}%  {bh_oos["sharpe"]:>+6.2f}  {bh_oos["calmar"]:>+7.2f}  {bh_oos["max_dd"]:>+7.1f}%'
        f'  {0:>10d}'
    )
    print(SEP)
    print()

    # ---- Grafy -------------------------------------------------------------
    plot_results_table(df_oos_res, bh_is, bh_oos)
    best_oos = plot_oos_equity_split(df_oos_res, df_full, df_oos)
    plot_full_period(df_oos_res, df_full, df_is, df_oos)

    print('Vybrané best OOS konfigurace pro thesis grafy:')
    for _, r in best_oos.iterrows():
        if r['strategy'] in ('ma_combined', 'ma_long'):
            pstr = f"fast={int(r['fast'])} slow={int(r['slow'])}"
        else:
            pstr = f"entry={int(r['entry'])} exit={int(r['exit'])}"
        print(
            f"  {r['fgi_col']:<10}  {r['strategy']:<23}  "
            f"{pstr:<22}  OOS return {r['oos_total_return']:+.1f}%"
        )

    print('\nAnalýza dokončena.')


if __name__ == '__main__':
    main()
