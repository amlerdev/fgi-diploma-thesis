"""
Out-of-sample validace — testování best IS parametrů na OOS datech.

Načte grid_results.csv, vybere TOP 3 per (strategie × fgi_col) podle IS
total_return a spustí backtest na OOS datech (2016–2026).

Speciální případ ma_combined a ma_long: rolling MA se počítá z celého datasetu
(IS+OOS dohromady), aby slow MA měla plný warmup na začátku OOS periody.
Trading ale začíná až od OOS_START.

Výstup: oos_results.csv s prefixovanými sloupci is_* a oos_*.

Všechny strategie používají stejný position-based execution model.
Short expozice je zjednodušená syntetická -1x denní návratnost bez borrow
a financing cost.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def load_best_params(grid_path: Path) -> pd.DataFrame:
    """
    Načte grid_results.csv a vrátí TOP 3 per (strategy × fgi_col)
    podle IS total_return.
    """
    df = pd.read_csv(grid_path)
    best = (
        df.sort_values('total_return', ascending=False)
        .groupby(['strategy', 'fgi_col'], sort=False)
        .head(3)
        .reset_index(drop=True)
    )
    return best


def run_oos(
    row:       pd.Series,
    df_full:   pd.DataFrame,
    df_oos:    pd.DataFrame,
) -> dict:
    """
    Spustí OOS backtest pro jednu konfiguraci (řádek z best params).

    Pro ma_combined: fg array pochází z celého datasetu (IS+OOS),
    ale trading startuje až indexem odpovídajícím OOS_START.
    Pro ostatní strategie: použije se pouze OOS slice.

    OOS backtest vždy startuje s cash = INITIAL (žádná přenesená pozice z IS).
    """
    strategy = row['strategy']
    fgi_col  = row['fgi_col']
    fn       = STRATEGIES[strategy]

    prices_oos = df_oos['SP500_Close'].to_numpy(dtype=float)

    if strategy == 'ma_combined':
        fast = int(row['fast'])
        slow = int(row['slow'])

        # Rolling MA z celého datasetu pro správný warmup slow MA
        fg_full       = df_full[fgi_col].to_numpy(dtype=float)
        oos_start_idx = df_full.index.get_loc(df_oos.index[0])
        ma_fast_oos   = pd.Series(fg_full).rolling(fast, min_periods=fast).mean().to_numpy()[oos_start_idx:]
        ma_slow_oos   = pd.Series(fg_full).rolling(slow, min_periods=slow).mean().to_numpy()[oos_start_idx:]

        eq, trades = _ma_combined_oos(prices_oos, ma_fast_oos, ma_slow_oos)
    elif strategy == 'ma_long':
        fast = int(row['fast'])
        slow = int(row['slow'])

        fg_full       = df_full[fgi_col].to_numpy(dtype=float)
        oos_start_idx = df_full.index.get_loc(df_oos.index[0])
        ma_fast_oos   = pd.Series(fg_full).rolling(fast, min_periods=fast).mean().to_numpy()[oos_start_idx:]
        ma_slow_oos   = pd.Series(fg_full).rolling(slow, min_periods=slow).mean().to_numpy()[oos_start_idx:]

        eq, trades = _ma_long_oos(prices_oos, ma_fast_oos, ma_slow_oos)
    else:
        fg_oos = df_oos[fgi_col].to_numpy(dtype=float)
        params = {'entry': int(row['entry']), 'exit': int(row['exit'])}
        eq, trades = fn(prices_oos, fg_oos, **params)

    return compute_metrics(eq, trades)


def _ma_combined_oos(
    prices:  np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    ma_combined s externě předpočítanými MA.
    MA jsou spočítány z celého IS+OOS datasetu — správný warmup na začátku OOS.
    Logika je identická s backtester.ma_combined včetně zjednodušené
    syntetické short expozice (-1x denní výnos bez borrow a financing cost).
    """
    return _ma_combined_from_arrays(prices, ma_fast, ma_slow)


def _ma_long_oos(
    prices:  np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    ma_long s externě předpočítanými MA.
    MA jsou spočítány z celého IS+OOS datasetu — správný warmup na začátku OOS.
    Logika je identická s backtester.ma_long ve stejném unified
    position-based execution frameworku.
    """
    return _ma_long_from_arrays(prices, ma_fast, ma_slow)


def main() -> None:
    # ---- Načtení dat -------------------------------------------------------
    df_full = pd.read_csv(INPUT, index_col='Date', parse_dates=True)
    df_is   = df_full.loc[IS_START:IS_END].copy()
    df_oos  = df_full.loc[OOS_START:OOS_END].copy()

    prices_is  = df_is['SP500_Close'].to_numpy(dtype=float)
    prices_oos = df_oos['SP500_Close'].to_numpy(dtype=float)

    # B&H benchmarky
    bh_is  = compute_metrics(INITIAL * prices_is  / prices_is[0],  0)
    bh_oos = compute_metrics(INITIAL * prices_oos / prices_oos[0], 0)

    # ---- Načtení best IS parametrů ----------------------------------------
    grid_path = STRATEGY_DIR / 'grid_results.csv'
    if not grid_path.exists():
        raise FileNotFoundError(
            f'Soubor {grid_path} nenalezen. Nejprve spusť 01_grid_search.py.'
        )
    best = load_best_params(grid_path)
    print(f'Načteno {len(best)} best konfigurací z {grid_path.name}')
    print(f'OOS perioda: {OOS_START} → {OOS_END}  ({len(df_oos)} dnů)\n')

    # ---- OOS backtest pro každou konfiguraci ------------------------------
    records = []
    for _, row in best.iterrows():
        is_metrics  = {f'is_{k}':  v for k, v in row.items()
                       if k in ('total_return', 'cagr', 'sharpe', 'max_dd', 'calmar', 'trades')}
        oos_metrics = run_oos(row, df_full, df_oos)
        oos_metrics = {f'oos_{k}': v for k, v in oos_metrics.items()}

        # Parametry pro výstupní CSV
        record = {
            'strategy':       row['strategy'],
            'fgi_col':        row['fgi_col'],
            'entry': row.get('entry', float('nan')),
            'exit':  row.get('exit',  float('nan')),
            'fast':           row.get('fast', float('nan')),
            'slow':           row.get('slow', float('nan')),
            **is_metrics,
            **oos_metrics,
        }
        records.append(record)

    df_out = pd.DataFrame(records)

    # ---- Uložení -----------------------------------------------------------
    out_path = STRATEGY_DIR / 'oos_results.csv'
    df_out.to_csv(out_path, index=False)
    print(f'Uloženo: {out_path}')

    # ---- Srovnávací tabulka IS vs OOS -------------------------------------
    SEP  = '=' * 100
    SEP2 = '-' * 100
    print(f'\n{SEP}')
    print('IS vs OOS — srovnání výkonnosti (TOP 3 per strategie × FGI)')
    print(SEP)

    hdr = (
        f'{"Strategie":<23}  {"FGI":<10}  {"Parametry":<20}'
        f'  {"IS Return":>9}  {"IS Sharpe":>9}  {"IS MaxDD":>8}'
        f'  {"OOS Return":>10}  {"OOS Sharpe":>10}  {"OOS MaxDD":>9}'
    )
    print(hdr)
    print(SEP2)

    for _, row in df_out.sort_values(['strategy', 'fgi_col']).iterrows():
        if row['strategy'] in ('ma_combined', 'ma_long'):
            pstr = f"f={int(row['fast'])} s={int(row['slow'])}"
        else:
            pstr = f"entry={int(row['entry'])} exit={int(row['exit'])}"

        print(
            f'{row["strategy"]:<23}  {row["fgi_col"]:<10}  {pstr:<20}'
            f'  {row["is_total_return"]:>+8.1f}%  {row["is_sharpe"]:>+8.2f}'
            f'  {row["is_max_dd"]:>+7.1f}%'
            f'  {row["oos_total_return"]:>+9.1f}%  {row["oos_sharpe"]:>+9.2f}'
            f'  {row["oos_max_dd"]:>+8.1f}%'
        )

    print(SEP2)
    print(
        f'{"Buy & Hold":<23}  {"IS":<10}  {"":<20}'
        f'  {bh_is["total_return"]:>+8.1f}%  {bh_is["sharpe"]:>+8.2f}'
        f'  {bh_is["max_dd"]:>+7.1f}%'
        f'  {"":<10}  {"":<10}  {"":<9}'
    )
    print(
        f'{"Buy & Hold":<23}  {"OOS":<10}  {"":<20}'
        f'  {"":<9}  {"":<9}  {"":<8}'
        f'  {bh_oos["total_return"]:>+9.1f}%  {bh_oos["sharpe"]:>+9.2f}'
        f'  {bh_oos["max_dd"]:>+8.1f}%'
    )
    print(SEP)


if __name__ == '__main__':
    main()
