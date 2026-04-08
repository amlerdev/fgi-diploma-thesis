"""
Grid search — in-sample optimalizace parametrů.

Prochází celý parametrový prostor na IS datech (1998–2015) a ukládá
výsledky do grid_results.csv. Paralelní výpočet přes všechna dostupná
jádra pomocí joblib + tqdm progress bar.

Výstup: grid_results.csv (strategy, fgi_col, entry, exit,
        fast, slow, total_return, cagr, sharpe, max_dd, calmar, trades)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ENTRY_KONTR_RANGE, EXIT_KONTR_RANGE,
    ENTRY_TREND_RANGE, EXIT_TREND_RANGE,
    FGI_COLS, FAST_RANGE, INITIAL, INPUT,
    IS_END, IS_START, MIN_TRADES,
    SLOW_RANGE, STRATEGY_DIR,
)
from backtester import STRATEGIES, compute_metrics

_KONTR_STRATEGIES = ['kontrarian_long', 'kontrarian_combined']
_TREND_STRATEGIES = ['trend_long',      'trend_combined']


# ---------------------------------------------------------------------------
# Sestavení tasků
# ---------------------------------------------------------------------------

def build_tasks() -> list[tuple[str, str, dict]]:
    """
    Vrátí seznam všech (strategy, fgi_col, params) kombinací.

    Kontrariánské: entry ∈ ENTRY_KONTR_RANGE (1–48), exit ∈ EXIT_KONTR_RANGE (50–99)
                   podmínka entry < exit vždy splněna (max entry=48 < 50=min exit).
    Trendové:      entry ∈ ENTRY_TREND_RANGE (50–99), exit ∈ EXIT_TREND_RANGE (1–48)
                   podmínka entry > exit vždy splněna (min entry=50 > 48=max exit).
    MA combined:   fast ∈ FAST_RANGE (1–10), slow ∈ SLOW_RANGE (5–63)
                   podmínka fast < slow vynucena explicitně.
    MA long:       fast ∈ FAST_RANGE (1–10), slow ∈ SLOW_RANGE (5–63)
                   podmínka fast < slow vynucena explicitně.
    """
    tasks: list[tuple[str, str, dict]] = []

    # Kontrariánské strategie
    for fgi_col in FGI_COLS:
        for entry in ENTRY_KONTR_RANGE:
            for exit_ in EXIT_KONTR_RANGE:
                for strategy in _KONTR_STRATEGIES:
                    tasks.append((strategy, fgi_col,
                                  {'entry': entry, 'exit': exit_}))

    # Trendové strategie
    for fgi_col in FGI_COLS:
        for entry in ENTRY_TREND_RANGE:
            for exit_ in EXIT_TREND_RANGE:
                for strategy in _TREND_STRATEGIES:
                    tasks.append((strategy, fgi_col,
                                  {'entry': entry, 'exit': exit_}))

    # MA combined
    for fgi_col in FGI_COLS:
        for fast in FAST_RANGE:
            for slow in SLOW_RANGE:
                if fast >= slow:
                    continue
                tasks.append(('ma_combined', fgi_col,
                               {'fast': fast, 'slow': slow}))

    # MA long
    for fgi_col in FGI_COLS:
        for fast in FAST_RANGE:
            for slow in SLOW_RANGE:
                if fast >= slow:
                    continue
                tasks.append(('ma_long', fgi_col,
                               {'fast': fast, 'slow': slow}))

    return tasks


# ---------------------------------------------------------------------------
# Worker funkce (čistá, bez globálních proměnných)
# ---------------------------------------------------------------------------

def run_task(
    prices:   np.ndarray,
    fg:       np.ndarray,
    strategy: str,
    fgi_col:  str,
    params:   dict,
) -> dict | None:
    """
    Spustí jeden backtest a vrátí dict výsledků.
    Vrátí None pokud trades < MIN_TRADES (výsledek se vyfiltruje).

    Data (prices, fg) jsou předávána jako argumenty —
    žádné globální proměnné ve workerech.
    """
    fn = STRATEGIES[strategy]
    eq, trades = fn(prices, fg, **params)

    if trades < MIN_TRADES:
        return None

    m = compute_metrics(eq, trades)
    return {
        'strategy': strategy,
        'fgi_col':  fgi_col,
        'entry':    params.get('entry', float('nan')),
        'exit':     params.get('exit',  float('nan')),
        'fast':     params.get('fast',  float('nan')),
        'slow':     params.get('slow',  float('nan')),
        **m,
    }


# ---------------------------------------------------------------------------
# Hlavní funkce
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Načtení IS dat ---------------------------------------------------
    df    = pd.read_csv(INPUT, index_col='Date', parse_dates=True)
    df_is = df.loc[IS_START:IS_END].copy()

    prices_is = df_is['SP500_Close'].to_numpy(dtype=float)
    fg_dict   = {col: df_is[col].to_numpy(dtype=float) for col in FGI_COLS}

    # B&H benchmark pro IS
    bh_equity  = INITIAL * prices_is / prices_is[0]
    bh_metrics = compute_metrics(bh_equity, 0)

    # ---- Sestavení tasků --------------------------------------------------
    all_tasks = build_tasks()

    n_kontr = (len(FGI_COLS) * len(_KONTR_STRATEGIES)
               * len(ENTRY_KONTR_RANGE) * len(EXIT_KONTR_RANGE))
    n_trend = (len(FGI_COLS) * len(_TREND_STRATEGIES)
               * len(ENTRY_TREND_RANGE) * len(EXIT_TREND_RANGE))
    n_level = n_kontr + n_trend
    n_ma = 0
    for _col in FGI_COLS:
        for fast in FAST_RANGE:
            for slow in SLOW_RANGE:
                if fast < slow:
                    n_ma += 1
    n_ma *= 2   # dvě MA strategie: ma_combined + ma_long

    print(f'Grid search — IS: {IS_START} → {IS_END}  ({len(df_is)} barů)')
    print(f'  Kontr. tasků      : {n_kontr:>6d}'
          f'  (2 FGI × 2 strat × {len(ENTRY_KONTR_RANGE)} × {len(EXIT_KONTR_RANGE)})')
    print(f'  Trend. tasků      : {n_trend:>6d}'
          f'  (2 FGI × 2 strat × {len(ENTRY_TREND_RANGE)} × {len(EXIT_TREND_RANGE)})')
    print(f'  Level-based celkem: {n_level:>6d}')
    print(f'  MA tasků          : {n_ma:>6d}'
          f'  (2 FGI × 2 strat × {n_ma // (len(FGI_COLS) * 2)} fast/slow kombinací)')
    print(f'  Celkem            : {len(all_tasks):>6d}')
    est_sec = len(all_tasks) * 2e-3   # ~2 ms/task odhad
    print(f'  Odhadovaný čas    : ~{est_sec:.0f}s (záleží na počtu jader)')
    print()

    # ---- Paralelní grid search + tqdm progress bar -----------------------
    t0          = time.perf_counter()
    raw_results = []

    with tqdm(total=len(all_tasks), desc='Grid search',
              unit='task', dynamic_ncols=True) as pbar:
        for result in Parallel(n_jobs=-1, return_as='generator')(
            delayed(run_task)(prices_is, fg_dict[fgi_col], strategy, fgi_col, params)
            for strategy, fgi_col, params in all_tasks
        ):
            raw_results.append(result)
            pbar.update(1)

    elapsed = time.perf_counter() - t0

    # ---- Filtrování a uložení ---------------------------------------------
    results   = [r for r in raw_results if r is not None]
    n_total   = len(all_tasks)
    n_valid   = len(results)
    n_filtered = n_total - n_valid

    print(f'\nDokončeno za {elapsed:.1f}s')
    print(f'Výsledků celkem  : {n_total}')
    print(f'Vyfiltrováno     : {n_filtered}  (trades < {MIN_TRADES})')
    print(f'Zachováno        : {n_valid}')

    df_res   = pd.DataFrame(results)
    out_path = STRATEGY_DIR / 'grid_results.csv'
    df_res.to_csv(out_path, index=False)
    print(f'Uloženo          : {out_path}')

    # ---- TOP 1 per (strategie × FGI) podle total_return ------------------
    SEP  = '=' * 86
    SEP2 = '-' * 86
    print(f'\n{SEP}')
    print('TOP 1 per (strategie × FGI varianta) — IS total_return')
    print(SEP)
    hdr = (f'{"Strategie":<23}  {"FGI":<10}  {"Parametry":<22}'
           f'  {"Return%":>9}  {"CAGR%":>6}  {"Sharpe":>6}  {"MaxDD%":>7}  {"Trades":>6}')
    print(hdr)
    print(SEP2)

    # Pro každou kombinaci (strategy, fgi_col) vyber řádek s nejvyšším total_return
    df_sorted = df_res.sort_values('total_return', ascending=False)
    best      = df_sorted.groupby(['strategy', 'fgi_col'], sort=False).first().reset_index()
    best      = best.sort_values(['strategy', 'fgi_col'])

    for _, row in best.iterrows():
        if row['strategy'] in ('ma_combined', 'ma_long'):
            pstr = f"fast={int(row['fast'])} slow={int(row['slow'])}"
        else:
            pstr = f"entry={int(row['entry'])} exit={int(row['exit'])}"
        print(
            f'{row["strategy"]:<23}  {row["fgi_col"]:<10}  {pstr:<22}'
            f'  {row["total_return"]:>+8.1f}%  {row["cagr"]:>+5.1f}%'
            f'  {row["sharpe"]:>+5.2f}  {row["max_dd"]:>+6.1f}%  {int(row["trades"]):>6d}'
        )

    print(SEP2)
    print(
        f'{"Buy & Hold":<23}  {"IS benchmark":<10}  {"":<22}'
        f'  {bh_metrics["total_return"]:>+8.1f}%  {bh_metrics["cagr"]:>+5.1f}%'
        f'  {bh_metrics["sharpe"]:>+5.2f}  {bh_metrics["max_dd"]:>+6.1f}%  {"0":>6}'
    )
    print(SEP)


if __name__ == '__main__':
    main()
