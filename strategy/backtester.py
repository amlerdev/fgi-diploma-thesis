"""
Backtestovací engine pro FGI Backtesting System v2.

Každá strategie je samostatná a čitelná bez znalosti zbytku kódu.
Short pozice modelovány jako inverzní ETF:
  equity[i] = invested * (entry_price / prices[i])

Parametry strategií:
  kontrarian: entry = práh strachu (1–49), exit = práh chamtivosti (50–100)
  trend:      entry = práh chamtivosti (50–100), exit = práh strachu (1–49)
  ma:         fast, slow = délky klouzavých průměrů sentimentu
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEE, INITIAL


# ---------------------------------------------------------------------------
# Výkonnostní metriky — beze změny
# ---------------------------------------------------------------------------

def compute_metrics(equity: np.ndarray, trades: int) -> dict:
    """
    Vypočítá výkonnostní metriky z equity křivky.

    Parametry
    ---------
    equity : np.ndarray
        Denní hodnota portfolia (délka = počet obchodních dní).
    trades : int
        Celkový počet uzavřených obchodů.

    Vrací
    -----
    dict s klíči: total_return, cagr, sharpe, max_dd, calmar, trades
    """
    n         = len(equity)
    start_val = equity[0]
    end_val   = equity[-1]

    # Celkový výnos v %
    total_return = (end_val / start_val - 1.0) * 100.0

    # CAGR v %
    years = n / 252.0  # 252 trading days per year
    cagr  = ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0

    # Sharpe ratio (annualizovaný, rf = 0) — ignoruje NaN (equity == 0 bary)
    daily_r = np.diff(equity) / equity[:-1]
    std_r   = np.nanstd(daily_r)
    if std_r == 0.0 or np.isnan(std_r):
        sharpe = 0.0
    else:
        sharpe = (np.nanmean(daily_r) / std_r) * np.sqrt(252.0)

    # Maximum drawdown v % (záporné číslo)
    cummax = np.maximum.accumulate(equity)
    dd     = (equity - cummax) / cummax * 100.0
    max_dd = float(dd.min())

    # Calmar ratio
    if max_dd == 0.0:
        calmar = 0.0
    else:
        calmar = cagr / abs(max_dd)

    return {
        'total_return': total_return,
        'cagr':         cagr,
        'sharpe':       sharpe,
        'max_dd':       max_dd,
        'calmar':       calmar,
        'trades':       trades,
    }


# ---------------------------------------------------------------------------
# Strategie 1 — kontrarian_long
# ---------------------------------------------------------------------------

def kontrarian_long(
    prices: np.ndarray,
    fg:     np.ndarray,
    entry:  int,
    exit:   int,
) -> tuple[np.ndarray, int]:
    """
    Kontrariánská long strategie.
    Nakupuje při strachu (FGI < entry), prodává při euforii (FGI > exit).
    Podmínka: entry < exit.
    """
    cash   = float(INITIAL)
    shares = 0.0
    trades = 0
    equity = np.empty(len(prices))

    for i in range(len(prices) - 1):

        # Nakup při extrémním strachu
        if shares == 0.0 and fg[i] < entry:
            shares  = cash * (1 - FEE) / prices[i + 1]
            cash    = 0.0
            trades += 1

        # Prodej při extrémní euforii
        elif shares > 0.0 and fg[i] > exit:
            cash    = shares * prices[i + 1] * (1 - FEE)
            shares  = 0.0
            trades += 1

        equity[i] = cash + shares * prices[i]

        # pojistka: pokud equity <= 0 (selhání strategie / bankrot) → ukonči backtest a vynuluj zbytek
        if equity[i] <= 0.0:
            equity[i:] = 0.0
            return equity, trades

    # Uzavři pozici na konci období
    if shares > 0.0:
        cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity, trades


# ---------------------------------------------------------------------------
# Strategie 2 — kontrarian_combined
# ---------------------------------------------------------------------------

def kontrarian_combined(
    prices: np.ndarray,
    fg:     np.ndarray,
    entry:  int,
    exit:   int,
) -> tuple[np.ndarray, int]:
    """
    Kontrariánská combined strategie — vždy LONG nebo SHORT, nikdy cash po prvním vstupu.
    FGI < entry AND jsme SHORT → přepni na LONG  (2 obchody)
    FGI > exit  AND jsme LONG  → přepni na SHORT (2 obchody)
    První vstup: FGI < entry → LONG, FGI > exit → SHORT, jinak čekej v cash.
    Podmínka: entry < exit.
    """
    cash        = float(INITIAL)
    shares      = 0.0    # počet akcií v long pozici
    invested    = 0.0    # hodnota short pozice (vstupní investice)
    entry_price = 0.0    # vstupní cena short pozice
    trades      = 0
    equity      = np.empty(len(prices))

    for i in range(len(prices) - 1):

        if fg[i] < entry and invested > 0.0:
            # Strach — přepni SHORT → LONG: uzavři short...
            cash        = invested * (entry_price / prices[i + 1]) * (1 - FEE)
            invested    = 0.0
            entry_price = 0.0
            trades     += 1
            # ...pak nakup long
            shares  = cash * (1 - FEE) / prices[i + 1]
            cash    = 0.0
            trades += 1

        elif fg[i] > exit and shares > 0.0:
            # Euforie — přepni LONG → SHORT: prodej long...
            cash    = shares * prices[i + 1] * (1 - FEE)
            shares  = 0.0
            trades += 1
            # ...pak otevři short
            invested    = cash * (1 - FEE)
            entry_price = prices[i + 1]
            cash        = 0.0
            trades     += 1

        elif shares == 0.0 and invested == 0.0:
            # Před prvním obchodem jsme v cash — čekáme na první signál (tato větev platí jen jednou)
            if fg[i] < entry:
                # První vstup: strach → jdi LONG
                shares  = cash * (1 - FEE) / prices[i + 1]
                cash    = 0.0
                trades += 1
            elif fg[i] > exit:
                # První vstup: euforie → jdi SHORT
                invested    = cash * (1 - FEE)
                entry_price = prices[i + 1]
                cash        = 0.0
                trades     += 1
            # else: neutrální — zůstaň v cash a čekej dál

        # Aktuální hodnota portfolia
        if shares > 0.0:
            equity[i] = shares * prices[i]
        elif invested > 0.0:
            equity[i] = invested * (entry_price / prices[i])   # inverzní ETF model
        else:
            equity[i] = cash

        # pojistka: pokud equity <= 0 (selhání strategie / bankrot) → ukonči backtest a vynuluj zbytek
        if equity[i] <= 0.0:
            equity[i:] = 0.0
            return equity, trades

    # Uzavři pozici na konci období
    if shares > 0.0:
        cash = shares * prices[-1] * (1 - FEE)
    elif invested > 0.0:
        cash = invested * (entry_price / prices[-1]) * (1 - FEE)
    equity[-1] = cash
    return equity, trades


# ---------------------------------------------------------------------------
# Strategie 3 — trend_long
# ---------------------------------------------------------------------------

def trend_long(
    prices: np.ndarray,
    fg:     np.ndarray,
    entry:  int,
    exit:   int,
) -> tuple[np.ndarray, int]:
    """
    Trendová long strategie.
    Nakupuje při euforii (FGI > entry), prodává při strachu (FGI < exit).
    Podmínka: entry > exit.
    """
    cash   = float(INITIAL)
    shares = 0.0
    trades = 0
    equity = np.empty(len(prices))

    for i in range(len(prices) - 1):

        # Nakup při euforii — trend pokračuje nahoru
        if shares == 0.0 and fg[i] > entry:
            shares  = cash * (1 - FEE) / prices[i + 1]
            cash    = 0.0
            trades += 1

        # Prodej při strachu — trend se obrací dolů
        elif shares > 0.0 and fg[i] < exit:
            cash    = shares * prices[i + 1] * (1 - FEE)
            shares  = 0.0
            trades += 1

        equity[i] = cash + shares * prices[i]

        # pojistka: pokud equity <= 0 (selhání strategie / bankrot) → ukonči backtest a vynuluj zbytek
        if equity[i] <= 0.0:
            equity[i:] = 0.0
            return equity, trades

    # Uzavři pozici na konci období
    if shares > 0.0:
        cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity, trades


# ---------------------------------------------------------------------------
# Strategie 4 — trend_combined
# ---------------------------------------------------------------------------

def trend_combined(
    prices: np.ndarray,
    fg:     np.ndarray,
    entry:  int,
    exit:   int,
) -> tuple[np.ndarray, int]:
    """
    Trendová combined strategie — vždy LONG nebo SHORT, nikdy cash po prvním vstupu.
    FGI > entry AND jsme SHORT → přepni na LONG  (2 obchody)
    FGI < exit  AND jsme LONG  → přepni na SHORT (2 obchody)
    První vstup: FGI > entry → LONG, FGI < exit → SHORT, jinak čekej v cash.
    Podmínka: entry > exit.
    """
    cash        = float(INITIAL)
    shares      = 0.0    # počet akcií v long pozici
    invested    = 0.0    # hodnota short pozice (vstupní investice)
    entry_price = 0.0    # vstupní cena short pozice
    trades      = 0
    equity      = np.empty(len(prices))

    for i in range(len(prices) - 1):

        if fg[i] > entry and invested > 0.0:
            # Euforie — přepni SHORT → LONG: uzavři short...
            cash        = invested * (entry_price / prices[i + 1]) * (1 - FEE)
            invested    = 0.0
            entry_price = 0.0
            trades     += 1
            # ...pak nakup long
            shares  = cash * (1 - FEE) / prices[i + 1]
            cash    = 0.0
            trades += 1

        elif fg[i] < exit and shares > 0.0:
            # Strach — přepni LONG → SHORT: prodej long...
            cash    = shares * prices[i + 1] * (1 - FEE)
            shares  = 0.0
            trades += 1
            # ...pak otevři short
            invested    = cash * (1 - FEE)
            entry_price = prices[i + 1]
            cash        = 0.0
            trades     += 1

        elif shares == 0.0 and invested == 0.0:
            # Před prvním obchodem jsme v cash — čekáme na první signál (tato větev platí jen jednou)
            if fg[i] > entry:
                # První vstup: euforie → jdi LONG
                shares  = cash * (1 - FEE) / prices[i + 1]
                cash    = 0.0
                trades += 1
            elif fg[i] < exit:
                # První vstup: strach → jdi SHORT
                invested    = cash * (1 - FEE)
                entry_price = prices[i + 1]
                cash        = 0.0
                trades     += 1
            # else: neutrální — zůstaň v cash a čekej dál

        # Aktuální hodnota portfolia
        if shares > 0.0:
            equity[i] = shares * prices[i]
        elif invested > 0.0:
            equity[i] = invested * (entry_price / prices[i])   # inverzní ETF model
        else:
            equity[i] = cash

        # pojistka: pokud equity <= 0 (selhání strategie / bankrot) → ukonči backtest a vynuluj zbytek
        if equity[i] <= 0.0:
            equity[i:] = 0.0
            return equity, trades

    # Uzavři pozici na konci období
    if shares > 0.0:
        cash = shares * prices[-1] * (1 - FEE)
    elif invested > 0.0:
        cash = invested * (entry_price / prices[-1]) * (1 - FEE)
    equity[-1] = cash
    return equity, trades


# ---------------------------------------------------------------------------
# Strategie 5 — ma_long
# ---------------------------------------------------------------------------

def ma_long(
    prices: np.ndarray,
    fg:     np.ndarray,
    fast:   int,
    slow:   int,
) -> tuple[np.ndarray, int]:
    """
    MA crossover sentimentu — pouze long strategie.
    Nakupuje když fast MA > slow MA, prodává do cash když fast MA < slow MA.
    Při rovnosti MA drží stávající pozici. Nikdy nedrží short.
    Warmup: prvních `slow` barů zůstává v cash (slow MA není ještě inicializována).
    """
    # Předpočítej klouzavé průměry sentimentu — mimo smyčku
    ma_fast = pd.Series(fg).rolling(fast, min_periods=fast).mean().to_numpy()
    ma_slow = pd.Series(fg).rolling(slow, min_periods=slow).mean().to_numpy()

    cash   = float(INITIAL)
    shares = 0.0
    trades = 0
    equity = np.empty(len(prices))

    for i in range(len(prices) - 1):

        # Warmup — slow MA ještě nemá dostatek dat
        if np.isnan(ma_slow[i]):
            equity[i] = cash
            continue

        if ma_fast[i] > ma_slow[i]:
            # Fast MA nad slow MA — sentiment roste, nakup pokud nejsme LONG
            if shares == 0.0:
                shares  = cash * (1 - FEE) / prices[i + 1]
                cash    = 0.0
                trades += 1
            # else: už jsme LONG — nic neděláme

        elif ma_fast[i] < ma_slow[i]:
            # Fast MA pod slow MA — sentiment klesá, prodej do cash pokud jsme LONG
            if shares > 0.0:
                cash    = shares * prices[i + 1] * (1 - FEE)
                shares  = 0.0
                trades += 1
            # else: už jsme v cash — nic neděláme

        # else: fast MA == slow MA — drž stávající pozici beze změny

        equity[i] = cash + shares * prices[i]

        # pojistka: pokud equity <= 0 (selhání strategie / bankrot) → ukonči backtest a vynuluj zbytek
        if equity[i] <= 0.0:
            equity[i:] = 0.0
            return equity, trades

    # Uzavři pozici na konci období
    if shares > 0.0:
        cash = shares * prices[-1] * (1 - FEE)
    equity[-1] = cash
    return equity, trades


# ---------------------------------------------------------------------------
# Strategie 6 — ma_combined
# ---------------------------------------------------------------------------

def ma_combined(
    prices: np.ndarray,
    fg:     np.ndarray,
    fast:   int,
    slow:   int,
) -> tuple[np.ndarray, int]:
    """
    MA crossover sentimentu — long+short strategie.
    Nakupuje když fast MA > slow MA, shortuje když fast MA < slow MA.
    Při rovnosti MA drží stávající pozici. Nikdy nedrží cash po inicializaci.
    Warmup: prvních `slow` barů zůstává v cash (slow MA není ještě inicializována).
    """
    # Předpočítej klouzavé průměry sentimentu — mimo smyčku
    ma_fast = pd.Series(fg).rolling(fast, min_periods=fast).mean().to_numpy()
    ma_slow = pd.Series(fg).rolling(slow, min_periods=slow).mean().to_numpy()

    cash        = float(INITIAL)
    shares      = 0.0    # počet akcií v long pozici
    invested    = 0.0    # hodnota short pozice (vstupní investice)
    entry_price = 0.0    # vstupní cena short pozice
    trades      = 0
    equity      = np.empty(len(prices))

    for i in range(len(prices) - 1):

        # Warmup — slow MA ještě nemá dostatek dat
        if np.isnan(ma_slow[i]):
            equity[i] = cash
            continue

        if ma_fast[i] > ma_slow[i]:
            # Fast MA nad slow MA — sentiment roste, chceme být LONG

            if invested > 0.0:
                # Přechod SHORT → LONG: uzavři short...
                cash        = invested * (entry_price / prices[i + 1]) * (1 - FEE)
                invested    = 0.0
                entry_price = 0.0
                trades     += 1
                # ...a nakup long
                shares  = cash * (1 - FEE) / prices[i + 1]
                cash    = 0.0
                trades += 1

            elif shares == 0.0:
                # První vstup do LONG po warmup
                shares  = cash * (1 - FEE) / prices[i + 1]
                cash    = 0.0
                trades += 1

            # else: už jsme LONG — nic neděláme

        elif ma_fast[i] < ma_slow[i]:
            # Fast MA pod slow MA — sentiment klesá, chceme být SHORT

            if shares > 0.0:
                # Přechod LONG → SHORT: prodej long...
                cash    = shares * prices[i + 1] * (1 - FEE)
                shares  = 0.0
                trades += 1
                # ...a otevři short
                invested    = cash * (1 - FEE)
                entry_price = prices[i + 1]
                cash        = 0.0
                trades     += 1

            elif invested == 0.0:
                # První vstup do SHORT po warmup
                invested    = cash * (1 - FEE)
                entry_price = prices[i + 1]
                cash        = 0.0
                trades     += 1

            # else: už jsme SHORT — nic neděláme

        # else: fast MA == slow MA — drž stávající pozici beze změny

        # Aktuální hodnota portfolia
        if shares > 0.0:
            equity[i] = shares * prices[i]
        elif invested > 0.0:
            equity[i] = invested * (entry_price / prices[i])   # inverzní ETF model
        else:
            equity[i] = cash

        # pojistka: pokud equity <= 0 (selhání strategie / bankrot) → ukonči backtest a vynuluj zbytek
        if equity[i] <= 0.0:
            equity[i:] = 0.0
            return equity, trades

    # Uzavři pozici na konci období
    if shares > 0.0:
        cash = shares * prices[-1] * (1 - FEE)
    elif invested > 0.0:
        cash = invested * (entry_price / prices[-1]) * (1 - FEE)
    equity[-1] = cash
    return equity, trades


# ---------------------------------------------------------------------------
# Dispatcher registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, callable] = {
    'kontrarian_long':     kontrarian_long,
    'kontrarian_combined': kontrarian_combined,
    'trend_long':          trend_long,
    'trend_combined':      trend_combined,
    'ma_long':             ma_long,
    'ma_combined':         ma_combined,
}


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    rng = np.random.default_rng(42)
    n   = 500

    fake_prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))
    fake_fg     = rng.uniform(0, 100, n)

    print(f'{"Strategie":<25}  {"return":>8}  {"trades":>6}  {"sharpe":>7}  {"max_dd":>8}')
    print('-' * 60)
    for name, fn in STRATEGIES.items():
        if name in ('ma_long', 'ma_combined'):
            eq, tr = fn(fake_prices, fake_fg, fast=10, slow=50)
        else:
            eq, tr = fn(fake_prices, fake_fg, entry=25, exit=75)
        m = compute_metrics(eq, tr)
        print(
            f'{name:<25}  {m["total_return"]:>+7.1f}%  {m["trades"]:>6d}'
            f'  {m["sharpe"]:>+6.2f}  {m["max_dd"]:>+7.1f}%'
        )

    print('\nSMOKE TEST OK')
