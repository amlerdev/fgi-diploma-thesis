"""
Backtestovací engine pro FGI Backtesting System v2.

Každá strategie je zapsaná co nejčitelněji, skoro jako pseudokód.
Všechny strategie používají stejný jednoduchý position-based execution model:
  LONG  -> equity_next = equity_current * (1 + r)
  SHORT -> equity_next = equity_current * (1 - r)
  CASH  -> equity_next = equity_current

Short expozice je zjednodušená syntetická -1x denní návratnost.
Model záměrně neobsahuje borrow cost, financing cost ani další realistické vrstvy.

Parametry strategií:
  kontrarian: entry = práh strachu (1-49), exit = práh chamtivosti (50-100)
  trend:      entry = práh chamtivosti (50-100), exit = práh strachu (1-49)
  ma:         fast, slow = délky klouzavých průměrů sentimentu
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import FEE, INITIAL


LONG = 1
CASH = 0
SHORT = -1


# ---------------------------------------------------------------------------
# Výkonnostní metriky
# ---------------------------------------------------------------------------

def compute_metrics(equity: np.ndarray, transactions: int) -> dict:
    """
    Vypočítá výkonnostní metriky z equity křivky.

    Parametry
    ---------
    equity : np.ndarray
        Denní hodnota portfolia.
    transactions : int
        Celkový počet transakčních jednotek při změnách pozice.

    Vrací
    -----
    dict s klíči: total_return, cagr, sharpe, max_dd, calmar, transactions
    """
    n = len(equity)
    start_val = equity[0]
    end_val = equity[-1]

    total_return = (end_val / start_val - 1.0) * 100.0

    years = n / 252.0
    cagr = ((end_val / start_val) ** (1.0 / years) - 1.0) * 100.0

    daily_r = np.diff(equity) / equity[:-1]
    std_r = np.nanstd(daily_r)
    if std_r == 0.0 or np.isnan(std_r):
        sharpe = 0.0
    else:
        sharpe = (np.nanmean(daily_r) / std_r) * np.sqrt(252.0)

    cummax = np.maximum.accumulate(equity)
    dd = (equity - cummax) / cummax * 100.0
    max_dd = float(dd.min())

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
        'transactions': transactions,
    }


# ---------------------------------------------------------------------------
# Strategie 1 - kontrarian_long
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
    position = CASH
    transactions = 0
    equity = np.empty(len(prices))
    equity[0] = float(INITIAL)

    for i in range(len(prices) - 1):
        is_last_execution = i == len(prices) - 2
        # Jsme v posledním dni smyčky, takže po tomto kroku už
        # nebudeme otevírat novou pozici.

        # 1) Nejdřív spočítáme hodnotu portfolia pro další den podle pozice,
        #    kterou jsme drželi od dne i do dne i+1.
        if position == CASH:
            equity[i + 1] = equity[i]
        elif position == LONG:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 + daily_return)
        elif position == SHORT:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 - daily_return)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        # 2) Dnešní signál říká, co chceme držet od dalšího dne.
        desired_position = position
        if fg[i] < entry:
            desired_position = LONG
        elif fg[i] > exit:
            desired_position = CASH

        # 3) V posledním dni smyčky už novou pozici neotvíráme.
        if desired_position == position:
            next_position = position
        elif is_last_execution:
            next_position = CASH
        else:
            next_position = desired_position

        # 4) Fee platíme jen při změně pozice.
        fee_units = 0
        if position == CASH and next_position == LONG:
            fee_units = 1
        elif position == CASH and next_position == SHORT:
            fee_units = 1
        elif position == LONG and next_position == CASH:
            fee_units = 1
        elif position == SHORT and next_position == CASH:
            fee_units = 1
        elif position == LONG and next_position == SHORT:
            fee_units = 2
        elif position == SHORT and next_position == LONG:
            fee_units = 2

        if fee_units == 1:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
        elif fee_units == 2:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        transactions += fee_units
        position = next_position

        # 5) Pokud equity spadne na nulu, backtest ukončíme.
        if equity[i + 1] <= 0.0:
            equity[i + 1:] = 0.0
            return equity, transactions

    # 6) Pokud jsme na konci stále v pozici, zavřeme ji za 1 fee.
    if position != CASH:
        equity[-1] = equity[-1] * (1.0 - FEE)
        if equity[-1] < 0.0:
            equity[-1] = 0.0

    return equity, transactions


# ---------------------------------------------------------------------------
# Strategie 2 - kontrarian_combined
# ---------------------------------------------------------------------------

def kontrarian_combined(
    prices: np.ndarray,
    fg:     np.ndarray,
    entry:  int,
    exit:   int,
) -> tuple[np.ndarray, int]:
    """
    Kontrariánská combined strategie.
    Jde LONG při strachu (FGI < entry).
    Jde SHORT při euforii (FGI > exit).
    Jinak drží stávající pozici.

    Podmínka: entry < exit.

    Short expozice je modelována jako zjednodušená syntetická -1x denní
    návratnost. Model záměrně neobsahuje borrow cost ani financing cost.
    """
    position = CASH
    transactions = 0
    equity = np.empty(len(prices))
    equity[0] = float(INITIAL)

    for i in range(len(prices) - 1):
        is_last_execution = i == len(prices) - 2
        # Jsme v posledním dni smyčky, takže po tomto kroku už
        # nebudeme otevírat novou pozici.

        if position == CASH:
            equity[i + 1] = equity[i]
        elif position == LONG:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 + daily_return)
        elif position == SHORT:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 - daily_return)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        desired_position = position
        if fg[i] < entry:
            desired_position = LONG
        elif fg[i] > exit:
            desired_position = SHORT

        if desired_position == position:
            next_position = position
        elif is_last_execution:
            next_position = CASH
        else:
            next_position = desired_position

        fee_units = 0
        if position == CASH and next_position == LONG:
            fee_units = 1
        elif position == CASH and next_position == SHORT:
            fee_units = 1
        elif position == LONG and next_position == CASH:
            fee_units = 1
        elif position == SHORT and next_position == CASH:
            fee_units = 1
        elif position == LONG and next_position == SHORT:
            fee_units = 2
        elif position == SHORT and next_position == LONG:
            fee_units = 2

        if fee_units == 1:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
        elif fee_units == 2:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        transactions += fee_units
        position = next_position

        if equity[i + 1] <= 0.0:
            equity[i + 1:] = 0.0
            return equity, transactions

    if position != CASH:
        equity[-1] = equity[-1] * (1.0 - FEE)
        if equity[-1] < 0.0:
            equity[-1] = 0.0

    return equity, transactions


# ---------------------------------------------------------------------------
# Strategie 3 - trend_long
# ---------------------------------------------------------------------------

def trend_long(
    prices: np.ndarray,
    fg:     np.ndarray,
    entry:  int,
    exit:   int,
) -> tuple[np.ndarray, int]:
    """
    Trendová long strategie.
    Nakupuje při silném sentimentu (FGI > entry), prodává do cash při
    slabém sentimentu (FGI < exit).

    Podmínka: entry > exit.
    """
    position = CASH
    transactions = 0
    equity = np.empty(len(prices))
    equity[0] = float(INITIAL)

    for i in range(len(prices) - 1):
        is_last_execution = i == len(prices) - 2
        # Jsme v posledním dni smyčky, takže po tomto kroku už
        # nebudeme otevírat novou pozici.

        if position == CASH:
            equity[i + 1] = equity[i]
        elif position == LONG:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 + daily_return)
        elif position == SHORT:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 - daily_return)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        desired_position = position
        if fg[i] > entry:
            desired_position = LONG
        elif fg[i] < exit:
            desired_position = CASH

        if desired_position == position:
            next_position = position
        elif is_last_execution:
            next_position = CASH
        else:
            next_position = desired_position

        fee_units = 0
        if position == CASH and next_position == LONG:
            fee_units = 1
        elif position == CASH and next_position == SHORT:
            fee_units = 1
        elif position == LONG and next_position == CASH:
            fee_units = 1
        elif position == SHORT and next_position == CASH:
            fee_units = 1
        elif position == LONG and next_position == SHORT:
            fee_units = 2
        elif position == SHORT and next_position == LONG:
            fee_units = 2

        if fee_units == 1:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
        elif fee_units == 2:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        transactions += fee_units
        position = next_position

        if equity[i + 1] <= 0.0:
            equity[i + 1:] = 0.0
            return equity, transactions

    if position != CASH:
        equity[-1] = equity[-1] * (1.0 - FEE)
        if equity[-1] < 0.0:
            equity[-1] = 0.0

    return equity, transactions


# ---------------------------------------------------------------------------
# Strategie 4 - trend_combined
# ---------------------------------------------------------------------------

def trend_combined(
    prices: np.ndarray,
    fg:     np.ndarray,
    entry:  int,
    exit:   int,
) -> tuple[np.ndarray, int]:
    """
    Trendová combined strategie.
    Jde LONG při silném sentimentu (FGI > entry).
    Jde SHORT při slabém sentimentu (FGI < exit).
    Jinak drží stávající pozici.

    Podmínka: entry > exit.

    Short expozice je modelována jako zjednodušená syntetická -1x denní
    návratnost. Model záměrně neobsahuje borrow cost ani financing cost.
    """
    position = CASH
    transactions = 0
    equity = np.empty(len(prices))
    equity[0] = float(INITIAL)

    for i in range(len(prices) - 1):
        is_last_execution = i == len(prices) - 2
        # Jsme v posledním dni smyčky, takže po tomto kroku už
        # nebudeme otevírat novou pozici.

        if position == CASH:
            equity[i + 1] = equity[i]
        elif position == LONG:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 + daily_return)
        elif position == SHORT:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 - daily_return)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        desired_position = position
        if fg[i] > entry:
            desired_position = LONG
        elif fg[i] < exit:
            desired_position = SHORT

        if desired_position == position:
            next_position = position
        elif is_last_execution:
            next_position = CASH
        else:
            next_position = desired_position

        fee_units = 0
        if position == CASH and next_position == LONG:
            fee_units = 1
        elif position == CASH and next_position == SHORT:
            fee_units = 1
        elif position == LONG and next_position == CASH:
            fee_units = 1
        elif position == SHORT and next_position == CASH:
            fee_units = 1
        elif position == LONG and next_position == SHORT:
            fee_units = 2
        elif position == SHORT and next_position == LONG:
            fee_units = 2

        if fee_units == 1:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
        elif fee_units == 2:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        transactions += fee_units
        position = next_position

        if equity[i + 1] <= 0.0:
            equity[i + 1:] = 0.0
            return equity, transactions

    if position != CASH:
        equity[-1] = equity[-1] * (1.0 - FEE)
        if equity[-1] < 0.0:
            equity[-1] = 0.0

    return equity, transactions


# ---------------------------------------------------------------------------
# Strategie 5 - ma_long
# ---------------------------------------------------------------------------

def ma_long(
    prices: np.ndarray,
    fg:     np.ndarray,
    fast:   int,
    slow:   int,
) -> tuple[np.ndarray, int]:
    """
    MA crossover sentimentu - pouze long strategie.
    Jde LONG když fast MA > slow MA.
    Jde do CASH když fast MA < slow MA.
    Při rovnosti drží stávající pozici.

    Warmup: zůstává v cash, dokud slow MA ještě neexistuje.
    """
    ma_fast = pd.Series(fg).rolling(fast, min_periods=fast).mean().to_numpy()
    ma_slow = pd.Series(fg).rolling(slow, min_periods=slow).mean().to_numpy()

    return _ma_long_from_arrays(prices, ma_fast, ma_slow)


# ---------------------------------------------------------------------------
# Strategie 6 - ma_combined
# ---------------------------------------------------------------------------

def ma_combined(
    prices: np.ndarray,
    fg:     np.ndarray,
    fast:   int,
    slow:   int,
) -> tuple[np.ndarray, int]:
    """
    MA crossover sentimentu - long + short strategie.
    Jde LONG když fast MA > slow MA.
    Jde SHORT když fast MA < slow MA.
    Při rovnosti drží stávající pozici.

    Warmup: zůstává v cash, dokud slow MA ještě neexistuje.

    Short expozice je modelována jako zjednodušená syntetická -1x denní
    návratnost. Model záměrně neobsahuje borrow cost ani financing cost.
    """
    ma_fast = pd.Series(fg).rolling(fast, min_periods=fast).mean().to_numpy()
    ma_slow = pd.Series(fg).rolling(slow, min_periods=slow).mean().to_numpy()

    return _ma_combined_from_arrays(prices, ma_fast, ma_slow)


def _ma_combined_from_arrays(
    prices:  np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Interní helper pro ma_combined a jeho OOS / plotting rekonstrukci.

    Short expozice je modelována jako zjednodušená syntetická -1x denní
    návratnost. Model záměrně neobsahuje borrow cost ani financing cost.
    Při rovnosti ma_fast == ma_slow se desired_position nemění, takže
    strategie ponechává předchozí pozici beze změny.
    """
    position = CASH
    transactions = 0
    equity = np.empty(len(prices))
    equity[0] = float(INITIAL)

    for i in range(len(prices) - 1):
        is_last_execution = i == len(prices) - 2
        # Jsme v posledním dni smyčky, takže po tomto kroku už
        # nebudeme otevírat novou pozici.

        if position == CASH:
            equity[i + 1] = equity[i]
        elif position == LONG:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 + daily_return)
        elif position == SHORT:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 - daily_return)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        # Defaultně držíme předchozí pozici; při rovnosti MA tedy
        # explicitně nedochází ke změně pozice.
        desired_position = position
        if np.isnan(ma_slow[i]):
            desired_position = CASH
        elif ma_fast[i] > ma_slow[i]:
            desired_position = LONG
        elif ma_fast[i] < ma_slow[i]:
            desired_position = SHORT

        if desired_position == position:
            next_position = position
        elif is_last_execution:
            next_position = CASH
        else:
            next_position = desired_position

        fee_units = 0
        if position == CASH and next_position == LONG:
            fee_units = 1
        elif position == CASH and next_position == SHORT:
            fee_units = 1
        elif position == LONG and next_position == CASH:
            fee_units = 1
        elif position == SHORT and next_position == CASH:
            fee_units = 1
        elif position == LONG and next_position == SHORT:
            fee_units = 2
        elif position == SHORT and next_position == LONG:
            fee_units = 2

        if fee_units == 1:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
        elif fee_units == 2:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        transactions += fee_units
        position = next_position

        if equity[i + 1] <= 0.0:
            equity[i + 1:] = 0.0
            return equity, transactions

    if position != CASH:
        equity[-1] = equity[-1] * (1.0 - FEE)
        if equity[-1] < 0.0:
            equity[-1] = 0.0

    return equity, transactions


def _ma_long_from_arrays(
    prices:  np.ndarray,
    ma_fast: np.ndarray,
    ma_slow: np.ndarray,
) -> tuple[np.ndarray, int]:
    """
    Interní helper pro ma_long a jeho OOS / plotting rekonstrukci.

    Long expozice používá stejný position-based framework jako ostatní
    strategie. Warmup explicitně drží cash a short větev se nepoužívá.
    Při rovnosti ma_fast == ma_slow se desired_position nemění, takže
    strategie ponechává předchozí pozici beze změny.
    """
    position = CASH
    transactions = 0
    equity = np.empty(len(prices))
    equity[0] = float(INITIAL)

    for i in range(len(prices) - 1):
        is_last_execution = i == len(prices) - 2
        # Jsme v posledním dni smyčky, takže po tomto kroku už
        # nebudeme otevírat novou pozici.

        if position == CASH:
            equity[i + 1] = equity[i]
        elif position == LONG:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 + daily_return)
        elif position == SHORT:
            daily_return = (prices[i + 1] - prices[i]) / prices[i]
            equity[i + 1] = equity[i] * (1.0 - daily_return)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        # Defaultně držíme předchozí pozici; při rovnosti MA tedy
        # explicitně nedochází ke změně pozice.
        desired_position = position
        if np.isnan(ma_slow[i]):
            desired_position = CASH
        elif ma_fast[i] > ma_slow[i]:
            desired_position = LONG
        elif ma_fast[i] < ma_slow[i]:
            desired_position = CASH

        if desired_position == position:
            next_position = position
        elif is_last_execution:
            next_position = CASH
        else:
            next_position = desired_position

        fee_units = 0
        if position == CASH and next_position == LONG:
            fee_units = 1
        elif position == CASH and next_position == SHORT:
            fee_units = 1
        elif position == LONG and next_position == CASH:
            fee_units = 1
        elif position == SHORT and next_position == CASH:
            fee_units = 1
        elif position == LONG and next_position == SHORT:
            fee_units = 2
        elif position == SHORT and next_position == LONG:
            fee_units = 2

        if fee_units == 1:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
        elif fee_units == 2:
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)
            equity[i + 1] = equity[i + 1] * (1.0 - FEE)

        if equity[i + 1] < 0.0:
            equity[i + 1] = 0.0

        transactions += fee_units
        position = next_position

        if equity[i + 1] <= 0.0:
            equity[i + 1:] = 0.0
            return equity, transactions

    if position != CASH:
        equity[-1] = equity[-1] * (1.0 - FEE)
        if equity[-1] < 0.0:
            equity[-1] = 0.0

    return equity, transactions


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
    n = 500

    fake_prices = 100.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, n))
    fake_fg = rng.uniform(0, 100, n)

    print(f'{"Strategie":<25}  {"return":>8}  {"transactions":>12}  {"sharpe":>7}  {"max_dd":>8}')
    print('-' * 60)
    for name, fn in STRATEGIES.items():
        if name in ('ma_long', 'ma_combined'):
            eq, tx = fn(fake_prices, fake_fg, fast=10, slow=50)
        else:
            eq, tx = fn(fake_prices, fake_fg, entry=25, exit=75)
        m = compute_metrics(eq, tx)
        print(
            f'{name:<25}  {m["total_return"]:>+7.1f}%  {m["transactions"]:>12d}'
            f'  {m["sharpe"]:>+6.2f}  {m["max_dd"]:>+7.1f}%'
        )

    print('\nSMOKE TEST OK')
