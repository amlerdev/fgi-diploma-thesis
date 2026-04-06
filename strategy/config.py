"""Sdílené konstanty pro FGI Backtesting System v2."""

from pathlib import Path

STRATEGY_DIR = Path(__file__).resolve().parent
INPUT        = STRATEGY_DIR.parent / 'index' / 'fgi_index_final.csv'

# Časové periody
IS_START  = '1998-01-01'
IS_END    = '2015-12-31'
OOS_START = '2016-01-01'
OOS_END   = '2026-12-31'

INITIAL = 10_000.0   # počáteční kapitál
FEE     = 0.001      # 0.1 % per obchod
FGI_COLS = ['FGI_Equal', 'FGI_OLS']

# Grid search rozsahy — level-based strategie
# Kontrariánské: entry (strach, 1–49) < exit (chamtivost, 50–100) — vždy splněno
ENTRY_KONTR_RANGE = list(range(1,  49, 1))
EXIT_KONTR_RANGE  = list(range(50, 100, 1))
# Trendové: entry (chamtivost, 50–100) > exit (strach, 1–49) — vždy splněno
ENTRY_TREND_RANGE = list(range(50, 100, 1))
EXIT_TREND_RANGE  = list(range(1,  49, 1))

# Grid search rozsahy — MA combined
FAST_RANGE = list(range(1,  35, 1))
SLOW_RANGE = list(range(20, 301, 5))
# Podmínka fast < slow vynucena při sestavování tasků

# Minimální počet obchodů pro zahrnutí výsledku
MIN_TRADES = 5
