# Fear & Greed Index — Diplomová práce

Rekonstrukce CNN Fear & Greed Indexu ze 7 veřejných datových zdrojů (1998–2026)
a backtesting algoritmických obchodních strategií na S&P 500 Total Return.

**Autor:** Petr Amler
**Instituce:** VŠB-TU Ostrava, Ekonomická fakulta, katedra financí
**Typ:** Diplomová práce (navazuje na semestrální práci)

---

## Struktura projektu

```
code/
│
├── data/                            # Vstupní data
│   ├── fear_greed_historical.csv    # CNN F&G historická data (2011–2026)
│   └── fear_and_greed_dwn.py        # Downloader CNN dat
│
├── components/                      # 7 komponent F&G indexu
│   ├── 01_market_momentum/          # S&P 500 vs 125-denní MA       (r=0.832)
│   ├── 02_stock_price_strength/     # NYSE 52-týdenní maxima/minima  (r=0.696)
│   ├── 03_stock_price_breadth/      # McClellan Oscillator           (r=0.816)
│   ├── 04_put_call_ratio/           # CBOE Put/Call ratio            (r=0.641)
│   ├── 05_market_volatility_vix/    # VIX index                      (r=0.644)
│   ├── 06_safe_haven_demand/        # Akcie vs dluhopisy             (r=0.672)
│   └── 07_junk_bond_demand/         # BB–BBB kreditní spread         (r=0.658)
│
├── index/                           # Pipeline sestavení indexu
│   ├── 01_merge_components.py       # Merge komponent + S&P500 TR + CNN
│   ├── 02_compute_weights.py        # Equal (1/7) + OLS váhy
│   ├── 03_build_index.py            # Výpočet FG_Equal a FG_OLS
│   ├── 04_validate_index.py         # Validace vs CNN (r, MAE, RMSE)
│   ├── fg_dataset.csv               # Merged dataset (7 102 řádků)
│   ├── fg_weights.csv               # Equal + OLS váhy
│   └── fg_index_final.csv           # Finální index (1998–2026)
│
├── strategy/                        # Backtesting obchodních strategií (v2)
│   ├── backtester.py                # Engine: 5 strategií + compute_metrics
│   ├── config.py                    # Sdílené konstanty (rozsahy, periody, poplatky)
│   ├── 01_grid_search.py            # Grid search IS (2.36M kombinací, paralelní)
│   ├── 02_out_of_sample.py          # OOS validace (2016–2026)
│   ├── 03_analysis.py               # Grafy + tabulka výsledků
│   ├── grid_results.csv             # Výsledky grid searche (17 975 platných kombinací)
│   ├── oos_results.csv              # IS/OOS výsledky top strategií
│   ├── full_period.png              # Equity křivky 1998–2026 (IS+OOS)
│   ├── oos_equity.png               # Equity křivky OOS 2016–2026
│   └── results_table.png            # Přehledová tabulka strategií
│
├── _archive/strategy_v1/            # Stará verze backtesteru (pro referenci)
│
├── ZDROJE.md                        # Literatura a citace
└── README.md                        # Tento soubor
```

---

## Pipeline — pořadí spuštění

### 1. Index
```bash
python index/01_merge_components.py   # → fg_dataset.csv
python index/02_compute_weights.py    # → fg_weights.csv
python index/03_build_index.py        # → fg_index_final.csv
python index/04_validate_index.py     # → validation_chart.png
```

### 2. Strategie
```bash
python strategy/01_grid_search.py     # → grid_results.csv  (~25 min)
python strategy/02_out_of_sample.py   # → oos_results.csv
python strategy/03_analysis.py        # → full_period.png, oos_equity.png, results_table.png
```

---

## Klíčové výsledky

### Index (validace vs CNN, overlap 2011–2026)
| Varianta | Pearson r | R² | MAE |
|----------|-----------|----|-----|
| FG_Equal (1/7 každá komponenta) | 0.933 | 0.870 | 6.52 |
| FG_OLS (regresní váhy)          | 0.939 | 0.882 | 5.85 |

### Strategie — OOS 2016–2026 (S&P 500 Total Return, poplatky 0.1 %)
| Strategie | Parametry | OOS Return | OOS CAGR | OOS Sharpe | OOS MaxDD |
|-----------|-----------|-----------|----------|-----------|----------|
| Buy & Hold (benchmark) | — | 277 % | 13.6 % | 0.81 | −33.8 % |
| kontrarian_long  FGI_OLS   | entry=38, exit=79 | **267 %** | 13.6 % | **0.84** | −30.5 % |
| kontrarian_combined FGI_OLS | entry=38, exit=79 | 245 % | 12.9 % | 0.75 | −30.5 % |
| trend_long FGI_Equal        | entry=58, exit=8  | 225 % | 12.3 % | **0.91** | −27.1 % |
| trend_long FGI_OLS          | entry=54, exit=11 | 217 % | 12.0 % | 0.90 | −27.3 % |
| kontrarian_long FGI_Equal   | entry=40, exit=82 | 180 % | 10.6 % | 0.69 | −33.8 % |

---

## Backtestovací engine (strategy/backtester.py)

Pět strategií, každá jako samostatná čitelná funkce:

| Strategie | Popis |
|-----------|-------|
| `kontrarian_long`     | Long při strachu (FGI < entry), exit při euforii (FGI > exit) |
| `kontrarian_combined` | Střídá LONG/SHORT bez cash; přepíná na sentimentových extrémech |
| `trend_long`          | Long při euforii (FGI > entry), exit při strachu (FGI < exit) |
| `trend_combined`      | Trendová verze long+short bez cash |
| `ma_combined`         | MA crossover sentimentu; long když fast MA > slow MA, jinak short |

Short pozice modelovány jako inverzní ETF: `equity = invested × (entry_price / current_price)`

---

## Metodologie

| Aspekt | Detail |
|--------|--------|
| Normalizace komponent | Rolling Z-score (okna optimalizována per komponenta) |
| Váhy indexu | Equal (1/7) a OLS regrese na CNN datech 2011–2026 |
| IS/OOS split | IS: 1998–2015, OOS: 2016–2026 |
| Backtesting | Next-day execution, long i short strategie |
| Transakční náklady | 0.1 % per trade |
| Benchmark | S&P 500 Total Return Index (^SP500TR) |
| Paralelizace | joblib n_jobs=−1 (všechna CPU jádra) |
| Grid search | ~2.36M kombinací parametrů na IS datech |
