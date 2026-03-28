# Fear & Greed Index — Diplomová práce

Rekonstrukce CNN Fear & Greed Indexu ze 7 veřejných datových zdrojů (1998–2026)
a backtesting algoritmických obchodních strategií na S&P 500 Total Return.

**Autor:** Petr Amler (AML0005)
**Instituce:** VŠB-TU Ostrava, Ekonomická fakulta, Financí
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
├── strategy/                        # Backtesting obchodních strategií
│   ├── 01_grid_search.py            # Unified grid search (2.36M kombinací)
│   ├── 02_out_of_sample.py          # OOS validace (2016–2026)
│   ├── 03_analysis.py               # Grafy + tabulka výsledků
│   ├── grid_results.csv             # Výsledky grid searche
│   ├── oos_results.csv              # IS/OOS výsledky
│   ├── analysis_chart.png           # Equity křivky, drawdown, krize
│   └── results_table.png            # Přehledová tabulka strategií
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
python strategy/03_analysis.py        # → analysis_chart.png, results_table.png
```

---

## Klíčové výsledky

### Index (validace vs CNN, overlap 2011–2026)
| Varianta | Pearson r | R² | MAE |
|----------|-----------|----|-----|
| FG_Equal (1/7 každá komponenta) | 0.927 | 0.859 | ~6.5 |
| FG_OLS (regresní váhy) | 0.936 | 0.876 | ~6.0 |

### Strategie — OOS 2016–2026 (S&P 500 Total Return, poplatky 0.1%)
| Strategie | OOS Return | OOS Sharpe | OOS MaxDD |
|-----------|-----------|-----------|----------|
| Buy & Hold (benchmark) | 277% | 0.81 | -33.8% |
| MR Long+MA FG_Equal | 229% | 0.75 | -33.8% |
| MR Long FG_OLS | 213% | 0.74 | -33.8% |
| Mom Long FG_Equal | 147% | 0.80 | -31.2% |
| Mom Long FG_OLS | 119% | 0.82 | -18.8% |

---

## Metodologie

| Aspekt | Detail |
|--------|--------|
| Normalizace komponent | Rolling Z-score (okna optimalizována per komponenta) |
| Váhy indexu | Equal (1/7) a OLS regrese na CNN datech 2011–2026 |
| IS/OOS split | IS: 1998–2015, OOS: 2016–2026 |
| Backtesting | Next-day execution, long-only i short strategie |
| Transakční náklady | 0.1% per trade |
| Benchmark | S&P 500 Total Return Index (^SP500TR) |
| Paralelizace | joblib n_jobs=-1 (všechna CPU jádra) |
