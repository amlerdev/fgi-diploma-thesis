# Fear & Greed Index - diplomová práce

Tento repozitář obsahuje finální implementaci k diplomové práci zaměřené na rekonstrukci CNN Fear & Greed Indexu z veřejně dostupných dat a na backtesting sentimentových obchodních strategií na americkém akciovém trhu.

Praktickým cílem projektu je:

- rekonstruovat index zpět do období `1998-01-02` až `2026-03-20`,
- ověřit kvalitu rekonstrukce vůči historické řadě CNN v překryvu `2011-2026`,
- otestovat, zda lze rekonstruovaný sentiment využít pro systematické řízení expozice na `S&P 500 Total Return (^SP500TR)`.

Repozitář odpovídá finální podobě práce v souboru `DIPLOMKA_OFFICIAL.pdf` a aktuálním skriptům ve složce `code/`.

**Autor:** Petr Amler  
**Instituce:** VŠB-TU Ostrava, Ekonomická fakulta  
**Typ práce:** diplomová práce

---

## Co projekt dělá

Projekt je rozdělen do čtyř logických vrstev:

1. `data/` zajišťuje referenční historickou řadu CNN Fear & Greed Indexu.
2. `components/` implementuje sedm dílčích komponent indexu a jejich normalizaci.
3. `index/` skládá finální kompozitní index `FGI_Equal` a `FGI_OLS` a validuje je vůči CNN.
4. `strategy/` provádí in-sample optimalizaci, out-of-sample validaci a finální vizualizace výsledků.

Výsledná datová řada má `7097` obchodních dní a pokrývá dot-com období, krizi `2008-2009`, covidový propad i nejnovější roky do `2026-03-20`.

---

## Metodika v kostce

- Index je rekonstruován ze 7 veřejných proxy komponent odpovídajících metodice CNN.
- Každá komponenta je normalizována pomocí `rolling Z-score` s komponentově optimalizovaným oknem.
- Finální index existuje ve dvou variantách:
  - `FGI_Equal`: prostý průměr 7 komponent, tedy metodika blízká původnímu CNN indexu.
  - `FGI_OLS`: váhy odhadnuté regresí `OLS` vůči archivované řadě CNN v období `2011-2026`.
- Backtesting používá jednotný `next-day execution` model, počáteční kapitál `10 000 USD` a transakční náklady `0.1 %` za změnu pozice.
- Výzkumný design používá split:
  - `in-sample`: `1998-01-01` až `2015-12-31`
  - `out-of-sample`: `2016-01-01` až `2026-03-20`
- Benchmarkem je `Buy and Hold` na `S&P 500 Total Return`.

Testováno je 6 strategií:

- `kontrarian_long`
- `kontrarian_combined`
- `trend_long`
- `trend_combined`
- `ma_long`
- `ma_combined`

Po in-sample grid searchi zůstalo `16 589` validních konfigurací. Do out-of-sample validace postupuje `36` konfigurací, vždy 3 nejlepší pro každou dvojici `strategie x varianta indexu`.

---

## Hlavní výsledky

### Validace rekonstrukce proti CNN (`2011-2026`)

| Varianta | Počet dní | Pearson r | MAE |
|----------|-----------|-----------|-----|
| `FGI_Equal` | 3826 | 0.9251 | 6.79 |
| `FGI_OLS` | 3826 | 0.9396 | 5.75 |

Obě varianty dosahují vysoké shody s historickou řadou CNN. `FGI_OLS` se trefuje přesněji, ale je potřeba ji chápat jako kalibrovanou rekonstrukci, ne jako plně nezávisle validovanou variantu.

### Out-of-sample výsledky (`2016-01-01` až `2026-03-20`)

| Strategie | Varianta | Parametry | OOS Return | OOS CAGR | OOS Sharpe | OOS MaxDD |
|-----------|----------|-----------|------------|----------|------------|-----------|
| `Buy and Hold` | benchmark | - | 285.4 % | 14.16 % | 0.83 | -33.79 % |
| `kontrarian_long` | `FGI_OLS` | `entry=38, exit=79` | 260.8 % | 13.42 % | 0.84 | -28.43 % |
| `trend_long` | `FGI_OLS` | `entry=58, exit=11` | 226.2 % | 12.30 % | 0.93 | -27.08 % |
| `trend_long` | `FGI_Equal` | `entry=58, exit=8` | 220.1 % | 12.09 % | 0.92 | -27.08 % |

Praktický závěr diplomové práce je podobný i v aktuálních artefaktech:

- aktivní strategie nepřekonaly benchmark v absolutním výnosu,
- některé long-only varianty dosáhly lepšího poměru výnosu a rizika,
- `trend_long` a `kontrarian_long` vyšly výrazně lépe než MA strategie,
- `ma_combined` v out-of-sample období selhává na obou variantách indexu.

---

## Datové zdroje

Projekt kombinuje několik veřejných zdrojů:

- archiv historických hodnot CNN Fear & Greed přes `finhacker.cz` a skript `data/fear_and_greed_dwn.py`,
- tržní časové řady z `Yahoo Finance`, včetně `^SP500TR`,
- data z `StockCharts` pro NYSE strength a breadth,
- data z `FRED`, CBOE a dalších veřejných zdrojů pro kreditní spready a put/call řadu.

Pomocný downloader pro StockCharts je ve `components/stockcharts_downloader.py`.

---

## Struktura projektu

```text
code/
├── data/
│   ├── fear_and_greed_dwn.py
│   └── fear_greed_historical.csv
├── components/
│   ├── 01_market_momentum/
│   ├── 02_stock_price_strength/
│   ├── 03_stock_price_breadth/
│   ├── 04_put_call_ratio/
│   ├── 05_market_volatility_vix/
│   ├── 06_safe_haven_demand/
│   ├── 07_junk_bond_demand/
│   └── stockcharts_downloader.py
├── index/
│   ├── 01_merge_components.py
│   ├── 02_compute_weights.py
│   ├── 03_build_index.py
│   ├── 04_validate_index.py
│   ├── fgi_dataset.csv
│   ├── fgi_weights.csv
│   ├── fgi_index_final.csv
│   └── validation_chart.png
├── strategy/
│   ├── backtester.py
│   ├── config.py
│   ├── 01_grid_search.py
│   ├── 02_out_of_sample.py
│   ├── 03_analysis.py
│   ├── grid_results.csv
│   ├── oos_results.csv
│   ├── results_table.png
│   ├── oos_equity_equal.png
│   ├── oos_equity_ols.png
│   └── full_period.png
└── README.md
```

---

## Reprodukce pipeline

### 1. Referenční CNN řada

```bash
python data/fear_and_greed_dwn.py
```

### 2. Výpočet sedmi komponent

```bash
python components/01_market_momentum/market_momentum.py
python components/02_stock_price_strength/stock_price_strength.py
python components/03_stock_price_breadth/stock_price_breadth.py
python components/04_put_call_ratio/put_call_ratio.py
python components/05_market_volatility_vix/market_volatility_vix.py
python components/06_safe_haven_demand/safe_haven_demand.py
python components/07_junk_bond_demand/junk_bond_demand.py
```

Volitelně lze znovu spouštět i komponentové grid searche přes odpovídající `*_grid.py` skripty v jednotlivých složkách `components/`.

### 3. Konstrukce indexu

```bash
python index/01_merge_components.py
python index/02_compute_weights.py
python index/03_build_index.py
python index/04_validate_index.py
```

### 4. Backtesting a analýza strategií

```bash
python strategy/01_grid_search.py
python strategy/02_out_of_sample.py
python strategy/03_analysis.py
```

Finální analytické artefakty jsou:

- `index/validation_chart.png`
- `strategy/results_table.png`
- `strategy/oos_equity_equal.png`
- `strategy/oos_equity_ols.png`
- `strategy/full_period.png`

---

## Metodická upozornění

README záměrně přiznává několik důležitých omezení, která jsou rozebrána i v diplomové práci:

- `FGI_OLS` používá váhy fitované na překryvu s CNN v letech `2011-2026`, takže je vhodné ji interpretovat primárně jako kalibrovanou rekonstrukci.
- Short expozice je v backtesteru modelována jako zjednodušená syntetická `-1x` denní návratnost bez borrow a financing cost.
- Při konstrukci finálního indexu jsou sporadické chybějící hodnoty některých komponent technicky doplňovány přes `forward fill`.
- Výběr top konfigurací pro out-of-sample validaci je založen na ukazateli `total_return`, nikoli na risk-adjusted kritériu.

Z metodického hlediska je proto nejčistší ex-ante interpretace obvykle spojena s variantou `FGI_Equal`, zatímco `FGI_OLS` je vhodná hlavně pro srovnání s referenční řadou CNN.
