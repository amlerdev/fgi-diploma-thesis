# Fear & Greed Index - diplomová práce

Tento repozitář obsahuje finální výstup diplomové práce zaměřené na rekonstrukci CNN Fear & Greed Indexu z veřejně dostupných dat a na backtesting sentimentových obchodních strategií na americkém akciovém trhu.

Praktickým cílem projektu je:

- rekonstruovat index zpět do období `1998-01-02` až `2026-03-20`,
- ověřit kvalitu rekonstrukce vůči historické řadě CNN v překryvu `2011-2026`,
- otestovat, zda lze rekonstruovaný sentiment využít pro systematické řízení expozice na `S&P 500 Total Return (^SP500TR)`.

Finální text diplomové práce je dostupný v souboru `Diplomova_prace_Petr_Amler.pdf`.

**Autor:** Petr Amler  
**Instituce:** VŠB-TU Ostrava, Ekonomická fakulta  
**Typ práce:** diplomová práce

---

## Co projekt dělá

Projekt rekonstruuje sentimentový index inspirovaný metodikou CNN Fear & Greed Indexu, porovnává jej s historickou řadou CNN a následně testuje, zda může sloužit jako vstup pro obchodní strategie na americkém akciovém trhu.

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

Obě varianty dosahují vysoké shody s historickou řadou CNN. `FGI_OLS` se trefuje přesněji, zatímco `FGI_Equal` zachovává jednodušší konstrukci založenou na rovných vahách.

### Out-of-sample výsledky (`2016-01-01` až `2026-03-20`)

| Strategie | Varianta | Parametry | OOS Return | OOS CAGR | OOS Sharpe | OOS MaxDD |
|-----------|----------|-----------|------------|----------|------------|-----------|
| `Buy and Hold` | benchmark | - | 285.4 % | 14.16 % | 0.83 | -33.79 % |
| `kontrarian_long` | `FGI_OLS` | `entry=38, exit=79` | 260.8 % | 13.42 % | 0.84 | -28.43 % |
| `trend_long` | `FGI_OLS` | `entry=58, exit=11` | 226.2 % | 12.30 % | 0.93 | -27.08 % |
| `trend_long` | `FGI_Equal` | `entry=58, exit=8` | 220.1 % | 12.09 % | 0.92 | -27.08 % |

Praktický závěr diplomové práce:

- aktivní strategie nepřekonaly benchmark v absolutním výnosu,
- některé long-only varianty dosáhly lepšího poměru výnosu a rizika,
- `trend_long` a `kontrarian_long` vyšly výrazně lépe než MA strategie,
- `ma_combined` v out-of-sample období selhává na obou variantách indexu.

---

## Datové zdroje

Projekt kombinuje několik veřejných zdrojů:

- archiv historických hodnot CNN Fear & Greed,
- tržní časové řady z `Yahoo Finance`, včetně `^SP500TR`,
- data ze `StockCharts` pro NYSE strength a breadth,
- data z `FRED`, CBOE a dalších veřejných zdrojů pro kreditní spready a put/call řadu.

---

## Obsah repozitáře

- `Diplomova_prace_Petr_Amler.pdf` - finální verze diplomové práce,
- podkladové analytické výstupy, datové soubory a pomocné materiály k praktické části práce.
