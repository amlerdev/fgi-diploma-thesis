"""
StockCharts Historical Data Downloader
=======================================
Generická šablona pro stahování historických dat ze StockCharts.com.

Metodika:
    StockCharts.com poskytuje interaktivní grafy s historickými daty přes
    JavaScript, která se načítají dynamicky z interního API endpointu.
    Tento endpoint byl identifikován analýzou síťového provozu (HAR soubory)
    při prohlížení grafů v nástroji Advanced Charting Platform (ACP).

    Nalezený endpoint:
        https://stockcharts.com/quotebrain/historyandquote/d

    Tento skript endpoint volá přímo a data parsuje do formátu DataFrame.

Dostupné tickery (výběr):
    $NYHGH  - NYSE New 52-Week Highs       (od 1980)
    $NYLOW  - NYSE New 52-Week Lows        (od 1980)
    $NYMO   - NYSE McClellan Oscillator    (od 1998)
    $NYUPV  - NYSE Up Volume               (od 1980)
    $NYDNV  - NYSE Down Volume             (od 1980)
    $CPC    - CBOE Total Put/Call Ratio    (od 1995)

Použití:
    Nastavte konstanty TICKER, DATE_FROM, DATE_TO a OUTPUT_FILE,
    poté skript spusťte.

Autor: Petr Amler (AML0005)
"""

import requests
import pandas as pd
from datetime import datetime

# =============================================================================
# KONFIGURACE - upravte podle potřeby
# =============================================================================

TICKER      = "$NYHGH"          # StockCharts ticker
DATE_FROM   = "19800101"        # Začátek období (YYYYMMDD)
DATE_TO     = "20260320"  # Konec období (fixed)
OUTPUT_FILE = "output_data.csv" # Výstupní soubor

# =============================================================================
# KONSTANTY - neměňte
# =============================================================================

API_URL = "https://stockcharts.com/quotebrain/historyandquote/d"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://stockcharts.com/acp/",
    "Accept":     "application/json",
}

PARAMS = {
    "symbol":     TICKER,
    "start":      DATE_FROM,
    "end":        DATE_TO,
    "windowid":   "main",
    "chartid":    "main",
    "fromde":     "false",
    "numCharts":  "1",
    "numWindows": "1",
    "appv":       "1.91",
    "z":          "true",
    "extended":   "true",
}

# =============================================================================
# STAŽENÍ DAT
# =============================================================================

def download_data(ticker: str, date_from: str, date_to: str) -> dict:
    """
    Stáhne historická data pro zadaný ticker z QuoteBrain API.

    Parametry
    ----------
    ticker : str
        StockCharts ticker, např. '$NYHGH'
    date_from : str
        Počáteční datum ve formátu YYYYMMDD
    date_to : str
        Koncové datum ve formátu YYYYMMDD

    Návratová hodnota
    -----------------
    dict
        Surová JSON odpověď z API
    """
    params = {**PARAMS, "symbol": ticker, "start": date_from, "end": date_to}

    print(f"Stahuji data pro {ticker} ({date_from} – {date_to})...")

    response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()

    return response.json()


def parse_response(data: dict) -> pd.DataFrame:
    """
    Parsuje JSON odpověď do DataFrame.

    Struktura odpovědi:
        data['history']['intervals'] -> seznam denních záznamů
        Každý záznam obsahuje: start.time, open, high, low, close, volume

    Parametry
    ----------
    data : dict
        Surová JSON odpověď z API

    Návratová hodnota
    -----------------
    pd.DataFrame
        DataFrame se sloupci: Date, Open, High, Low, Close, Volume
    """
    intervals = data["history"]["intervals"]

    records = [
        {
            "Date":   entry["start"]["time"].split()[0],
            "Open":   entry["open"],
            "High":   entry["high"],
            "Low":    entry["low"],
            "Close":  entry["close"],
            "Volume": entry["volume"],
        }
        for entry in intervals
    ]

    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df


# =============================================================================
# HLAVNÍ PROGRAM
# =============================================================================

if __name__ == "__main__":

    # Stažení
    raw = download_data(TICKER, DATE_FROM, DATE_TO)

    # Parsování
    df = parse_response(raw)

    # Základní přehled
    print(f"\nStaženo {len(df)} obchodních dní")
    print(f"Období: {df['Date'].min().date()} – {df['Date'].max().date()}")
    print(f"\nPřehled (Close):")
    print(f"  Průměr:  {df['Close'].mean():.2f}")
    print(f"  Medián:  {df['Close'].median():.2f}")
    print(f"  Min:     {df['Close'].min():.2f}  ({df.loc[df['Close'].idxmin(), 'Date'].date()})")
    print(f"  Max:     {df['Close'].max():.2f}  ({df.loc[df['Close'].idxmax(), 'Date'].date()})")

    print(f"\nPrvní řádky:")
    print(df.head())

    # Uložení
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nData uložena: {OUTPUT_FILE}")
