from pathlib import Path

import pandas as pd
import yfinance as yf


START_DATE = '1990-01-01'
END_DATE = '2026-03-20'


def _save_canonical_vix_csv(vix: pd.Series, csv_path: Path) -> None:
    out = vix.rename('Close').to_frame()
    out.index.name = 'Date'
    out.to_csv(csv_path)


def _download_vix_series(csv_path: Path) -> pd.Series:
    raw = yf.download('^VIX', start=START_DATE, end=END_DATE, progress=False)
    if raw is None or raw.empty:
        raise ValueError("Yahoo Finance nevrátil žádná VIX data.")

    close = raw['Close']
    vix = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
    vix.index = pd.to_datetime(vix.index).normalize()
    vix = vix.astype(float).dropna().sort_index()
    _save_canonical_vix_csv(vix, csv_path)
    return vix


def load_vix_series(csv_path: Path, refresh: bool = False) -> pd.Series:
    if refresh:
        return _download_vix_series(csv_path)

    if not csv_path.exists():
        return _download_vix_series(csv_path)

    try:
        vix_df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        if 'Close' not in vix_df.columns:
            raise ValueError("Chybí sloupec Close")
        vix = vix_df['Close'].astype(float).dropna().sort_index()
        vix.index = pd.to_datetime(vix.index).normalize()
        return vix
    except Exception:
        # Přechodný fallback pro starší CSV formát z yfinance s víceřádkovou hlavičkou.
        legacy = pd.read_csv(csv_path, skiprows=2)
        legacy.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        legacy['Date'] = pd.to_datetime(legacy['Date'])
        vix = legacy.set_index('Date')['Close'].astype(float).dropna().sort_index()
        vix.index = pd.to_datetime(vix.index).normalize()
        _save_canonical_vix_csv(vix, csv_path)
        return vix
