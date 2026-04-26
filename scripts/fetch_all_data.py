"""
Auto-Fetch ALL Available Market Data

This script automatically downloads:
- VIX (Volatility Index)
- Oil (WTI Crude)
- Bitcoin (BTCUSD)
- EURUSD
- Silver (XAGUSD)
- Gold ETF holdings (GLD)
- Fed data from FRED

All FREE data sources, no API keys needed for basics.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import logging
import os
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_yahoo_data(symbol, start_date, end_date, name):
    """
    Fetch data from Yahoo Finance

    Args:
        symbol: Yahoo Finance symbol (e.g., '^VIX', 'CL=F')
        start_date: Start date string
        end_date: End date string
        name: Name for logging

    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"📥 Fetching {name} ({symbol})...")

    try:
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval='1d',
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if len(df) == 0:
            logger.warning(f"⚠️ No data returned for {name}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # Rename columns to lowercase
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        # Select relevant columns
        date_col = 'date' if 'date' in df.columns else 'datetime'
        df = df[[date_col, 'open', 'high', 'low', 'close', 'volume']]
        df = df.rename(columns={date_col: 'time'})
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_localize(None)

        logger.info(f"   ✅ {name}: {len(df)} bars from {df['time'].iloc[0]} to {df['time'].iloc[-1]}")

        return df

    except Exception as e:
        logger.error(f"   ❌ Error fetching {name}: {e}")
        return None


def fetch_vix(start_date, end_date):
    """Fetch VIX (Volatility Index) - Fear gauge"""
    return fetch_yahoo_data('^VIX', start_date, end_date, 'VIX')


def fetch_oil(start_date, end_date):
    """Fetch WTI Crude Oil prices"""
    return fetch_yahoo_data('CL=F', start_date, end_date, 'WTI Crude Oil')


def fetch_bitcoin(start_date, end_date):
    """Fetch Bitcoin (BTCUSD)"""
    # Bitcoin only liquid since ~2017
    btc_start = max(start_date, '2017-01-01')
    return fetch_yahoo_data('BTC-USD', btc_start, end_date, 'Bitcoin')


def fetch_eurusd(start_date, end_date):
    """Fetch EURUSD forex pair"""
    return fetch_yahoo_data('EURUSD=X', start_date, end_date, 'EURUSD')


def fetch_silver(start_date, end_date):
    """Fetch Silver (XAGUSD)"""
    return fetch_yahoo_data('SI=F', start_date, end_date, 'Silver')


def fetch_gld_holdings(start_date, end_date):
    """Fetch GLD ETF (Gold ETF) price as proxy for institutional positioning"""
    return fetch_yahoo_data('GLD', start_date, end_date, 'GLD Gold ETF')


def fetch_us_dollar_index(start_date, end_date):
    """Fetch US Dollar Index (DXY) - backup if needed"""
    return fetch_yahoo_data('DX-Y.NYB', start_date, end_date, 'US Dollar Index')


def fetch_spx(start_date, end_date):
    """Fetch S&P 500 index"""
    return fetch_yahoo_data('^GSPC', start_date, end_date, 'S&P 500')


def fetch_us10y(start_date, end_date):
    """Fetch 10-year Treasury yield index from Yahoo (^TNX, quoted in percent)."""
    return fetch_yahoo_data('^TNX', start_date, end_date, 'US 10Y Treasury Yield')


def align_to_hourly(df_daily, df_hourly_reference):
    """
    Align daily data to hourly frequency by forward-filling

    Args:
        df_daily: Daily data DataFrame
        df_hourly_reference: Hourly reference DataFrame (for timestamps)

    Returns:
        DataFrame aligned to hourly frequency
    """
    # Convert to datetime
    df_daily['time'] = pd.to_datetime(df_daily['time'])
    df_hourly_reference['time'] = pd.to_datetime(df_hourly_reference['time'])

    # Set date as index
    df_daily = df_daily.set_index('time')

    # Resample to hourly and forward fill
    df_hourly = df_daily.resample('H').ffill()

    # Align to reference timestamps
    df_aligned = df_hourly.reindex(df_hourly_reference['time'], method='ffill')

    return df_aligned.reset_index()


def parse_args():
    parser = argparse.ArgumentParser(description="Download daily macro market data from Yahoo Finance")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="End date YYYY-MM-DD (Yahoo end is exclusive)",
    )
    parser.add_argument("--data-dir", default="data", help="Directory to write CSV files")
    return parser.parse_args()


def main():
    """Main function to fetch all data"""

    # Configuration
    args = parse_args()
    START_DATE = args.start
    END_DATE = args.end
    DATA_DIR = args.data_dir

    os.makedirs(DATA_DIR, exist_ok=True)

    logger.info("="*70)
    logger.info("🚀 FETCHING ALL AVAILABLE MARKET DATA")
    logger.info("="*70)
    logger.info(f"\n📅 Date Range: {START_DATE} to {END_DATE}")
    logger.info(f"📂 Save Directory: {DATA_DIR}/\n")

    # Dictionary to store all data
    datasets = {}

    sources = [
        ("DXY", "dxy_daily.csv", fetch_us_dollar_index),
        ("SPX", "spx_daily.csv", fetch_spx),
        ("US10Y", "us10y_daily.csv", fetch_us10y),
        ("VIX", "vix_daily.csv", fetch_vix),
        ("OIL", "oil_wti_daily.csv", fetch_oil),
        ("BTC", "bitcoin_daily.csv", fetch_bitcoin),
        ("EURUSD", "eurusd_daily.csv", fetch_eurusd),
        ("SILVER", "silver_daily.csv", fetch_silver),
        ("GLD", "gld_etf_daily.csv", fetch_gld_holdings),
    ]

    for name, filename, fetcher in sources:
        df = fetcher(START_DATE, END_DATE)
        if df is not None:
            df.to_csv(f'{DATA_DIR}/{filename}', index=False)
            datasets[name] = df

    # Summary
    logger.info("\n" + "="*70)
    logger.info("📊 DOWNLOAD SUMMARY")
    logger.info("="*70)

    for name, df in datasets.items():
        logger.info(f"✅ {name:15} {len(df):6,} bars")

    logger.info(f"\n✅ Successfully downloaded {len(datasets)}/{len(sources)} datasets")

    logger.info("\nFiles are ready for features.macro_features.load_macro_data().")

    return datasets


if __name__ == "__main__":
    # Check if yfinance is installed
    try:
        import yfinance
    except ImportError:
        print("❌ ERROR: yfinance not installed")
        print("\n📥 Installing yfinance...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'yfinance'])
        print("✅ yfinance installed!")
        import yfinance

    # Run
    datasets = main()

    print("\n🔥 Data fetch complete! Check the data/ directory.")
