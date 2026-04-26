"""
Download OHLC data with incremental cache.

Supported sources:
- local MetaTrader 5 terminal (mt5)
- external Yahoo Finance (yfinance)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

TIMEFRAME_SECONDS = {
    "M1": 60,
    "M5": 5 * 60,
    "M15": 15 * 60,
    "M30": 30 * 60,
    "H1": 60 * 60,
    "H4": 4 * 60 * 60,
    "D1": 24 * 60 * 60,
}


@dataclass
class DownloadTarget:
    symbol: str
    timeframe: str
    provider_symbol: str | None = None

    @property
    def output_filename(self) -> str:
        return f"{self.symbol.lower()}_{self.timeframe.lower()}.csv"

    @property
    def cache_filename(self) -> str:
        return f"{self.symbol.lower()}_{self.timeframe.lower()}.json"


def parse_dt(value: str) -> datetime:
    if value.lower() == "now":
        return datetime.now(timezone.utc)
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"tick_volume": "tick_volume"})
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
    cols = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
    keep = [c for c in cols if c in df.columns]
    out = df[keep].copy()
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return out


def read_existing(csv_path: Path) -> pd.DataFrame | None:
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    return df.reset_index(drop=True)


def fetch_range(symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    tf = TIMEFRAME_MAP[timeframe]
    rates = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)
    if rates is None:
        raise RuntimeError(f"MT5 copy_rates_range failed for {symbol} {timeframe}: {mt5.last_error()}")
    if len(rates) == 0:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
    return normalize_bars(pd.DataFrame(rates))


def fetch_range_yfinance(provider_symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    import yfinance as yf

    interval_map = {
        "M1": "1m",
        "M5": "5m",
        "M15": "15m",
        "M30": "30m",
        "H1": "60m",
        "D1": "1d",
    }
    interval = interval_map.get(timeframe)
    if interval is None:
        raise ValueError(f"Timeframe {timeframe} is not supported for yfinance source")

    df = yf.download(
        provider_symbol,
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "tick_volume",
        }
    )
    df = df.reset_index().rename(columns={"Datetime": "time", "Date": "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    keep = ["time", "open", "high", "low", "close", "tick_volume"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["time"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)


def incremental_update(
    target: DownloadTarget,
    out_dir: Path,
    cache_dir: Path,
    from_dt: datetime,
    to_dt: datetime,
    full_refresh: bool,
    fallback_days: int,
    source: str,
) -> None:
    csv_path = out_dir / target.output_filename
    cache_path = cache_dir / target.cache_filename

    existing = None if full_refresh else read_existing(csv_path)
    effective_from = from_dt

    if existing is not None and len(existing) > 0:
        last_time = existing["time"].iloc[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        # +1 bar to avoid duplicated tail fetch
        effective_from = max(from_dt, last_time + timedelta(seconds=TIMEFRAME_SECONDS[target.timeframe]))

    provider_symbol = target.provider_symbol or target.symbol
    if source == "mt5":
        fetched = fetch_range(target.symbol, target.timeframe, effective_from, to_dt)
    else:
        fetched = fetch_range_yfinance(provider_symbol, target.timeframe, effective_from, to_dt)
    if fetched.empty and fallback_days > 0:
        fallback_from = to_dt - timedelta(days=fallback_days)
        print(
            f"⚠️ {target.symbol} {target.timeframe}: no bars in requested window "
            f"{effective_from} -> {to_dt}. Trying fallback window ({fallback_days}d)..."
        )
        if source == "mt5":
            fetched = fetch_range(target.symbol, target.timeframe, fallback_from, to_dt)
        else:
            fetched = fetch_range_yfinance(provider_symbol, target.timeframe, fallback_from, to_dt)
    if existing is None:
        merged = fetched
    else:
        merged = pd.concat([existing, fetched], ignore_index=True)
        merged["time"] = pd.to_datetime(merged["time"], errors="coerce")
        merged = merged.dropna(subset=["time"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)

    meta = {
        "symbol": target.symbol,
        "timeframe": target.timeframe,
        "rows": int(len(merged)),
        "from_utc": str(merged["time"].iloc[0]) if len(merged) else None,
        "to_utc": str(merged["time"].iloc[-1]) if len(merged) else None,
        "last_sync_utc": datetime.now(timezone.utc).isoformat(),
        "full_refresh": full_refresh,
    }
    cache_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if len(merged) == 0:
        print(f"⚠️ {target.symbol} {target.timeframe}: broker returned 0 bars. Keep terminal open and ensure symbol history is loaded.")
        return

    earliest = pd.to_datetime(merged["time"].iloc[0]).to_pydatetime().replace(tzinfo=timezone.utc)
    if from_dt < earliest:
        print(
            f"⚠️ {target.symbol} {target.timeframe}: broker history starts at {earliest.isoformat()} "
            f"(requested {from_dt.isoformat()}). Using available history only."
        )

    print(f"✅ {target.symbol} {target.timeframe}: {len(merged):,} rows -> {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Download market bars from local MT5 terminal with cache")
    parser.add_argument("--symbol", action="append", default=["XAUUSD"], help="Repeatable symbol, e.g. --symbol XAUUSD --symbol EURUSD")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Optional source mapping in form SYMBOL:PROVIDER_SYMBOL, e.g. XAUUSD:GC=F",
    )
    parser.add_argument("--timeframe", action="append", default=["M5", "M15"], help="Repeatable timeframe: M1/M5/M15/M30/H1/H4/D1")
    parser.add_argument("--from", dest="from_dt", default="2015-01-01T00:00:00Z", help="UTC start datetime, e.g. 2018-01-01T00:00:00Z")
    parser.add_argument("--to", dest="to_dt", default="now", help="UTC end datetime or 'now'")
    parser.add_argument("--out-dir", default="data", help="Output CSV directory")
    parser.add_argument("--cache-dir", default="data/.cache/mt5_download", help="Metadata cache directory")
    parser.add_argument("--full-refresh", action="store_true", help="Ignore local CSV cache and rebuild from --from")
    parser.add_argument("--source", choices=["mt5", "yfinance"], default="mt5", help="Data source backend")
    parser.add_argument(
        "--fallback-days",
        type=int,
        default=365,
        help="If requested range returns empty, retry with last N days (0 disables fallback)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = [s.upper() for s in args.symbol]
    timeframes = [t.upper() for t in args.timeframe]

    for tf in timeframes:
        if tf not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {tf}. Allowed: {sorted(TIMEFRAME_MAP.keys())}")

    from_dt = parse_dt(args.from_dt)
    to_dt = parse_dt(args.to_dt)
    if from_dt >= to_dt:
        raise ValueError("--from must be earlier than --to")

    provider_map = {}
    for item in args.map:
        if ":" not in item:
            raise ValueError(f"Invalid --map '{item}', expected SYMBOL:PROVIDER_SYMBOL")
        local_symbol, provider_symbol = item.split(":", 1)
        provider_map[local_symbol.strip().upper()] = provider_symbol.strip()

    default_provider_map = {
        "XAUUSD": "GC=F",
        "EURUSD": "EURUSD=X",
        "BTCUSD": "BTC-USD",
        "SPX": "^GSPC",
    }

    mt5_ready = False
    if args.source == "mt5":
        if not mt5.initialize():
            raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")
        mt5_ready = True

    try:
        for symbol in symbols:
            if args.source == "mt5" and not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"Cannot select symbol in MT5 Market Watch: {symbol}")
            provider_symbol = provider_map.get(symbol, default_provider_map.get(symbol, symbol))
            for tf in timeframes:
                incremental_update(
                    DownloadTarget(symbol=symbol, timeframe=tf, provider_symbol=provider_symbol),
                    out_dir=Path(args.out_dir),
                    cache_dir=Path(args.cache_dir),
                    from_dt=from_dt,
                    to_dt=to_dt,
                    full_refresh=args.full_refresh,
                    fallback_days=args.fallback_days,
                    source=args.source,
                )
    finally:
        if mt5_ready:
            mt5.shutdown()


if __name__ == "__main__":
    main()
