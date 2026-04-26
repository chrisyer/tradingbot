"""
Download OHLC data with incremental cache.

Supported sources:
- local MetaTrader 5 terminal (mt5)
- external Yahoo Finance (yfinance)
- external Dukascopy feed (dukascopy)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


MT5_TIMEFRAME_ATTR = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
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


def to_utc_naive(series: pd.Series, unit: str | None = None) -> pd.Series:
    """Normalize any datetime-like series to UTC, then drop tz info for CSV consistency."""
    return pd.to_datetime(series, errors="coerce", utc=True, unit=unit).dt.tz_convert(None)


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"tick_volume": "tick_volume"})
    df["time"] = to_utc_naive(pd.to_numeric(df["time"], errors="coerce"), unit="s")
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
    df["time"] = to_utc_naive(df["time"])
    df = df.dropna(subset=["time"]).sort_values("time").drop_duplicates("time")
    return df.reset_index(drop=True)


def get_mt5_module():
    try:
        import MetaTrader5 as mt5
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MetaTrader5 package is required only when --source mt5 is used. "
            "Install with: pip install MetaTrader5"
        ) from exc
    return mt5


def fetch_range_mt5(mt5, symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    tf = getattr(mt5, MT5_TIMEFRAME_ATTR[timeframe])
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
    df["time"] = to_utc_naive(df["time"])
    keep = ["time", "open", "high", "low", "close", "tick_volume"]
    return df[[c for c in keep if c in df.columns]].dropna(subset=["time"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _coerce_dukascopy_row(row) -> dict | None:
    if isinstance(row, dict):
        t = row.get("time", row.get("timestamp", row.get("ts", row.get("ctm"))))
        o = row.get("open", row.get("o"))
        h = row.get("high", row.get("h"))
        l = row.get("low", row.get("l"))
        c = row.get("close", row.get("c"))
        v = row.get("volume", row.get("v", row.get("vol", 0)))
    elif isinstance(row, list) and len(row) >= 5:
        t = row[0]
        o, h, l, c = row[1], row[2], row[3], row[4]
        v = row[5] if len(row) > 5 else 0
    else:
        return None

    if t is None or o is None or h is None or l is None or c is None:
        return None
    return {"time": t, "open": o, "high": h, "low": l, "close": c, "tick_volume": v}


def fetch_range_dukascopy(provider_symbol: str, timeframe: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe for dukascopy: {timeframe}")
    interval_min = int(TIMEFRAME_SECONDS[timeframe] // 60)
    instrument = provider_symbol.replace("/", "").upper()
    current_end_ms = int(end_dt.timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)
    rows = []

    # Pull backward chunks until we cover requested start.
    for _ in range(30):
        params = {
            "path": "chart/json3",
            "instrument": instrument,
            "offer_side": "BID",
            "interval": str(interval_min),
            "splits": "true",
            "time_direction": "P",
            "timestamp": str(current_end_ms),
        }
        url = f"https://freeserv.dukascopy.com/2.0/index.php?{urlencode(params)}"
        with urlopen(url, timeout=30) as resp:
            payload = resp.read().decode("utf-8", errors="replace").strip()

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            # Some endpoints may return prefixed text; try to recover array/object.
            left = min([i for i in [payload.find("["), payload.find("{")] if i >= 0], default=-1)
            right = max(payload.rfind("]"), payload.rfind("}"))
            if left < 0 or right < 0:
                chunk = []
            else:
                chunk = json.loads(payload[left : right + 1])

        if not chunk:
            break

        normalized = []
        for raw in chunk:
            one = _coerce_dukascopy_row(raw)
            if one is not None:
                normalized.append(one)
        if not normalized:
            break

        rows.extend(normalized)
        chunk_df = pd.DataFrame(normalized)
        chunk_df["time"] = pd.to_numeric(chunk_df["time"], errors="coerce")
        chunk_df = chunk_df.dropna(subset=["time"])
        if chunk_df.empty:
            break

        oldest_ms = int(chunk_df["time"].min())
        if oldest_ms <= start_ms:
            break
        # move back one bar
        current_end_ms = oldest_ms - (TIMEFRAME_SECONDS[timeframe] * 1000)

    if not rows:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])

    df = pd.DataFrame(rows)
    # Accept epoch in seconds or milliseconds.
    numeric_time = pd.to_numeric(df["time"], errors="coerce")
    unit = "ms" if numeric_time.dropna().max() > 10_000_000_000 else "s"
    df["time"] = to_utc_naive(numeric_time, unit=unit)
    for col in ["open", "high", "low", "close", "tick_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df[(df["time"] >= start_dt.replace(tzinfo=None)) & (df["time"] <= end_dt.replace(tzinfo=None))]
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def incremental_update(
    target: DownloadTarget,
    out_dir: Path,
    cache_dir: Path,
    from_dt: datetime,
    to_dt: datetime,
    full_refresh: bool,
    fallback_days: int,
    source: str,
    mt5=None,
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
        fetched = fetch_range_mt5(mt5, target.symbol, target.timeframe, effective_from, to_dt)
    elif source == "yfinance":
        fetched = fetch_range_yfinance(provider_symbol, target.timeframe, effective_from, to_dt)
    else:
        fetched = fetch_range_dukascopy(provider_symbol, target.timeframe, effective_from, to_dt)
    if fetched.empty and fallback_days > 0:
        fallback_from = to_dt - timedelta(days=fallback_days)
        print(
            f"⚠️ {target.symbol} {target.timeframe}: no bars in requested window "
            f"{effective_from} -> {to_dt}. Trying fallback window ({fallback_days}d)..."
        )
        if source == "mt5":
            fetched = fetch_range_mt5(mt5, target.symbol, target.timeframe, fallback_from, to_dt)
        elif source == "yfinance":
            fetched = fetch_range_yfinance(provider_symbol, target.timeframe, fallback_from, to_dt)
        else:
            fetched = fetch_range_dukascopy(provider_symbol, target.timeframe, fallback_from, to_dt)
    if existing is None:
        merged = fetched
    else:
        merged = pd.concat([existing, fetched], ignore_index=True)
        merged["time"] = to_utc_naive(merged["time"])
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
    parser.add_argument("--source", choices=["mt5", "yfinance", "dukascopy"], default="mt5", help="Data source backend")
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
        if tf not in TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe: {tf}. Allowed: {sorted(TIMEFRAME_SECONDS.keys())}")

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
    mt5 = None
    if args.source == "mt5":
        mt5 = get_mt5_module()
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
                    mt5=mt5,
                )
    finally:
        if mt5_ready:
            mt5.shutdown()


if __name__ == "__main__":
    main()
