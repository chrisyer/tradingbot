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
import lzma
import struct
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
from tqdm.auto import tqdm


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

DUKASCOPY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "*/*",
}

DUKASCOPY_EMPTY_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]
DUKASCOPY_DAY_CACHE: dict[tuple[str, str], bytes | None] = {}


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


def _empty_dukascopy_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DUKASCOPY_EMPTY_COLUMNS)


def _dukascopy_instrument(provider_symbol: str) -> str:
    return provider_symbol.replace("/", "").upper()


def _dukascopy_price_scale(instrument: str) -> int:
    # Dukascopy stores most FX pairs in 1/100000 units, while JPY crosses and
    # metals use 1/1000 units in the candle feed.
    if instrument.endswith("JPY") or instrument.startswith(("XAU", "XAG")):
        return 1_000
    return 100_000


def _dukascopy_day_url(instrument: str, day: datetime) -> str:
    return (
        f"https://datafeed.dukascopy.com/datafeed/{instrument}/"
        f"{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/BID_candles_min_1.bi5"
    )


def _download_dukascopy_day(instrument: str, day: datetime) -> bytes | None:
    cache_key = (instrument, day.strftime("%Y-%m-%d"))
    if cache_key in DUKASCOPY_DAY_CACHE:
        return DUKASCOPY_DAY_CACHE[cache_key]

    url = _dukascopy_day_url(instrument, day)
    request = Request(url, headers=DUKASCOPY_HEADERS)
    try:
        with urlopen(request, timeout=30) as resp:
            raw = resp.read()
            DUKASCOPY_DAY_CACHE[cache_key] = raw
            return raw
    except HTTPError as exc:
        if exc.code == 404:
            DUKASCOPY_DAY_CACHE[cache_key] = None
            return None
        raise RuntimeError(f"Dukascopy rejected {instrument} request ({exc.code}) for {day.date()}: {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Dukascopy download failed for {instrument} {day.date()}: {exc}") from exc


def _parse_dukascopy_minute_bi5(raw: bytes, instrument: str, day: datetime) -> list[dict]:
    if not raw:
        return []

    try:
        payload = lzma.decompress(raw)
    except lzma.LZMAError as exc:
        raise RuntimeError(f"Could not decompress Dukascopy candle file for {instrument} {day.date()}") from exc

    record_size = struct.calcsize(">Iiiiif")
    if len(payload) % record_size != 0:
        raise RuntimeError(
            f"Unexpected Dukascopy candle record size for {instrument} {day.date()}: "
            f"{len(payload)} bytes"
        )

    day_start = datetime.combine(day.date(), time.min, tzinfo=timezone.utc)
    price_scale = _dukascopy_price_scale(instrument)
    rows = []
    for offset_seconds, open_raw, high_raw, low_raw, close_raw, volume in struct.iter_unpack(">Iiiiif", payload):
        rows.append(
            {
                "time": (day_start + timedelta(seconds=int(offset_seconds))).replace(tzinfo=None),
                "open": open_raw / price_scale,
                "high": high_raw / price_scale,
                "low": low_raw / price_scale,
                "close": close_raw / price_scale,
                "tick_volume": volume,
            }
        )
    return rows


def _iter_utc_days(start_dt: datetime, end_dt: datetime):
    current = datetime.combine(start_dt.date(), time.min, tzinfo=timezone.utc)
    last = datetime.combine(end_dt.date(), time.min, tzinfo=timezone.utc)
    while current <= last:
        yield current
        current += timedelta(days=1)


def _utc_day_count(start_dt: datetime, end_dt: datetime) -> int:
    return (end_dt.date() - start_dt.date()).days + 1


def fetch_range_dukascopy(
    provider_symbol: str,
    timeframe: str,
    start_dt: datetime,
    end_dt: datetime,
    show_progress: bool = True,
    progress_label: str | None = None,
    max_workers: int = 8,
) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_SECONDS:
        raise ValueError(f"Unsupported timeframe for dukascopy: {timeframe}")
    if timeframe == "D1":
        resample_rule = "1D"
    else:
        interval_min = int(TIMEFRAME_SECONDS[timeframe] // 60)
        resample_rule = f"{interval_min}min"

    instrument = _dukascopy_instrument(provider_symbol)
    rows = []
    days = list(_iter_utc_days(start_dt, end_dt))
    progress = None
    if show_progress:
        label = progress_label or f"{instrument} {timeframe}"
        progress = tqdm(
            total=len(days),
            desc=f"Dukascopy {label}",
            unit="day",
            dynamic_ncols=True,
            leave=False,
            disable=not sys.stderr.isatty(),
        )

    try:
        if max_workers <= 1 or len(days) <= 1:
            for day in days:
                raw = _download_dukascopy_day(instrument, day)
                if raw is not None:
                    rows.extend(_parse_dukascopy_minute_bi5(raw, instrument, day))
                if progress is not None:
                    progress.update(1)
        else:
            workers = min(max_workers, len(days))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_download_dukascopy_day, instrument, day): day for day in days}
                for future in as_completed(futures):
                    day = futures[future]
                    raw = future.result()
                    if raw is not None:
                        rows.extend(_parse_dukascopy_minute_bi5(raw, instrument, day))
                    if progress is not None:
                        progress.update(1)
    finally:
        if progress is not None:
            progress.close()

    if not rows:
        return _empty_dukascopy_frame()

    df = pd.DataFrame(rows)
    df["time"] = to_utc_naive(df["time"])
    df = df.dropna(subset=["time", "open", "high", "low", "close"])
    df = df[(df["time"] >= start_dt.replace(tzinfo=None)) & (df["time"] <= end_dt.replace(tzinfo=None))]
    if df.empty:
        return _empty_dukascopy_frame()

    if timeframe != "M1":
        df = (
            df.set_index("time")
            .resample(resample_rule, label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )

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
    show_progress: bool,
    dukascopy_workers: int,
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
        fetched = fetch_range_dukascopy(
            provider_symbol,
            target.timeframe,
            effective_from,
            to_dt,
            show_progress=show_progress,
            progress_label=f"{target.symbol} {target.timeframe}",
            max_workers=dukascopy_workers,
        )
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
            fetched = fetch_range_dukascopy(
                provider_symbol,
                target.timeframe,
                fallback_from,
                to_dt,
                show_progress=show_progress,
                progress_label=f"{target.symbol} {target.timeframe} fallback",
                max_workers=dukascopy_workers,
            )
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
    parser.add_argument("--symbol", action="append", default=None, help="Repeatable symbol, e.g. --symbol XAUUSD --symbol EURUSD")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Optional source mapping in form SYMBOL:PROVIDER_SYMBOL, e.g. XAUUSD:GC=F",
    )
    parser.add_argument("--timeframe", action="append", default=None, help="Repeatable timeframe: M1/M5/M15/M30/H1/H4/D1")
    parser.add_argument("--from", dest="from_dt", default="2015-01-01T00:00:00Z", help="UTC start datetime, e.g. 2018-01-01T00:00:00Z")
    parser.add_argument("--to", dest="to_dt", default="now", help="UTC end datetime or 'now'")
    parser.add_argument("--out-dir", default="data", help="Output CSV directory")
    parser.add_argument("--cache-dir", default="data/.cache/mt5_download", help="Metadata cache directory")
    parser.add_argument("--full-refresh", action="store_true", help="Ignore local CSV cache and rebuild from --from")
    parser.add_argument("--source", choices=["mt5", "yfinance", "dukascopy"], default="mt5", help="Data source backend")
    parser.add_argument("--no-progress", action="store_true", help="Disable Dukascopy download progress bars")
    parser.add_argument(
        "--dukascopy-workers",
        type=int,
        default=8,
        help="Concurrent Dukascopy day downloads (default: 8, use 1 for serial)",
    )
    parser.add_argument(
        "--fallback-days",
        type=int,
        default=365,
        help="If requested range returns empty, retry with last N days (0 disables fallback)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    symbols = [s.upper() for s in (args.symbol or ["XAUUSD"])]
    timeframes = [t.upper() for t in (args.timeframe or ["M5", "M15"])]

    for tf in timeframes:
        if tf not in TIMEFRAME_SECONDS:
            raise ValueError(f"Unsupported timeframe: {tf}. Allowed: {sorted(TIMEFRAME_SECONDS.keys())}")

    from_dt = parse_dt(args.from_dt)
    to_dt = parse_dt(args.to_dt)
    if from_dt >= to_dt:
        raise ValueError("--from must be earlier than --to")
    if args.dukascopy_workers < 1:
        raise ValueError("--dukascopy-workers must be >= 1")

    provider_map = {}
    for item in args.map:
        if ":" not in item:
            raise ValueError(f"Invalid --map '{item}', expected SYMBOL:PROVIDER_SYMBOL")
        local_symbol, provider_symbol = item.split(":", 1)
        provider_map[local_symbol.strip().upper()] = provider_symbol.strip()

    default_yfinance_map = {
        "XAUUSD": "GC=F",
        "EURUSD": "EURUSD=X",
        "BTCUSD": "BTC-USD",
        "SPX": "^GSPC",
    }
    default_dukascopy_map = {
        "XAUUSD": "XAUUSD",
        "EURUSD": "EURUSD",
        "BTCUSD": "BTCUSD",
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
            source_default_map = default_yfinance_map if args.source == "yfinance" else default_dukascopy_map
            provider_symbol = provider_map.get(symbol, source_default_map.get(symbol, symbol))
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
                    show_progress=not args.no_progress,
                    dukascopy_workers=args.dukascopy_workers,
                    mt5=mt5,
                )
    finally:
        if mt5_ready:
            mt5.shutdown()


if __name__ == "__main__":
    main()
