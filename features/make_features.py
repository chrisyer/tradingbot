# features/make_features.py
import numpy as np
import pandas as pd
from data.load_data import load_ohlc_csv

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def compute_features(df, normalize=True, norm_stats=None, return_norm_stats=False):
    df = df.copy()
    
    # 1. Gold Features
    df["ret"] = np.log(df["close"]).diff().fillna(0.0)
    df["vol"] = df["ret"].rolling(24).std().fillna(0.0)         
    df["mom"] = df["close"].pct_change(24).fillna(0.0)          
    
    # Moving Averages
    df["ma_fast"] = df["close"].rolling(24).mean()
    df["ma_slow"] = df["close"].rolling(120).mean()
    df["ma_diff"] = ((df["ma_fast"] - df["ma_slow"]) / df["close"]).fillna(0.0)
    
    df["rsi"] = compute_rsi(df["close"], period=14) / 100.0

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_diff"] = (macd - signal) / df["close"]
    df["macd_diff"] = df["macd_diff"].fillna(0.0)

    # 2. MACRO FEATURES (The "God Mode" Inputs)
    macro_cols = {"dxy_close", "spx_close", "us10y_close"}
    if macro_cols.issubset(df.columns):
        # DXY Returns
        df["dxy_ret"] = np.log(df["dxy_close"]).diff().fillna(0.0)
        # SPX Returns
        df["spx_ret"] = np.log(df["spx_close"]).diff().fillna(0.0)
        # US10Y Change
        df["us10y_chg"] = df["us10y_close"].diff().fillna(0.0)
        
        # Correlations (Is Gold moving with or against the Dollar?)
        # 24h rolling correlation
        df["corr_dxy"] = df["ret"].rolling(24).corr(df["dxy_ret"]).fillna(0.0)
        df["corr_spx"] = df["ret"].rolling(24).corr(df["spx_ret"]).fillna(0.0)
        
        feature_cols = [
            "ret", "vol", "mom", "ma_diff", "rsi", "macd_diff",
            "dxy_ret", "spx_ret", "us10y_chg", "corr_dxy", "corr_spx"
        ]
    else:
        # Fallback if macro data missing
        feature_cols = ["ret", "vol", "mom", "ma_diff", "rsi", "macd_diff"]

    # Clean data
    df = df.iloc[120:].reset_index(drop=True)

    feats = df[feature_cols].to_numpy(dtype=np.float32)
    rets = df["ret"].to_numpy(dtype=np.float32)

    # Force cleanup
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    stats = None
    if normalize:
        if norm_stats is None:
            mu = feats.mean(axis=0, keepdims=True)
            sig = feats.std(axis=0, keepdims=True) + 1e-8
        else:
            mu, sig = norm_stats
        feats = (feats - mu) / sig
        stats = (mu, sig)

    if return_norm_stats:
        return df, feats, rets, stats
    return df, feats, rets


def make_features(csv_path: str, window: int = 64, normalize=True):
    df = load_ohlc_csv(csv_path)
    return compute_features(df, normalize=normalize)


def _load_aux_ohlc(path):
    df = load_ohlc_csv(path)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _compute_regime_columns(df, prefix):
    out = pd.DataFrame({"time": df["time"]})
    close = df["close"].astype(float)
    ret = np.log(close).diff().fillna(0.0)

    ma_fast = close.rolling(10).mean()
    ma_slow = close.rolling(50).mean()
    out[f"{prefix}_ma_diff"] = ((ma_fast - ma_slow) / close).fillna(0.0)
    out[f"{prefix}_trend"] = np.where(ma_fast > ma_slow, 1.0, -1.0)
    out[f"{prefix}_mom_5"] = close.pct_change(5).fillna(0.0)
    out[f"{prefix}_mom_20"] = close.pct_change(20).fillna(0.0)
    out[f"{prefix}_vol_20"] = ret.rolling(20).std().fillna(0.0)
    return out


def _merge_asof_features(base_df, feature_df):
    return pd.merge_asof(
        base_df.sort_values("time"),
        feature_df.sort_values("time"),
        on="time",
        direction="backward",
    )


def make_regime_features(csv_path: str, data_dir: str = "data", window: int = 64, normalize=True):
    """Basic H1 features plus lightweight higher-timeframe and macro regime state."""
    base = load_ohlc_csv(csv_path)
    enriched = base.sort_values("time").reset_index(drop=True)

    for tf_name, filename in [("h4", "xauusd_h4.csv"), ("d1", "xauusd_d1.csv")]:
        aux = _load_aux_ohlc(f"{data_dir}/{filename}")
        enriched = _merge_asof_features(enriched, _compute_regime_columns(aux, tf_name))

    for macro_name, filename in [
        ("dxy", "dxy_daily.csv"),
        ("us10y", "us10y_daily.csv"),
        ("vix", "vix_daily.csv"),
    ]:
        macro = _load_aux_ohlc(f"{data_dir}/{filename}")
        enriched = _merge_asof_features(enriched, _compute_regime_columns(macro, macro_name))

    enriched = enriched.fillna(0.0)
    df, feats, rets = compute_features(enriched, normalize=False)

    regime_cols = [
        c
        for c in df.columns
        if c.startswith(("h4_", "d1_", "dxy_", "us10y_", "vix_"))
        and c not in {"dxy_close", "us10y_close"}
    ]
    regime_feats = df[regime_cols].to_numpy(dtype=np.float32)
    regime_feats = np.nan_to_num(regime_feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.concatenate([feats, regime_feats], axis=1).astype(np.float32)

    if normalize:
        mu = feats.mean(axis=0, keepdims=True)
        sig = feats.std(axis=0, keepdims=True) + 1e-8
        feats = (feats - mu) / sig

    return df, feats, rets
