#!/usr/bin/env python3
"""Generate the latest PPO XAUUSD trading signal."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.make_features import make_features, make_regime_features
from features.ultimate_150_features import make_ultimate_features


def model_stems(model_path: str) -> list[str]:
    stem = model_path[:-4] if model_path.endswith(".zip") else model_path
    stems = [stem]
    if stem.endswith("_latest"):
        stems.append(stem[: -len("_latest")])
    elif re.search(r"_\d+k$", stem):
        stems.append(stem.rsplit("_", 1)[0])
    return list(dict.fromkeys(stems))


def first_existing(stems: list[str], suffix: str) -> str:
    for stem in stems:
        path = f"{stem}{suffix}"
        if os.path.exists(path):
            return path
    return f"{stems[-1]}{suffix}"


def load_metadata(model_path: str) -> tuple[dict, str, str]:
    stems = model_stems(model_path)
    meta_path = first_existing(stems, "_meta.json")
    norm_path = first_existing(stems, "_norm.npz")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return meta, meta_path, norm_path


def load_features(args, meta):
    feature_set = meta.get("feature_set", args.feature_set)
    data_path = meta.get("data", args.data)
    base_tf = meta.get("base_tf", args.base_tf)
    window = int(meta.get("window", args.window))

    if feature_set == "ultimate":
        x_raw, returns, timestamps = make_ultimate_features(base_timeframe=base_tf)
        feature_df = pd.DataFrame({"time": pd.to_datetime(timestamps)})
    elif feature_set == "regime":
        feature_df, x_raw, returns = make_regime_features(data_path, window=window, normalize=False)
    else:
        feature_df, x_raw, returns = make_features(data_path, window=window, normalize=False)

    feature_df = feature_df.copy()
    feature_df["time"] = pd.to_datetime(feature_df["time"])
    return feature_df, x_raw, returns, feature_set, data_path, base_tf, window


def normalize_features(x_raw, norm_path: str, timestamps, train_end: str, window: int):
    if os.path.exists(norm_path):
        stats = np.load(norm_path)
        mu = stats["mu"]
        sig = stats["sig"]
    else:
        train_end_idx = np.searchsorted(pd.to_datetime(timestamps).to_numpy(), np.datetime64(train_end))
        if train_end_idx <= window:
            raise ValueError(f"Not enough data before train_end={train_end}")
        mu = x_raw[:train_end_idx].mean(axis=0, keepdims=True)
        sig = x_raw[:train_end_idx].std(axis=0, keepdims=True) + 1e-8
    return (x_raw - mu) / sig


def dxy_mom20_q75_filter(feature_df: pd.DataFrame, train_end: str) -> tuple[bool, float, float]:
    if "dxy_mom_20" not in feature_df.columns:
        raise ValueError("dxy_mom20_q75_train requires regime features with dxy_mom_20")
    train_mask = feature_df["time"] < pd.Timestamp(train_end)
    threshold = float(feature_df.loc[train_mask, "dxy_mom_20"].quantile(0.75))
    latest_value = float(feature_df["dxy_mom_20"].iloc[-1])
    return latest_value > threshold, latest_value, threshold


def action_to_position(action: int, position_mode: str) -> int:
    if position_mode == "long_short":
        return {0: -1, 1: 0, 2: 1}.get(action, 0)
    return 1 if action == 1 else 0


def position_label(position: int) -> str:
    if position > 0:
        return "LONG"
    if position < 0:
        return "SHORT"
    return "FLAT"


def main():
    parser = argparse.ArgumentParser(description="Generate latest PPO XAUUSD signal")
    parser.add_argument("--model", default="train/ppo_xauusd_aggressive.zip")
    parser.add_argument("--data", default="data/xauusd_h1.csv")
    parser.add_argument("--feature-set", choices=["basic", "regime", "ultimate"], default="regime")
    parser.add_argument("--base-tf", choices=["M5", "M15", "H1"], default="H1")
    parser.add_argument("--train-end", default="2022-01-01")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--current-position", choices=["flat", "long", "short"], default="flat")
    parser.add_argument("--risk-filter", choices=["none", "dxy_mom20_q75_train"], default="dxy_mom20_q75_train")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--output", default=None, help="Optional path to write JSON signal")
    args = parser.parse_args()

    meta, meta_path, norm_path = load_metadata(args.model)
    feature_df, x_raw, _, feature_set, data_path, base_tf, window = load_features(args, meta)
    train_end = meta.get("train_end", args.train_end)
    position_mode = meta.get("position_mode", "long_only")

    if len(x_raw) < window:
        raise ValueError(f"Need at least {window} feature rows, got {len(x_raw)}")

    x = normalize_features(x_raw, norm_path, feature_df["time"], train_end, window)
    current_pos = {"flat": 0, "long": 1, "short": -1}[args.current_position]
    obs = np.concatenate([x[-window:].reshape(-1), np.array([current_pos], dtype=np.float32)]).astype(np.float32)

    model = PPO.load(args.model, device=args.device)
    action, _ = model.predict(obs, deterministic=True)
    raw_position = action_to_position(int(action), position_mode)

    risk_triggered = False
    risk_value = None
    risk_threshold = None
    if args.risk_filter == "dxy_mom20_q75_train":
        risk_triggered, risk_value, risk_threshold = dxy_mom20_q75_filter(feature_df, train_end)

    final_position = 0 if risk_triggered else raw_position
    latest = feature_df.iloc[-1]
    signal = {
        "timestamp": str(latest["time"]),
        "model": args.model,
        "meta": meta_path,
        "norm_stats": norm_path,
        "data": data_path,
        "feature_set": feature_set,
        "base_tf": base_tf,
        "window": window,
        "position_mode": position_mode,
        "current_position": current_pos,
        "raw_action": int(action),
        "raw_position": raw_position,
        "raw_signal": position_label(raw_position),
        "risk_filter": args.risk_filter,
        "risk_triggered": bool(risk_triggered),
        "risk_metric": "dxy_mom_20" if args.risk_filter != "none" else None,
        "risk_value": risk_value,
        "risk_threshold": risk_threshold,
        "final_position": final_position,
        "final_signal": position_label(final_position),
        "close": float(latest["close"]) if "close" in latest else None,
    }

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(signal, f, indent=2)

    if args.json:
        print(json.dumps(signal, indent=2))
        return

    print("=" * 72)
    print("XAUUSD PPO SIGNAL")
    print("=" * 72)
    print(f"Time:          {signal['timestamp']}")
    print(f"Close:         {signal['close']:.2f}" if signal["close"] is not None else "Close:         n/a")
    print(f"Model:         {signal['model']}")
    print(f"Raw signal:    {signal['raw_signal']} (action={signal['raw_action']})")
    print(f"Risk filter:   {signal['risk_filter']}")
    if args.risk_filter != "none":
        print(f"DXY mom20:     {signal['risk_value']:.6f}")
        print(f"Threshold:     {signal['risk_threshold']:.6f}")
        print(f"Risk active:   {signal['risk_triggered']}")
    print(f"Final signal:  {signal['final_signal']}")
    print("=" * 72)


if __name__ == "__main__":
    main()
