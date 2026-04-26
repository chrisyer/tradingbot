"""Train one PPO policy across multiple assets (e.g., XAUUSD/EURUSD/BTCUSD/SPX)."""

import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.multi_asset_env import AssetDataset, MultiAssetTradingEnv
from features.make_features import make_features


def load_asset_dataset(symbol: str, csv_path: str, window: int, train_end_date: str) -> tuple[AssetDataset, AssetDataset]:
    # NOTE: keep normalize=False here, then fit normalization stats on train split only
    # to avoid train/test leakage.
    df, features_raw, returns = make_features(csv_path, window=window, normalize=False)
    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64(train_end_date))
    if train_end <= 0 or train_end >= len(features_raw):
        raise ValueError(
            f"Invalid train split for {symbol}: train_end={train_end}, total={len(features_raw)}. "
            f"Check --train-end ({train_end_date}) against dataset timestamps."
        )

    x_train_raw = features_raw[:train_end]
    x_test_raw = features_raw[train_end:]
    mu = x_train_raw.mean(axis=0, keepdims=True)
    sig = x_train_raw.std(axis=0, keepdims=True) + 1e-8
    x_train = (x_train_raw - mu) / sig
    x_test = (x_test_raw - mu) / sig

    train_ds = AssetDataset(symbol=symbol, features=x_train, returns=returns[:train_end])
    test_ds = AssetDataset(symbol=symbol, features=x_test, returns=returns[train_end:])
    return train_ds, test_ds


def evaluate(model: PPO, test_datasets: list[AssetDataset], window: int):
    metrics = {}
    for ds in test_datasets:
        env = MultiAssetTradingEnv([ds], window=window)
        obs, _ = env.reset()
        equity_curve = []
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            equity_curve.append(info["equity"])
            if terminated or truncated:
                break
        metrics[ds.symbol] = equity_curve[-1] if equity_curve else 1.0
    return metrics


def parse_assets(asset_args: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for item in asset_args:
        if "=" not in item:
            raise ValueError(f"Invalid --asset '{item}', expected SYMBOL=path.csv")
        symbol, path = item.split("=", 1)
        parsed.append((symbol.strip().upper(), path.strip()))
    return parsed


def main():
    parser = argparse.ArgumentParser(description="Train PPO across multiple markets")
    parser.add_argument(
        "--asset",
        action="append",
        required=True,
        help="Asset mapping in form SYMBOL=csv_path (repeat for each asset)",
    )
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-end", default="2022-01-01")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--save-path", default="train/ppo_multi_asset_latest")
    args = parser.parse_args()

    asset_specs = parse_assets(args.asset)

    train_datasets = []
    test_datasets = []
    for symbol, csv_path in asset_specs:
        if not Path(csv_path).exists():
            raise FileNotFoundError(f"Missing CSV for {symbol}: {csv_path}")
        train_ds, test_ds = load_asset_dataset(symbol, csv_path, args.window, args.train_end)
        train_datasets.append(train_ds)
        test_datasets.append(test_ds)

    env = MultiAssetTradingEnv(train_datasets, window=args.window, max_episode_steps=20_000)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
    )
    model.learn(total_timesteps=args.timesteps)

    os.makedirs(Path(args.save_path).parent, exist_ok=True)
    model.save(args.save_path)

    per_asset_equity = evaluate(model, test_datasets, args.window)
    print("✅ Multi-asset training complete")
    print(f"Model saved: {args.save_path}.zip")
    print("Test equity by asset:")
    for symbol, equity in per_asset_equity.items():
        print(f"  {symbol}: {equity:.4f}")


if __name__ == "__main__":
    main()
