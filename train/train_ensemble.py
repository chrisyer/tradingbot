"""Train and evaluate an ensemble of PPO agents."""

import argparse
import os
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from env.xauusd_env import XAUUSDTradingEnv
from features.make_features import make_features


def split_data(data_path: str, window: int, train_end_date: str):
    df, x_raw, r = make_features(data_path, window=window, normalize=False)
    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64(train_end_date))
    x_train_raw, x_test_raw = x_raw[:train_end], x_raw[train_end:]
    mu = x_train_raw.mean(axis=0, keepdims=True)
    sig = x_train_raw.std(axis=0, keepdims=True) + 1e-8
    x_train = (x_train_raw - mu) / sig
    x_test = (x_test_raw - mu) / sig
    return x_train, r[:train_end], x_test, r[train_end:]


def majority_vote(actions: list[int]) -> int:
    ones = sum(actions)
    zeros = len(actions) - ones
    return 1 if ones > zeros else 0


def evaluate_ensemble(models: list[PPO], env: XAUUSDTradingEnv):
    obs, _ = env.reset()
    equities = []

    while True:
        actions = [int(model.predict(obs, deterministic=True)[0]) for model in models]
        action = majority_vote(actions)
        obs, _, terminated, truncated, info = env.step(action)
        equities.append(float(info["equity"]))
        if terminated or truncated:
            break

    return equities[-1] if equities else 1.0


def main():
    parser = argparse.ArgumentParser(description="Train an ensemble of PPO models")
    parser.add_argument("--data", default="data/xauusd_1h.csv")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--train-end", default="2022-01-01")
    parser.add_argument("--timesteps", type=int, default=150_000)
    parser.add_argument("--num-models", type=int, default=5)
    parser.add_argument("--save-dir", default="train/ensemble")
    args = parser.parse_args()

    x_train, r_train, x_test, r_test = split_data(args.data, args.window, args.train_end)

    os.makedirs(args.save_dir, exist_ok=True)

    models = []
    for seed in range(args.num_models):
        train_env = XAUUSDTradingEnv(x_train, r_train, window=args.window, cost_per_trade=0.0001, max_episode_steps=20_000)
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            seed=42 + seed,
            n_steps=1024,
            batch_size=256,
            gamma=0.99,
            learning_rate=3e-4,
        )
        model.learn(total_timesteps=args.timesteps)
        model_path = Path(args.save_dir) / f"ppo_seed_{42 + seed}"
        model.save(model_path)
        models.append(model)
        print(f"✅ saved {model_path}.zip")

    test_env = XAUUSDTradingEnv(x_test, r_test, window=args.window, cost_per_trade=0.0001)
    final_equity = evaluate_ensemble(models, test_env)
    print(f"✅ Ensemble final equity: {final_equity:.4f}")


if __name__ == "__main__":
    main()
