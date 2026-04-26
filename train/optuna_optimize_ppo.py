"""Optuna-based hyperparameter optimization for PPO on XAUUSDTradingEnv."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import optuna
from stable_baselines3 import PPO

from env.xauusd_env import XAUUSDTradingEnv
from features.make_features import make_features


DEFAULT_WINDOW = 64
DEFAULT_COST = 0.0001
DEFAULT_TRAIN_END_DATE = "2022-01-01"


def evaluate_model(model: PPO, env: XAUUSDTradingEnv) -> dict[str, float]:
    obs, _ = env.reset()
    equities = []
    rewards = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        equities.append(float(info["equity"]))
        if terminated or truncated:
            break

    returns = np.diff(np.array([1.0] + equities, dtype=np.float64))
    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 1e-12:
        sharpe = (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252 * 24)

    return {
        "final_equity": equities[-1] if equities else 1.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "sharpe": float(sharpe),
    }


def build_train_test_data(
    data_path: str, window: int, train_end_date: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Keep normalize=False here and fit stats on train split only to avoid leakage.
    df, features_raw, returns = make_features(data_path, window=window, normalize=False)
    train_end = np.searchsorted(df["time"].to_numpy(), np.datetime64(train_end_date))
    if train_end <= 0 or train_end >= len(features_raw):
        raise ValueError(
            f"Invalid train split: train_end={train_end}, total={len(features_raw)}. "
            f"Check --train-end ({train_end_date}) against dataset timestamps."
        )
    x_train_raw = features_raw[:train_end]
    x_test_raw = features_raw[train_end:]

    mu = x_train_raw.mean(axis=0, keepdims=True)
    sig = x_train_raw.std(axis=0, keepdims=True) + 1e-8
    x_train = (x_train_raw - mu) / sig
    x_test = (x_test_raw - mu) / sig
    return x_train, returns[:train_end], x_test, returns[train_end:]


def optimize(
    data_path: str,
    n_trials: int,
    timesteps: int,
    output: str,
    train_end_date: str = DEFAULT_TRAIN_END_DATE,
    window: int = DEFAULT_WINDOW,
):
    x_train, r_train, x_test, r_test = build_train_test_data(data_path, window, train_end_date)

    train_env = XAUUSDTradingEnv(x_train, r_train, window=window, cost_per_trade=DEFAULT_COST, max_episode_steps=20_000)
    test_env = XAUUSDTradingEnv(x_test, r_test, window=window, cost_per_trade=DEFAULT_COST, max_episode_steps=None)

    def objective(trial: optuna.Trial) -> float:
        n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        gamma = trial.suggest_float("gamma", 0.95, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-2, log=True)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

        # SB3 requires batch_size <= n_steps for non-vector envs.
        batch_size = min(batch_size, n_steps)

        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=0,
            n_steps=n_steps,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            learning_rate=lr,
            ent_coef=ent_coef,
            clip_range=clip_range,
        )
        model.learn(total_timesteps=timesteps)

        metrics = evaluate_model(model, test_env)
        trial.set_user_attr("final_equity", metrics["final_equity"])
        trial.set_user_attr("sharpe", metrics["sharpe"])

        # Blend equity and sharpe.
        return metrics["final_equity"] + 0.1 * metrics["sharpe"]

    study = optuna.create_study(direction="maximize", study_name="xauusd_ppo_optuna")
    study.optimize(objective, n_trials=n_trials)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_trial": {
            "number": study.best_trial.number,
            "user_attrs": study.best_trial.user_attrs,
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✅ Optimization done. Results saved to {output_path}")
    print(json.dumps(payload, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Optuna search for PPO hyperparameters")
    parser.add_argument("--data", default="data/xauusd_1h.csv", help="Input OHLCV CSV path")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--timesteps", type=int, default=75_000, help="Training timesteps per trial")
    parser.add_argument("--train-end", default=DEFAULT_TRAIN_END_DATE, help="Train/test split date (YYYY-MM-DD)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Lookback window")
    parser.add_argument("--output", default="train/optuna_best_params.json", help="Output JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    optimize(
        data_path=args.data,
        n_trials=args.trials,
        timesteps=args.timesteps,
        output=args.output,
        train_end_date=args.train_end,
        window=args.window,
    )
