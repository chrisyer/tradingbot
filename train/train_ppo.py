import os
import sys
import numpy as np
import multiprocessing as mp
import argparse
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from features.make_features import make_features, make_regime_features
from features.ultimate_150_features import make_ultimate_features
from env.xauusd_env import XAUUSDTradingEnv

WINDOW = 64
COST = 0.0001          # KEEP CONSISTENT everywhere
N_ENVS = 8             # try 4/8 depending on your Mac

TRAIN_END_DATE = "2022-01-01"

# Training schedule
CHUNK_STEPS = 50_000  # timesteps per chunk
N_CHUNKS = 10          # 10 chunks => 1,000,000 total

SAVE_DIR = "train"
SAVE_PREFIX = "ppo_xauusd"


def main():
    parser = argparse.ArgumentParser(description="Train PPO baseline on XAUUSD features")
    parser.add_argument("--data", default="data/xauusd_h1.csv")
    parser.add_argument("--feature-set", choices=["basic", "regime", "ultimate"], default="basic")
    parser.add_argument("--base-tf", choices=["M5", "M15", "H1"], default="H1")
    parser.add_argument("--train-end", default=TRAIN_END_DATE)
    parser.add_argument("--chunks", type=int, default=N_CHUNKS)
    parser.add_argument("--chunk-steps", type=int, default=CHUNK_STEPS)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--save-prefix", default=SAVE_PREFIX)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--cost", type=float, default=COST)
    parser.add_argument("--turnover-coef", type=float, default=0.0002)
    parser.add_argument("--flat-penalty", type=float, default=0.0)
    parser.add_argument("--hold-bonus", type=float, default=0.0)
    parser.add_argument("--position-mode", choices=["long_only", "long_short"], default="long_only")
    parser.add_argument("--random-start", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decision-interval", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    print(f"Using PPO device: {device}")

    if args.feature_set == "ultimate":
        X_raw, r, timestamps = make_ultimate_features(base_timeframe=args.base_tf)
        timestamps_np = np.asarray(timestamps, dtype="datetime64[ns]")
        data_label = f"ultimate:{args.base_tf}"
    elif args.feature_set == "regime":
        df, X_raw, r = make_regime_features(args.data, window=WINDOW, normalize=False)
        timestamps_np = df["time"].to_numpy()
        data_label = f"regime:{args.data}"
    else:
        df, X_raw, r = make_features(args.data, window=WINDOW, normalize=False)
        timestamps_np = df["time"].to_numpy()
        data_label = args.data

    train_end = np.searchsorted(timestamps_np, np.datetime64(args.train_end))
    X_train_raw, r_train = X_raw[:train_end], r[:train_end]
    X_test_raw, r_test = X_raw[train_end:], r[train_end:]

    # Fit normalization on TRAIN only to avoid test leakage.
    mu = X_train_raw.mean(axis=0, keepdims=True)
    sig = X_train_raw.std(axis=0, keepdims=True) + 1e-8
    X_train = (X_train_raw - mu) / sig
    X_test = (X_test_raw - mu) / sig

    norm_path = f"{SAVE_DIR}/{args.save_prefix}_norm.npz"
    meta_path = f"{SAVE_DIR}/{args.save_prefix}_meta.json"
    np.savez(norm_path, mu=mu, sig=sig)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "data": args.data,
                "data_label": data_label,
                "feature_set": args.feature_set,
                "base_tf": args.base_tf,
                "train_end": args.train_end,
                "window": WINDOW,
                "cost": args.cost,
                "turnover_coef": args.turnover_coef,
                "flat_penalty": args.flat_penalty,
                "hold_bonus": args.hold_bonus,
                "position_mode": args.position_mode,
                "random_start": args.random_start,
                "decision_interval": args.decision_interval,
                "learning_rate": args.learning_rate,
                "target_kl": args.target_kl,
                "feature_count": int(X_raw.shape[1]),
            },
            f,
            indent=2,
        )
    print(f"Saved normalization stats: {norm_path}")
    print(f"Saved training metadata: {meta_path}")

    def make_train_env():
        return XAUUSDTradingEnv(
            X_train,
            r_train,
            window=WINDOW,
            cost_per_trade=args.cost,
            turnover_coef=args.turnover_coef,
            flat_penalty=args.flat_penalty,
            hold_bonus=args.hold_bonus,
            position_mode=args.position_mode,
            random_start=args.random_start,
            decision_interval=args.decision_interval,
            max_episode_steps=20_000,
        )

    # parallel envs
    train_env = SubprocVecEnv([make_train_env for _ in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=1024,     # per env (total per update ~ n_steps * N_ENVS)
        batch_size=256,
        gamma=0.99,
        learning_rate=args.learning_rate,
        target_kl=args.target_kl,
        device=device,
    )

    total = 0
    for i in range(args.chunks):
        model.learn(total_timesteps=args.chunk_steps, reset_num_timesteps=False)
        total += args.chunk_steps

        ckpt_path = f"{SAVE_DIR}/{args.save_prefix}_{total//1000}k"
        model.save(ckpt_path)
        print(f"\n✅ Saved checkpoint: {ckpt_path}.zip\n")

    # also save "latest" for convenience
    model.save(f"{SAVE_DIR}/{args.save_prefix}_latest")
    print(f"\n✅ Saved latest: {SAVE_DIR}/{args.save_prefix}_latest.zip\n")

    # quick small evaluation at the end (optional)
    test_env = XAUUSDTradingEnv(
        X_test,
        r_test,
        window=WINDOW,
        cost_per_trade=args.cost,
        turnover_coef=args.turnover_coef,
        flat_penalty=args.flat_penalty,
        hold_bonus=args.hold_bonus,
        position_mode=args.position_mode,
        decision_interval=args.decision_interval,
        max_episode_steps=None,
    )
    obs, _ = test_env.reset()
    equities, positions = [], []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = test_env.step(action)
        equities.append(info["equity"])
        positions.append(info["pos"])
        if term or trunc:
            break

    trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    pos_arr = np.array(positions)
    pct_time_short = float(np.mean(pos_arr == -1))
    pct_time_flat = float(np.mean(pos_arr == 0))
    pct_time_long = float(np.mean(pos_arr == 1))

    print("FINAL QUICK TEST equity:", float(equities[-1]))
    print("FINAL QUICK TEST trades:", trades)
    print("FINAL QUICK TEST % time short:", pct_time_short)
    print("FINAL QUICK TEST % time flat:", pct_time_flat)
    print("FINAL QUICK TEST % time long:", pct_time_long)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # macOS safe
    main()
