import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.make_features import make_features
from env.xauusd_env import XAUUSDTradingEnv


def main():
    parser = argparse.ArgumentParser(description="Smoke test the XAUUSD trading environment")
    parser.add_argument("--data", default="data/xauusd_h1.csv")
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    df, X, r = make_features(args.data, window=args.window)
    env = XAUUSDTradingEnv(X, r, window=args.window, cost_per_trade=0.0002)

    obs, _ = env.reset()
    total = 0.0
    info = {"equity": 1.0, "pos": 0}
    for _ in range(args.steps):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total += reward
        if term or trunc:
            break

    print("Rows:", len(df), "Features:", X.shape)
    print("Smoke test total reward:", total)
    print("Equity:", info["equity"], "Last pos:", info["pos"])


if __name__ == "__main__":
    main()
