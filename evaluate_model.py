"""
Evaluate Trained DreamerV3 Model

This script evaluates a trained model on validation/test data and generates
comprehensive performance metrics and visualizations.
"""

import os
import sys
import argparse
import json
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.ultimate_150_features import make_ultimate_features
from features.make_features import make_features, make_regime_features
from models.dreamer_agent import DreamerV3Agent
from env.xauusd_env import XAUUSDTradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingEnvironment:
    """Simple trading environment for evaluation"""
    def __init__(self, features, returns, window=64, cost_per_trade=0.0001):
        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.T = len(self.r)
        self.reset()

    def reset(self):
        self.t = self.window
        self.pos = 0
        self.equity = 1.0
        return self._get_obs()

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def step(self, action_onehot):
        # action_onehot: [flat, long] probabilities
        action = np.argmax(action_onehot)  # 0 = flat, 1 = long

        # Get return
        ret = self.r[self.t]

        # Calculate reward
        if action == 1:  # Long position
            reward = ret - self.cost  # Profit/loss minus transaction cost
            self.pos = 1
        else:  # Flat position
            reward = -self.cost if self.pos == 1 else 0  # Only cost if exiting position
            self.pos = 0

        # Update equity
        if action == 1:
            self.equity *= (1 + ret - self.cost)
        elif self.pos == 1:  # Closing position
            self.equity *= (1 - self.cost)

        # Move forward
        self.t += 1
        done = (self.t >= self.T)

        next_obs = self._get_obs() if not done else self._get_obs()

        return next_obs, reward, done, {'equity': self.equity, 'position': self.pos}

    @property
    def observation_space(self):
        return self.window * self.X.shape[1] + 1

    @property
    def action_space(self):
        return 2  # flat or long


def evaluate_model(agent, env, timestamps):
    """
    Evaluate model on environment

    Returns:
        metrics: Dict of performance metrics
        equity_curve: Array of equity over time
        positions: Array of positions over time
        dates: Corresponding timestamps
    """
    logger.info("🎯 Running evaluation...")

    obs = env.reset()
    h, z = None, None

    equity_curve = [1.0]
    positions = []
    rewards = []

    for step in tqdm(range(env.T - env.window), desc="Evaluating"):
        # Get action from agent
        action, (h, z) = agent.act(obs, h, z, deterministic=True)
        action_idx = np.argmax(action)
        action_onehot = np.eye(env.action_space)[action_idx]

        # Step environment
        obs, reward, done, info = env.step(action_onehot)

        equity_curve.append(info['equity'])
        positions.append(info['position'])
        rewards.append(reward)

        if done:
            break

    # Convert to arrays
    equity_curve = np.array(equity_curve)
    positions = np.array(positions)
    rewards = np.array(rewards)
    dates = timestamps[env.window:env.window + len(positions)]

    # Calculate metrics
    returns = np.diff(equity_curve) / equity_curve[:-1]

    total_return = (equity_curve[-1] - 1) * 100

    # Annualized metrics (assuming 252 trading days)
    days = len(equity_curve) / (252 * 24 * 12)  # Convert 5-min bars to years
    annual_return = ((equity_curve[-1] ** (1 / days)) - 1) * 100 if days > 0 else 0

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 12)

    # Max drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax
    max_drawdown = np.min(drawdown) * 100

    # Win rate
    win_rate = np.mean(rewards > 0) * 100 if len(rewards) > 0 else 0

    # Position statistics
    long_pct = np.mean(positions) * 100 if len(positions) > 0 else 0

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_equity': equity_curve[-1],
        'num_trades': len(positions),
        'long_percentage': long_pct,
    }

    return metrics, equity_curve, positions, dates


def plot_results(equity_curve, positions, dates, metrics, save_path='results.png'):
    """Plot evaluation results"""
    logger.info("📊 Creating visualizations...")
    equity_dates = dates[: len(equity_curve) - 1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Equity curve
    axes[0].plot(equity_dates, equity_curve[1:], linewidth=2, color='green')
    axes[0].set_title(f'Equity Curve - Final: ${equity_curve[-1]:.2f}', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Equity ($)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    axes[0].legend()

    # Drawdown
    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / cummax * 100
    axes[1].fill_between(equity_dates, drawdown[1:], 0, color='red', alpha=0.3)
    axes[1].plot(equity_dates, drawdown[1:], color='darkred', linewidth=1)
    axes[1].set_title(f'Drawdown - Max: {metrics["max_drawdown"]:.2f}%', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Positions
    axes[2].fill_between(dates, positions, 0, alpha=0.3, color='blue')
    axes[2].set_title(
        f'Positions - Short: {metrics.get("short_percentage", 0):.1f}% | '
        f'Flat: {metrics.get("flat_percentage", 0):.1f}% | '
        f'Long: {metrics["long_percentage"]:.1f}%',
        fontsize=14,
        fontweight='bold',
    )
    axes[2].set_ylabel('Position', fontsize=12)
    axes[2].set_xlabel('Date', fontsize=12)
    axes[2].set_ylim(-1.1, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info(f"✅ Plot saved to: {save_path}")

    return fig


def infer_periods_per_year(timestamps):
    if len(timestamps) < 3:
        return 252
    ts = pd.to_datetime(pd.Series(timestamps)).sort_values()
    delta_seconds = ts.diff().dropna().dt.total_seconds()
    median_seconds = float(delta_seconds[delta_seconds > 0].median())
    if not np.isfinite(median_seconds) or median_seconds <= 0:
        return 252
    return (365.25 * 24 * 3600) / median_seconds


def calculate_metrics(equity_curve, rewards, positions, dates):
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int32)
    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12)
    total_return = (equity_curve[-1] - 1.0) * 100

    periods_per_year = infer_periods_per_year(dates)
    years = len(returns) / periods_per_year if periods_per_year > 0 else 0
    annual_return = ((equity_curve[-1] ** (1 / years)) - 1) * 100 if years > 0 and equity_curve[-1] > 0 else 0
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(periods_per_year) if len(returns) else 0

    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cummax) / np.maximum(cummax, 1e-12)
    max_drawdown = np.min(drawdown) * 100
    trade_count = int(np.sum(np.abs(np.diff(np.r_[0, positions])) > 0))
    short_pct = np.mean(positions == -1) * 100 if len(positions) else 0
    flat_pct = np.mean(positions == 0) * 100 if len(positions) else 0
    long_pct = np.mean(positions == 1) * 100 if len(positions) else 0
    exposure_pct = np.mean(np.abs(positions)) * 100 if len(positions) else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': np.mean(rewards > 0) * 100 if len(rewards) else 0,
        'final_equity': equity_curve[-1],
        'num_trades': trade_count,
        'short_percentage': short_pct,
        'flat_percentage': flat_pct,
        'long_percentage': long_pct,
        'exposure_percentage': exposure_pct,
    }


def select_period_mask(timestamps, period):
    if period == 'validation':
        return (timestamps >= '2022-01-01') & (timestamps < '2024-01-01'), "VALIDATION (2022-2023)"
    if period == 'test':
        return timestamps >= '2024-01-01', "TEST (2024-2025)"
    return np.ones(len(timestamps), dtype=bool), "ALL DATA"


def build_risk_filter(args, feature_df, timestamps):
    if args.risk_filter == "none":
        return None
    if args.risk_filter != "dxy_mom20_q75_train":
        raise ValueError(f"Unsupported risk filter: {args.risk_filter}")
    if "dxy_mom_20" not in feature_df.columns:
        raise ValueError("--risk-filter dxy_mom20_q75_train requires regime features with dxy_mom_20")

    train_mask = pd.to_datetime(timestamps) < pd.Timestamp(args.train_end)
    threshold = feature_df.loc[train_mask, "dxy_mom_20"].quantile(0.75)
    risk_mask = feature_df["dxy_mom_20"].to_numpy(dtype=np.float64) > float(threshold)
    logger.info(
        "Risk filter dxy_mom20_q75_train: threshold=%.6f, blocked=%.2f%%",
        threshold,
        risk_mask.mean() * 100,
    )
    return risk_mask


def load_ppo_context(args):
    from stable_baselines3 import PPO

    model_stem = args.model[:-4] if args.model.endswith(".zip") else args.model
    stats_stems = [model_stem]
    if model_stem.endswith("_latest"):
        stats_stems.append(model_stem[: -len("_latest")])
    elif re.search(r"_\d+k$", model_stem):
        stats_stems.append(model_stem.rsplit("_", 1)[0])
    stats_stems = list(dict.fromkeys(stats_stems))

    meta_path = next(
        (f"{stem}_meta.json" for stem in stats_stems if os.path.exists(f"{stem}_meta.json")),
        f"{stats_stems[-1]}_meta.json",
    )
    if args.norm_stats:
        norm_path = args.norm_stats
    else:
        norm_path = next(
            (f"{stem}_norm.npz" for stem in stats_stems if os.path.exists(f"{stem}_norm.npz")),
            f"{stats_stems[-1]}_norm.npz",
        )
    env_kwargs = {
        "cost_per_trade": args.cost,
        "turnover_coef": args.turnover_coef,
        "flat_penalty": args.flat_penalty,
        "hold_bonus": args.hold_bonus,
        "position_mode": args.position_mode,
        "decision_interval": args.decision_interval,
    }
    feature_set = args.feature_set
    base_tf = args.base_tf

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        args.train_end = meta.get("train_end", args.train_end)
        args.window = int(meta.get("window", args.window))
        feature_set = meta.get("feature_set", feature_set)
        base_tf = meta.get("base_tf", base_tf)
        if not args.override_meta_env:
            env_kwargs = {
                "cost_per_trade": float(meta.get("cost", env_kwargs["cost_per_trade"])),
                "turnover_coef": float(meta.get("turnover_coef", env_kwargs["turnover_coef"])),
                "flat_penalty": float(meta.get("flat_penalty", env_kwargs["flat_penalty"])),
                "hold_bonus": float(meta.get("hold_bonus", env_kwargs["hold_bonus"])),
                "position_mode": meta.get("position_mode", env_kwargs["position_mode"]),
                "decision_interval": int(meta.get("decision_interval", env_kwargs["decision_interval"])),
            }
        logger.info(f"Loaded PPO metadata: {meta_path}")
        if args.override_meta_env:
            logger.info(f"Using command-line environment overrides: {env_kwargs}")

    logger.info("📊 Loading PPO features...")
    if feature_set == "ultimate":
        X_raw, returns, timestamps_idx = make_ultimate_features(base_timeframe=base_tf)
        timestamps = pd.Series(pd.to_datetime(timestamps_idx))
        feature_df = pd.DataFrame({"time": timestamps})
    elif feature_set == "regime":
        feature_df, X_raw, returns = make_regime_features(args.data, window=args.window, normalize=False)
        timestamps = pd.to_datetime(feature_df["time"])
    else:
        feature_df, X_raw, returns = make_features(args.data, window=args.window, normalize=False)
        timestamps = pd.to_datetime(feature_df["time"])

    risk_mask = build_risk_filter(args, feature_df, timestamps)

    if os.path.exists(norm_path):
        stats = np.load(norm_path)
        mu = stats["mu"]
        sig = stats["sig"]
        logger.info(f"Loaded normalization stats: {norm_path}")
    else:
        train_end = np.searchsorted(timestamps.to_numpy(), np.datetime64(args.train_end))
        if train_end <= args.window:
            raise ValueError(f"Not enough training data before {args.train_end}")
        mu = X_raw[:train_end].mean(axis=0, keepdims=True)
        sig = X_raw[:train_end].std(axis=0, keepdims=True) + 1e-8
        logger.warning(f"Normalization stats not found ({norm_path}); recomputed from data before {args.train_end}")
    X = (X_raw - mu) / sig

    logger.info(f"🤖 Loading PPO model from: {args.model}")
    model = PPO.load(args.model, device=args.device)

    return model, X, returns, timestamps, env_kwargs, risk_mask


def run_ppo_on_slice(model, X_eval, returns_eval, timestamps_eval, args, env_kwargs, period_name, risk_mask=None):
    timestamps_eval = pd.Series(pd.to_datetime(timestamps_eval)).reset_index(drop=True)
    if len(X_eval) <= args.window + 1:
        raise ValueError(f"Not enough samples in {period_name}: {len(X_eval)}")

    env = XAUUSDTradingEnv(
        X_eval,
        returns_eval,
        window=args.window,
        **env_kwargs,
        force_flat_mask=risk_mask,
        max_episode_steps=None,
    )

    obs, _ = env.reset()
    equity_curve = [1.0]
    positions = []
    rewards = []

    logger.info(f"🎯 Evaluating PPO on {period_name}: {len(X_eval):,} samples")
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        equity_curve.append(info["equity"])
        positions.append(info["pos"])
        rewards.append(reward)
        if term or trunc:
            break

    dates = timestamps_eval.iloc[args.window : args.window + len(positions)].to_numpy()
    metrics = calculate_metrics(equity_curve, rewards, positions, dates)
    return metrics, np.array(equity_curve), np.array(positions), dates, rewards


def calculate_buy_hold_metrics(returns_eval, timestamps_eval, window):
    returns_eval = np.asarray(returns_eval, dtype=np.float64)
    bh_returns = returns_eval[window:]
    timestamps_eval = pd.Series(pd.to_datetime(timestamps_eval)).reset_index(drop=True)
    dates = timestamps_eval.iloc[window : window + len(bh_returns)].to_numpy()
    equity_curve = np.r_[1.0, np.cumprod(1.0 + bh_returns)]
    positions = np.ones(len(bh_returns), dtype=np.int32)
    return calculate_metrics(equity_curve, bh_returns, positions, dates)


def evaluate_ppo(args):
    model, X, returns, timestamps, env_kwargs, risk_mask = load_ppo_context(args)

    mask, period_name = select_period_mask(timestamps, args.period)
    X_eval = X[mask]
    returns_eval = returns[mask]
    risk_eval = None if risk_mask is None else risk_mask[mask]
    timestamps_eval = timestamps[mask].reset_index(drop=True)
    metrics, equity_curve, positions, dates, rewards = run_ppo_on_slice(
        model, X_eval, returns_eval, timestamps_eval, args, env_kwargs, period_name, risk_eval
    )
    print_metrics(metrics, title=f"PPO EVALUATION RESULTS - {period_name}")
    plot_results(equity_curve, positions, dates, metrics, save_path=args.save_plot)

    results_df = pd.DataFrame({
        'timestamp': dates,
        'equity': equity_curve[1:],
        'position': positions,
        'reward': rewards,
    })
    csv_path = args.save_plot.replace('.png', '.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Detailed results saved to: {csv_path}")


def plot_walk_forward(summary_df, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    colors = np.where(summary_df["total_return"] >= 0, "green", "red")

    axes[0].bar(summary_df["period"], summary_df["total_return"], color=colors, alpha=0.75)
    if "buy_hold_return" in summary_df:
        axes[0].plot(
            summary_df["period"],
            summary_df["buy_hold_return"],
            marker="o",
            color="black",
            linewidth=1.5,
            label="Buy & hold",
        )
        axes[0].legend()
    axes[0].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[0].set_title("Walk-Forward Total Return by Period", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Return (%)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(summary_df["period"], summary_df["max_drawdown"], marker="o", color="darkred")
    axes[1].set_title("Max Drawdown by Period", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Max DD (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(summary_df["period"], summary_df["num_trades"], color="steelblue", alpha=0.75)
    axes[2].set_title("Trades by Period", fontsize=14, fontweight="bold")
    axes[2].set_ylabel("Trades")
    axes[2].set_xlabel("Period")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"✅ Walk-forward plot saved to: {save_path}")


def evaluate_ppo_walk_forward(args):
    model, X, returns, timestamps, env_kwargs, risk_mask = load_ppo_context(args)
    timestamps = pd.Series(pd.to_datetime(timestamps))

    rows = []
    start_year = args.walk_start_year
    end_year = args.walk_end_year or int(timestamps.dt.year.max())
    for year in range(start_year, end_year + 1, args.walk_step_years):
        start = pd.Timestamp(year=year, month=1, day=1)
        end = start + pd.DateOffset(years=args.walk_window_years)
        mask = (timestamps >= start) & (timestamps < end)
        if int(mask.sum()) <= args.window + 1:
            logger.warning(f"Skipping {start:%Y-%m-%d} to {end:%Y-%m-%d}: not enough samples")
            continue

        period_name = f"{start:%Y-%m-%d} to {end:%Y-%m-%d}"
        metrics, _, _, _, _ = run_ppo_on_slice(
            model,
            X[mask.to_numpy()],
            returns[mask.to_numpy()],
            timestamps[mask].reset_index(drop=True),
            args,
            env_kwargs,
            period_name,
            None if risk_mask is None else risk_mask[mask.to_numpy()],
        )
        bh_metrics = calculate_buy_hold_metrics(
            returns[mask.to_numpy()],
            timestamps[mask].reset_index(drop=True),
            args.window,
        )
        rows.append(
            {
                "period": f"{year}-{end.year - 1}" if args.walk_window_years > 1 else str(year),
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
                "samples": int(mask.sum()),
                "buy_hold_return": bh_metrics["total_return"],
                "excess_return": metrics["total_return"] - bh_metrics["total_return"],
                "buy_hold_max_drawdown": bh_metrics["max_drawdown"],
                **metrics,
            }
        )

    if not rows:
        raise ValueError("No walk-forward windows were evaluated")

    summary_df = pd.DataFrame(rows)
    csv_path = args.save_plot.replace(".png", ".csv")
    summary_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Walk-forward summary saved to: {csv_path}")

    logger.info("\n" + "=" * 90)
    logger.info("📊 PPO WALK-FORWARD SUMMARY")
    logger.info("=" * 90)
    logger.info(
        summary_df[
            [
                "period",
                "total_return",
                "buy_hold_return",
                "excess_return",
                "sharpe_ratio",
                "max_drawdown",
                "final_equity",
                "num_trades",
                "long_percentage",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:,.2f}")
    )
    logger.info("=" * 90 + "\n")

    plot_walk_forward(summary_df, args.save_plot)


def print_metrics(metrics, title="EVALUATION RESULTS"):
    """Pretty print metrics"""
    logger.info("\n" + "="*70)
    logger.info(f"📊 {title}")
    logger.info("="*70)
    logger.info(f"💰 Total Return:      {metrics['total_return']:>10.2f}%")
    logger.info(f"📈 Annual Return:     {metrics['annual_return']:>10.2f}%")
    logger.info(f"📉 Max Drawdown:      {metrics['max_drawdown']:>10.2f}%")
    logger.info(f"⚡ Sharpe Ratio:      {metrics['sharpe_ratio']:>10.2f}")
    logger.info(f"🎯 Win Rate:          {metrics['win_rate']:>10.2f}%")
    logger.info(f"💵 Final Equity:      {metrics['final_equity']:>10.2f}x")
    logger.info(f"📉 Short %:           {metrics.get('short_percentage', 0):>10.2f}%")
    logger.info(f"➖ Flat %:            {metrics.get('flat_percentage', 0):>10.2f}%")
    logger.info(f"📊 Long %:            {metrics['long_percentage']:>10.2f}%")
    logger.info(f"📌 Exposure %:        {metrics.get('exposure_percentage', metrics['long_percentage']):>10.2f}%")
    logger.info(f"🔄 Num Trades:        {metrics['num_trades']:>10,}")
    logger.info("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained trading models')
    parser.add_argument('--kind', choices=['ppo', 'dreamer'], default='ppo')
    parser.add_argument('--model', type=str, default='train/ppo_xauusd_latest.zip', help='Path to PPO .zip model')
    parser.add_argument('--data', type=str, default='data/xauusd_h1.csv', help='OHLC CSV for PPO evaluation')
    parser.add_argument('--feature-set', choices=['basic', 'regime', 'ultimate'], default='basic')
    parser.add_argument('--base-tf', choices=['M5', 'M15', 'H1'], default='H1')
    parser.add_argument('--train-end', type=str, default='2022-01-01')
    parser.add_argument('--window', type=int, default=64)
    parser.add_argument('--cost', type=float, default=0.0001)
    parser.add_argument('--turnover-coef', type=float, default=0.0002)
    parser.add_argument('--flat-penalty', type=float, default=0.0)
    parser.add_argument('--hold-bonus', type=float, default=0.0)
    parser.add_argument('--position-mode', choices=['long_only', 'long_short'], default='long_only')
    parser.add_argument('--decision-interval', type=int, default=1)
    parser.add_argument(
        '--override-meta-env',
        action='store_true',
        help='Use command-line env/cost parameters instead of PPO metadata values',
    )
    parser.add_argument('--norm-stats', default=None, help='Optional PPO normalization .npz file')
    parser.add_argument(
        '--risk-filter',
        choices=['none', 'dxy_mom20_q75_train'],
        default='none',
        help='Optional evaluation-time risk overlay that can force the position flat',
    )
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--checkpoint', type=str, default='train/dreamer_ultimate/ultimate_150_xauusd_final.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--period', type=str, default='validation', choices=['validation', 'test', 'all', 'walk-forward'],
                       help='Evaluation period (validation=2022-2023, test=2024-2025, all=everything)')
    parser.add_argument('--save-plot', type=str, default='evaluation_results.png',
                       help='Path to save results plot')
    parser.add_argument('--walk-start-year', type=int, default=2018)
    parser.add_argument('--walk-end-year', type=int, default=None)
    parser.add_argument('--walk-window-years', type=int, default=1)
    parser.add_argument('--walk-step-years', type=int, default=1)

    args = parser.parse_args()

    if args.kind == 'ppo':
        if args.period == 'walk-forward':
            evaluate_ppo_walk_forward(args)
        else:
            evaluate_ppo(args)
        logger.info("\n🎉 Evaluation complete!")
        return

    # ========== LOAD FEATURES ==========
    logger.info("📊 Loading Ultimate 150+ features...")
    X, returns, timestamps = make_ultimate_features(base_timeframe='M5')

    logger.info(f"✅ Loaded {X.shape[1]} features, {len(X):,} samples")
    logger.info(f"📅 Date range: {timestamps[0]} to {timestamps[-1]}")

    # ========== SELECT PERIOD ==========
    if args.period == 'validation':
        # 2022-2023
        mask = (timestamps >= '2022-01-01') & (timestamps < '2024-01-01')
        period_name = "VALIDATION (2022-2023)"
    elif args.period == 'test':
        # 2024-2025
        mask = (timestamps >= '2024-01-01')
        period_name = "TEST (2024-2025)"
    else:
        # All data
        mask = np.ones(len(timestamps), dtype=bool)
        period_name = "ALL DATA"

    X_eval = X[mask]
    returns_eval = returns[mask]
    timestamps_eval = timestamps[mask]

    logger.info(f"\n📅 Evaluating on {period_name}")
    logger.info(f"   • Samples: {len(X_eval):,}")
    logger.info(f"   • Date range: {timestamps_eval[0]} to {timestamps_eval[-1]}")

    # ========== CREATE ENVIRONMENT ==========
    env = TradingEnvironment(X_eval, returns_eval, window=64, cost_per_trade=0.0001)

    # ========== LOAD AGENT ==========
    logger.info(f"\n🤖 Loading model from: {args.checkpoint}")

    agent = DreamerV3Agent(
        obs_dim=env.observation_space,
        action_dim=env.action_space,
        embed_dim=256,
        hidden_dim=512,
        stoch_dim=32,
        num_categories=32,
        device='cpu'  # Use CPU for evaluation
    )

    if os.path.exists(args.checkpoint):
        agent.load(args.checkpoint)
        logger.info("✅ Model loaded successfully")
    else:
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        return

    # ========== EVALUATE ==========
    metrics, equity_curve, positions, dates = evaluate_model(agent, env, timestamps_eval)

    # ========== PRINT RESULTS ==========
    print_metrics(metrics, title=f"EVALUATION RESULTS - {period_name}")

    # ========== PLOT RESULTS ==========
    plot_results(equity_curve, positions, dates, metrics, save_path=args.save_plot)

    # ========== SAVE DETAILED RESULTS ==========
    results_df = pd.DataFrame({
        'timestamp': dates,
        'equity': equity_curve[1:],  # Skip initial 1.0
        'position': positions,
    })

    csv_path = args.save_plot.replace('.png', '.csv')
    results_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Detailed results saved to: {csv_path}")

    logger.info("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
