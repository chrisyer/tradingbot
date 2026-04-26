"""Simple Streamlit dashboard for live trading monitoring."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="DRL Trading Monitor", layout="wide")
st.title("📈 DRL Trading Bot Monitoring Dashboard")

st.markdown("Load a monitor snapshot JSON generated during live trading.")

sample = {
    "running_hours": 12.4,
    "current_equity": 1.08,
    "total_return": 0.08,
    "daily_pnl": 0.012,
    "peak_equity": 1.11,
    "current_drawdown": 0.027,
    "sharpe_ratio": 2.15,
    "num_alerts": 3,
    "critical_alerts": 0,
    "shutdown_triggered": False,
}

snapshot_path = st.text_input("Snapshot JSON path", value="monitoring/latest_snapshot.json")
use_sample = st.checkbox("Use sample data", value=not Path(snapshot_path).exists())

if use_sample:
    stats = sample
else:
    try:
        stats = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    except Exception as exc:
        st.error(f"Failed to read snapshot: {exc}")
        st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Equity", f"{stats['current_equity']:.4f}")
col2.metric("Total Return", f"{stats['total_return']:.2%}")
col3.metric("Sharpe Ratio", f"{stats['sharpe_ratio']:.2f}")
col4.metric("Drawdown", f"{stats['current_drawdown']:.2%}")

risk_df = pd.DataFrame(
    {
        "metric": ["Daily PnL", "Alerts", "Critical Alerts", "Shutdown"],
        "value": [
            stats["daily_pnl"],
            stats["num_alerts"],
            stats["critical_alerts"],
            int(stats["shutdown_triggered"]),
        ],
    }
)

st.subheader("Risk State")
st.bar_chart(risk_df.set_index("metric"))
st.json(stats)
