"""
Streamlit Dashboard — Health Metrics Forecasting Explorer

Run with:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

from src.data_loader import (
    load_data, clean_data, add_time_features,
    make_lag_features, train_test_split_ts,
)
from src.evaluate import evaluate_all

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Health Metrics Forecasting",
    page_icon="💓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card { background: #1e1e2e; border-radius: 12px; padding: 16px; text-align: center; }
    .stMetric label { font-size: 0.75rem; color: #888; }
    section[data-testid="stSidebar"] { background: #0f0f1a; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data
def get_data():
    from data.generate_data import generate_health_data
    df = generate_health_data(n_days=730)
    df = clean_data(df)
    df = add_time_features(df)
    df = make_lag_features(df, "steps", lags=[1, 2, 3, 7, 14])
    df = make_lag_features(df, "heart_rate", lags=[1, 7])
    return df


@st.cache_data
def load_predictions():
    pred_path = Path("results/predictions.csv")
    if pred_path.exists():
        return pd.read_csv(pred_path, parse_dates=["date"])
    return None


@st.cache_data
def load_metrics():
    metrics_path = Path("results/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/heart-with-pulse.png", width=60)
    st.title("⚙️ Controls")
    st.markdown("---")

    page = st.radio("Navigate", ["📊 EDA", "🤖 Model Results", "🔮 Forecast"])
    st.markdown("---")

    metric_choice = st.selectbox(
        "Health Metric",
        ["steps", "heart_rate", "sleep_hours", "calories", "active_minutes"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    date_range = st.date_input(
        "Date Range",
        value=[pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")],
    )


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading health data..."):
    df = get_data()

if len(date_range) == 2:
    df = df.loc[str(date_range[0]):str(date_range[1])]


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EDA
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 EDA":
    st.title("💓 Health Metrics — Exploratory Analysis")

    # ── KPI row ───────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Steps",          f"{df['steps'].mean():,.0f}")
    c2.metric("Avg Heart Rate",     f"{df['heart_rate'].mean():.1f} bpm")
    c3.metric("Avg Sleep",          f"{df['sleep_hours'].mean():.1f} hrs")
    c4.metric("Avg Calories",       f"{df['calories'].mean():,.0f}")
    c5.metric("Days Tracked",       f"{len(df)}")
    st.markdown("---")

    # ── Time series chart ─────────────────────────────────────────────────────
    st.subheader(f"📈 {metric_choice.replace('_',' ').title()} Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df[metric_choice],
        mode="lines",
        line=dict(color="#7c3aed", width=1.5),
        name=metric_choice,
    ))
    # 7-day rolling average
    roll = df[metric_choice].rolling(7).mean()
    fig.add_trace(go.Scatter(
        x=df.index, y=roll,
        mode="lines",
        line=dict(color="#f97316", width=2.5),
        name="7-day MA",
    ))
    fig.update_layout(
        template="plotly_dark", height=380,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Weekly heatmap ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📅 Average by Day of Week")
        dow_avg = df.groupby("day_of_week")[metric_choice].mean()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        fig2 = px.bar(
            x=day_names, y=dow_avg.values,
            labels={"x": "Day", "y": metric_choice},
            color=dow_avg.values,
            color_continuous_scale="Purples",
            template="plotly_dark",
        )
        fig2.update_layout(showlegend=False, height=300, margin=dict(t=10))
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("📊 Distribution")
        fig3 = px.histogram(
            df, x=metric_choice, nbins=40,
            color_discrete_sequence=["#7c3aed"],
            template="plotly_dark",
        )
        fig3.update_layout(height=300, margin=dict(t=10))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.subheader("🔗 Correlation Matrix")
    corr_cols = ["steps", "heart_rate", "sleep_hours", "calories", "active_minutes", "spo2"]
    corr = df[corr_cols].corr()
    fig4 = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        template="plotly_dark",
        zmin=-1, zmax=1,
    )
    fig4.update_layout(height=450, margin=dict(t=10))
    st.plotly_chart(fig4, use_container_width=True)

    # ── Monthly trends ────────────────────────────────────────────────────────
    st.subheader("🗓️ Monthly Trends")
    df["month_name"] = df.index.strftime("%b %Y")
    monthly = df.groupby(df.index.to_period("M"))[metric_choice].mean()
    monthly.index = monthly.index.astype(str)
    fig5 = px.bar(
        x=monthly.index, y=monthly.values,
        labels={"x": "Month", "y": metric_choice},
        color=monthly.values,
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig5.update_layout(height=320, margin=dict(t=10))
    st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL RESULTS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 Model Results":
    st.title("🤖 Model Comparison Results")

    metrics = load_metrics()
    predictions = load_predictions()

    if metrics is None:
        st.warning("No results found. Run `python src/train.py` first, then refresh.")
        st.code("python src/train.py --models xgb lgbm prophet lstm")
        st.stop()

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.subheader("📋 Performance Metrics")
    metrics_df = pd.DataFrame(metrics).T[["MAE", "RMSE", "MAPE", "R2"]].round(3)

    # Highlight best per column
    def highlight_best(s):
        is_best = s == (s.min() if s.name != "R2" else s.max())
        return ["background-color: #1e3a1e; color: #4ade80" if v else "" for v in is_best]

    st.dataframe(
        metrics_df.style.apply(highlight_best),
        use_container_width=True,
    )

    # ── Bar chart comparison ───────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            metrics_df.reset_index(), x="index", y="MAE",
            title="Mean Absolute Error (lower is better)",
            color="MAE", color_continuous_scale="Reds_r",
            template="plotly_dark",
        )
        fig.update_layout(height=350, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            metrics_df.reset_index(), x="index", y="R2",
            title="R² Score (higher is better)",
            color="R2", color_continuous_scale="Greens",
            template="plotly_dark",
        )
        fig.update_layout(height=350, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

    # ── Predictions vs Actual ─────────────────────────────────────────────────
    if predictions is not None:
        st.subheader("📉 Predictions vs Actual")
        pred_cols = [c for c in predictions.columns if c.startswith("pred_")]
        selected_models = st.multiselect("Show models:", pred_cols, default=pred_cols[:2])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=predictions["date"], y=predictions["actual"],
            mode="lines", name="Actual",
            line=dict(color="white", width=2),
        ))
        colors = ["#7c3aed", "#f97316", "#06b6d4", "#10b981", "#f43f5e"]
        for i, col in enumerate(selected_models):
            fig.add_trace(go.Scatter(
                x=predictions["date"], y=predictions[col],
                mode="lines", name=col.replace("pred_", "").upper(),
                line=dict(color=colors[i % len(colors)], width=1.5, dash="dot"),
            ))
        fig.update_layout(
            template="plotly_dark", height=420,
            legend=dict(orientation="h", y=1.05),
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: FORECAST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Forecast":
    st.title("🔮 Interactive Forecast Explorer")
    st.info("Quick forecast using LightGBM (fastest model). Run `train.py` for full model comparison.")

    n_ahead = st.slider("Forecast horizon (days)", 7, 90, 30)
    target  = st.selectbox("Target", ["steps", "heart_rate", "sleep_hours", "calories"])

    if st.button("▶️  Run Forecast", type="primary"):
        with st.spinner("Training LightGBM on full dataset..."):
            from src.models.xgboost_model import LightGBMForecaster

            full_df = get_data()
            full_df = make_lag_features(full_df, target, lags=[1, 2, 3, 7, 14])
            full_df.dropna(inplace=True)

            fc = LightGBMForecaster()
            fc.fit(full_df, target=target)

            # Build future dataframe row by row
            last_row = full_df.iloc[-1:].copy()
            future_rows = []
            history = full_df[target].tolist()

            for i in range(n_ahead):
                future_date = full_df.index[-1] + pd.Timedelta(days=i + 1)
                row = pd.DataFrame(index=[future_date])
                row["day_of_week"]  = future_date.dayofweek
                row["month"]        = future_date.month
                row["is_weekend"]   = int(future_date.dayofweek >= 5)
                row["dow_sin"]      = np.sin(2 * np.pi * row["day_of_week"] / 7)
                row["dow_cos"]      = np.cos(2 * np.pi * row["day_of_week"] / 7)
                row["month_sin"]    = np.sin(2 * np.pi * row["month"] / 12)
                row["month_cos"]    = np.cos(2 * np.pi * row["month"] / 12)

                for lag in [1, 2, 3, 7, 14]:
                    row[f"{target}_lag{lag}"] = history[-lag] if len(history) >= lag else np.nan

                h = pd.Series(history)
                row[f"{target}_roll7_mean"]  = h.tail(7).mean()
                row[f"{target}_roll7_std"]   = h.tail(7).std()
                row[f"{target}_roll14_mean"] = h.tail(14).mean()
                row[f"{target}_ewm7"]        = h.ewm(span=7).mean().iloc[-1]

                # Other features: carry forward last known values
                for extra in ["heart_rate", "sleep_hours", "active_minutes", "calories"]:
                    if extra != target and extra in full_df.columns:
                        row[extra] = full_df[extra].iloc[-1]

                pred = fc.predict(row)[0]
                pred = max(pred, 0)
                history.append(pred)
                row[target] = pred
                future_rows.append(row)

        future_df = pd.concat(future_rows)

        # ── Plot ──────────────────────────────────────────────────────────────
        hist_display = full_df[target].tail(60)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_display.index, y=hist_display.values,
            mode="lines", name="Historical",
            line=dict(color="white", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=future_df.index, y=future_df[target].values,
            mode="lines+markers", name=f"Forecast ({n_ahead}d)",
            line=dict(color="#7c3aed", width=2.5, dash="dot"),
            marker=dict(size=5),
        ))
        # Shade forecast region
        fig.add_vrect(
            x0=future_df.index[0], x1=future_df.index[-1],
            fillcolor="#7c3aed", opacity=0.05,
            line_width=0,
        )
        fig.update_layout(
            template="plotly_dark", height=440,
            title=f"{target.replace('_',' ').title()} — {n_ahead}-Day Forecast",
            margin=dict(l=0, r=0, t=50, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Summary stats ─────────────────────────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast Mean",  f"{future_df[target].mean():.1f}")
        c2.metric("Forecast Min",   f"{future_df[target].min():.1f}")
        c3.metric("Forecast Max",   f"{future_df[target].max():.1f}")

        st.dataframe(
            future_df[[target]].rename(columns={target: f"Predicted {target}"}).round(1),
            use_container_width=True,
        )
