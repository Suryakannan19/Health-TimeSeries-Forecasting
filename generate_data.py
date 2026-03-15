"""
Synthetic Health Data Generator
Produces realistic wearable sensor data: steps, heart rate, sleep, calories.
Mirrors patterns seen in Fitbit/Apple Health exports.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def generate_health_data(n_days: int = 730, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic but realistic daily health metrics for n_days.

    Patterns modelled:
    - Weekly seasonality (lower steps on weekends for desk workers)
    - Annual seasonality (more steps in summer)
    - Correlation between steps, calories, and heart rate
    - Realistic noise and occasional missing values
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2022-01-01", periods=n_days, freq="D")

    t = np.arange(n_days)
    day_of_week = dates.dayofweek  # 0=Mon, 6=Sun

    # ── Steps ──────────────────────────────────────────────────────────────────
    # Base: ~8000 steps/day
    weekly_pattern = np.where(day_of_week >= 5, -1500, 500)          # weekend dip
    annual_wave    = 1000 * np.sin(2 * np.pi * t / 365 - np.pi / 2) # summer peak
    trend          = 0.5 * t                                          # gradual improvement
    noise_steps    = np.random.normal(0, 800, n_days)
    steps = np.clip(8000 + weekly_pattern + annual_wave + trend + noise_steps, 1000, 25000).astype(int)

    # ── Resting Heart Rate (bpm) ───────────────────────────────────────────────
    # Inversely correlated with activity; fitness improves over time
    fitness_trend  = -0.005 * t
    hr_noise       = np.random.normal(0, 2, n_days)
    steps_effect   = -0.0003 * (steps - 8000)
    heart_rate = np.clip(68 + fitness_trend + steps_effect + hr_noise, 45, 100).round(1)

    # ── Sleep (hours) ──────────────────────────────────────────────────────────
    sleep_base     = 7.0
    sleep_weekend  = np.where(day_of_week >= 5, 0.8, 0.0)
    sleep_noise    = np.random.normal(0, 0.5, n_days)
    sleep_hours    = np.clip(sleep_base + sleep_weekend + sleep_noise, 3, 11).round(2)

    # ── Calories Burned ────────────────────────────────────────────────────────
    calories = np.clip(1800 + 0.06 * steps + np.random.normal(0, 100, n_days), 1200, 4000).astype(int)

    # ── Active Minutes ─────────────────────────────────────────────────────────
    active_minutes = np.clip((steps / 100) + np.random.normal(0, 10, n_days), 0, 180).astype(int)

    # ── SpO2 (%) ───────────────────────────────────────────────────────────────
    spo2 = np.clip(np.random.normal(97.5, 0.8, n_days), 92, 100).round(1)

    df = pd.DataFrame({
        "date":           dates,
        "steps":          steps,
        "heart_rate":     heart_rate,
        "sleep_hours":    sleep_hours,
        "calories":       calories,
        "active_minutes": active_minutes,
        "spo2":           spo2,
    })

    # Inject ~2% missing values (realistic for wearable sync gaps)
    for col in ["heart_rate", "sleep_hours", "spo2"]:
        mask = np.random.random(n_days) < 0.02
        df.loc[mask, col] = np.nan

    df.set_index("date", inplace=True)
    return df


if __name__ == "__main__":
    out_path = Path(__file__).parent / "raw" / "health_data.csv"
    out_path.parent.mkdir(exist_ok=True)

    df = generate_health_data(n_days=730)
    df.to_csv(out_path)
    print(f"✅ Generated {len(df)} days of health data → {out_path}")
    print(df.describe().round(2))
