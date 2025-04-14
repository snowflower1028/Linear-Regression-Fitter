# analyzer.py
# This script analyzes time series data to find linear regions and visualize the results.
# It includes functions to convert time formats, detect linear regions, and plot the results.

import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def time_to_seconds_from_any(val, base=None):
    try:
        if isinstance(val, (datetime, pd.Timestamp)):
            seconds = val.hour * 3600 + val.minute * 60 + val.second
        elif isinstance(val, timedelta):
            seconds = val.total_seconds()
        elif isinstance(val, str):
            t = datetime.strptime(val, "%H:%M:%S")
            seconds = t.hour * 3600 + t.minute * 60 + t.second
        elif isinstance(val, (float, int)):
            seconds = val * 86400
        elif hasattr(val, 'hour') and hasattr(val, 'minute'):
            seconds = val.hour * 3600 + val.minute * 60 + val.second
        else:
            return np.nan
        return seconds - base if base is not None else seconds
    except Exception:
        return np.nan


def convert_time_column(time_series):
    base_seconds = time_to_seconds_from_any(time_series.iloc[0])
    return time_series.apply(lambda x: time_to_seconds_from_any(x, base=base_seconds))


def detect_linear_region(x, y, window=5, slope_threshold=0.05):
    slopes = []
    for i in range(len(x) - window):
        xi = x[i:i + window].reshape(-1, 1)
        yi = y[i:i + window]
        if np.any(np.isnan(xi)) or np.any(np.isnan(yi)):
            slopes.append(np.nan)
            continue
        model = LinearRegression().fit(xi, yi)
        slopes.append(model.coef_[0])
    for i, s in enumerate(slopes):
        if abs(s) < slope_threshold:
            return i
    return len(x)


def analyze_column(time_seconds, y_values, top_n=3):
    results = []
    for i in range(len(time_seconds)):
        for j in range(i + 10, len(time_seconds)):
            x = time_seconds.iloc[i:j].to_numpy().reshape(-1, 1)
            y = y_values.iloc[i:j]
            if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                continue
            model = LinearRegression().fit(x, y)
            r2 = r2_score(y, model.predict(x))
            slope_per_min = model.coef_[0] * 60
            intercept = model.intercept_
            results.append({
                "r2": r2,
                "slope": slope_per_min,
                "intercept": intercept,
                "start_idx": i,
                "end_idx": j - 1
            })

    results = sorted(results, key=lambda x: -x["r2"])[:top_n]
    top_results = []
    for res in results:
        t_start = str(timedelta(seconds=int(time_seconds.iloc[res["start_idx"]])))
        t_end = str(timedelta(seconds=int(time_seconds.iloc[res["end_idx"]])))
        slope_expr = f"{int(res['slope'])}x+{int(res['intercept'])}"
        top_results.append([f"{t_start}-{t_end}", round(res["r2"], 4), slope_expr, res])

    while len(top_results) < top_n:
        top_results.append(["N/A", "N/A", "N/A", None])

    return top_results


def analyze_column_with_saturation_cutoff(time_seconds, y_values, top_n=3, min_points=10, max_points=15, slope_threshold=0.05):
    x = time_seconds.reset_index(drop=True)
    y = y_values.reset_index(drop=True)
    cutoff = detect_linear_region(x.to_numpy(), y.to_numpy(), window=5, slope_threshold=slope_threshold)
    x = x[:cutoff]
    y = y[:cutoff]
    n = len(x)
    max_points = max_points or n
    results = []

    for window_size in range(min_points, max_points + 1):
        for start in range(n - window_size + 1):
            end = start + window_size
            x_win = x.iloc[start:end].to_numpy().reshape(-1, 1)
            y_win = y.iloc[start:end]
            if np.any(np.isnan(x_win)) or np.any(np.isnan(y_win)):
                continue
            model = LinearRegression().fit(x_win, y_win)
            r2 = r2_score(y_win, model.predict(x_win))
            slope = model.coef_[0] * 60
            intercept = model.intercept_
            results.append({
                "r2": r2,
                "slope": slope,
                "intercept": intercept,
                "start_idx": start,
                "end_idx": end - 1
            })

    results = sorted(results, key=lambda r: -r["r2"])[:top_n]
    top_results = []
    for res in results:
        t_start = str(timedelta(seconds=int(x.iloc[res["start_idx"]])))
        t_end = str(timedelta(seconds=int(x.iloc[res["end_idx"]])))
        slope_expr = f"{int(res['slope'])}x+{int(res['intercept'])}"
        top_results.append([f"{t_start}-{t_end}", round(res["r2"], 4), slope_expr, res])

    while len(top_results) < top_n:
        top_results.append(["N/A", "N/A", "N/A", None])

    return top_results, cutoff


def visualize_best_fits(time_seconds, y_values, best_results, label, out_dir="plots", cutoff_idx=None, num_best=3):
    os.makedirs(out_dir, exist_ok=True)

    x_all = time_seconds.to_numpy()
    y_all = y_values.to_numpy()
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    x_line = np.linspace(x_min, x_max, 200)

    colors = ['red', 'blue', 'green']
    labels = ['Best 1', 'Best 2', 'Best 3']

    plt.figure(figsize=(10, 6))
    plt.scatter(x_all, y_all, c='lightgray', s=10, label="All Data", zorder=0)

    for i in range(min(num_best, len(best_results))):
        range_txt, r2, slope_expr, res = best_results[i]
        if res is None:
            continue
        i_start = res["start_idx"]
        i_end = res["end_idx"]
        x_fit = time_seconds.iloc[i_start:i_end + 1].to_numpy()
        y_fit = y_values.iloc[i_start:i_end + 1].to_numpy()
        slope = res["slope"]
        intercept = res["intercept"]
        y_line = (slope / 60) * x_line + intercept

        plt.scatter(x_fit, y_fit, color=colors[i], s=40, edgecolors='black', label=f"{labels[i]} Region ({range_txt})", zorder=2 + i)
        plt.plot(x_line, y_line, color=colors[i], linewidth=2, label=f"{labels[i]} Fit: {slope_expr}, R²={r2}", zorder=3 + i)

    if cutoff_idx is not None and 0 <= cutoff_idx < len(time_seconds):
        cutoff_time = time_seconds.iloc[cutoff_idx]
        plt.axvline(x=cutoff_time, color='purple', linestyle='--', linewidth=2, label=f"Saturation Cutoff @ {str(timedelta(seconds=int(cutoff_time)))}")

    plt.xlabel("Time (sec)")
    plt.ylabel("Value")
    plt.title(f"{label} - Best {num_best} Fit(s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label}_best{num_best}_fits.png"))
    plt.close()
