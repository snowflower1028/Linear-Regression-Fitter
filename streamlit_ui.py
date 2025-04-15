import streamlit as st
import plotly.graph_objects as go
from datetime import timedelta
import numpy as np
import pandas as pd

def display_best_fits_plotly(time_seconds, y_values, best_results, label, cutoff_idx=None, num_best=3):
    """
    Streamlit-friendly interactive visualization using Plotly.
    :param time_seconds: Time series data (pandas Series).
    :param y_values: Corresponding values (pandas Series).
    :param best_results: List of best results with their R^2, slope, and intercept.
    :param label: Label for the plot title.
    :param cutoff_idx: Index of the cutoff point for saturation detection.
    :param num_best: Number of best fits to show.
    """
    x_all = time_seconds.to_numpy()
    y_all = y_values.to_numpy()
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    x_line = np.linspace(x_min, x_max, 200)

    colors = ['skyblue', 'lightgreen', 'salmon', 'gold', 'mediumpurple']
    labels = [f"Best {i+1}" for i in range(num_best)]

    fig = go.Figure()

    # All data
    fig.add_trace(go.Scatter(
        x=x_all,
        y=y_all,
        mode="markers",
        name="All Data",
        marker=dict(color='lightgray', size=6),
        hoverinfo="x+y"
    ))

    # Best fits
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

        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="markers",
            name=f"{labels[i]} Region ({range_txt})",
            marker=dict(color=colors[i], size=8, line=dict(width=1, color="black"))
        ))

        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"{labels[i]} Fit: {slope_expr}, R²={r2}",
            line=dict(color=colors[i], width=2)
        ))

    # Saturation cutoff
    if cutoff_idx is not None and 0 <= cutoff_idx < len(time_seconds):
        cutoff_time = time_seconds.iloc[cutoff_idx]
        fig.add_vline(
            x=cutoff_time,
            line=dict(color='purple', dash='dash', width=2),
            annotation_text=f"Saturation @ {str(timedelta(seconds=int(cutoff_time)))}",
            annotation_position="top left"
        )

    fig.update_layout(
        title=f"{label} - Best {num_best} Fit(s)",
        xaxis_title="Time (sec)",
        yaxis_title="Value",
        legend_title="Legend",
        template="plotly_white",
        height=500
    )

    return fig

def run_carousel_ui(df, time_seconds, top_results_dict, num_best=3):
    """
    Display interactive plot viewer using precomputed best-fit results.
    :param df: DataFrame with measurement data
    :param time_seconds: Converted time column
    :param top_results_dict: Dictionary {column: (top_results, cutoff_idx)}
    :param num_best: Number of fits to display
    """
    if not top_results_dict:
        st.warning("❗ 분석 결과가 없습니다.")
        return

    selected_col = st.selectbox("🔄 Select column to visualize", options=list(top_results_dict.keys()))

    if selected_col not in top_results_dict or top_results_dict[selected_col] is None:
        st.warning("❗ 선택한 열에 대한 결과가 없습니다.")
        return

    y_values = pd.to_numeric(df[selected_col], errors="coerce")
    top_results, cutoff_idx = top_results_dict[selected_col]

    fig = display_best_fits_plotly(
        time_seconds,
        y_values,
        best_results=top_results,
        label=selected_col,
        cutoff_idx=cutoff_idx,
        num_best=num_best
    )
    st.plotly_chart(fig, use_container_width=True)
