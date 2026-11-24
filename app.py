import hashlib
import os
import zipfile
from io import BytesIO
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from analyzer import (
    TIME_UNITS, read_measurements, convert_time_column, summarize_results,
    analyze_column_prefix, analyze_column_with_saturation_cutoff,
    visualize_best_fits,
)
from streamlit_ui import run_carousel_ui

st.set_page_config(page_title="Linear Fit Analyzer", layout="wide")
st.title("📈 Linear Regression Analyzer with Saturation Detection")
st.caption("Version 1.3.1 · Minsoo Lee · Seoul National University College of Pharmacy")
st.write("Find linear intervals in time-series measurements and export the fitted coefficients.")

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])
header_row = st.number_input("Header row", min_value=1, value=3, step=1,
                             help="1 means the first spreadsheet row. The first column contains time.")
unit_label = st.selectbox("Numeric time unit", list(TIME_UNITS),
                         help="Applies to numbers only. Text durations use MM:SS or HH:MM:SS.")
st.caption("Text 1:30 means 1 minute 30 seconds. Slopes are reported per minute; plot time is in seconds.")

if uploaded_file is None:
    st.session_state.analysis_done = False
else:
    try:
        df = read_measurements(uploaded_file, header_row=int(header_row))
        time_seconds = convert_time_column(df.iloc[:, 0], numeric_unit=TIME_UNITS[unit_label])
        data_columns = list(df.columns[1:])
        st.success(f"Loaded {len(df)} timepoints and {len(data_columns)} measurement columns.")
        analysis_mode = st.radio("Select Analysis Type", ["range", "saturation"])
        num_best = st.slider("Number of Best Fits to Display", 1, 5, 3)
        selected_columns = st.multiselect("Columns to analyze", data_columns, default=data_columns)
        min_ratio = st.slider("Minimum segment length (% of total points)", 5, 100, 30)
        signature = (hashlib.sha256(uploaded_file.getvalue()).hexdigest(), int(header_row),
                     unit_label, analysis_mode, num_best, tuple(selected_columns), min_ratio)
        if st.session_state.get("result_signature") != signature:
            st.session_state.analysis_done = False

        if st.button("Run Analysis"):
            if not selected_columns:
                st.warning("Select at least one measurement column.")
            else:
                top_results_dict, results, plot_paths = {}, {}, []
                progress = st.progress(0)
                min_points = max(2, int(len(time_seconds) * min_ratio / 100))
                for i, col in enumerate(selected_columns):
                    y_values = pd.to_numeric(df[col], errors="coerce")
                    missing = int((~np.isfinite(y_values.to_numpy(dtype=float))).sum())
                    if missing:
                        st.warning(f"{col}: {missing} missing/non-finite measurements. "
                                   "Intervals containing them are excluded; later valid intervals are still searched.")
                    if analysis_mode == "range":
                        top_results = analyze_column_prefix(time_seconds, y_values, min_points, top_n=num_best)
                        cutoff_idx = None
                    else:
                        top_results, cutoff_idx = analyze_column_with_saturation_cutoff(
                            time_seconds, y_values, top_n=num_best, min_points=min_points, max_points=None)
                    if top_results[0][3] is not None:
                        visualize_best_fits(time_seconds, y_values, top_results, label=str(col),
                                            num_best=num_best, cutoff_idx=cutoff_idx)
                        plot_paths.append(os.path.join("plots", f"{col}_best{num_best}_fits.png"))
                    else:
                        st.warning(f"{col}: no valid interval meets the selected minimum length.")
                    results[col] = top_results
                    top_results_dict[col] = (top_results, cutoff_idx)
                    progress.progress((i + 1) / len(selected_columns))
                st.session_state.update(
                    result_df=summarize_results(results, num_best),
                    top_results_dict=top_results_dict, plot_paths=plot_paths,
                    df=df, time_seconds=time_seconds, num_best=num_best,
                    result_signature=signature, analysis_done=True,
                )
                st.success("Analysis complete.")
    except (ValueError, TypeError, KeyError) as exc:
        st.session_state.analysis_done = False
        st.error(str(exc))

if st.session_state.analysis_done:
    result_df = st.session_state.result_df
    st.subheader("Summary of Best Fits")
    st.dataframe(result_df.astype(str), use_container_width=True)
    st.caption("Equation uses t_min (elapsed minutes). Numeric slope and intercept rows retain full precision in Excel.")
    buffer = BytesIO()
    result_df.to_excel(buffer, index=True)
    st.download_button("Download Excel Summary", buffer.getvalue(),
                       file_name=f"best_fit_summary_{datetime.now():%Y-%m-%d_%H-%M-%S}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if st.session_state.plot_paths:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as archive:
            for path in st.session_state.plot_paths:
                if os.path.exists(path):
                    archive.write(path, arcname=os.path.basename(path))
        st.download_button("Download All Plots as ZIP", zip_buffer.getvalue(),
                           file_name=f"plots_{datetime.now():%Y-%m-%d_%H-%M-%S}.zip", mime="application/zip")
    st.subheader("Interactive Plot Viewer")
    run_carousel_ui(st.session_state.df, st.session_state.time_seconds,
                    st.session_state.top_results_dict, num_best=st.session_state.num_best)
