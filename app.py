# app.py
import os
import base64
import zipfile
from datetime import datetime
import streamlit as st
import pandas as pd
from io import BytesIO
from analyzer import (
    convert_time_column,
    analyze_column,
    analyze_column_with_saturation_cutoff,
    visualize_best_fits
)

st.set_page_config(page_title="Linear Fit Analyzer", layout="wide")
st.title("📈 Linear Regression Analyzer with Saturation Detection")
st.markdown(
    """
    This app analyzes time series data to find linear regions and visualize the results.
    `version 0.1.1 (2025-04-14, WLab Minsoo Lee)`
    - Upload an Excel file with time series data.    
    """
)

# Initialize session state variables
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "result_df" not in st.session_state:
    st.session_state["result_df"] = None
if "plot_paths" not in st.session_state:
    st.session_state["plot_paths"] = []
if "img_idx" not in st.session_state:
    st.session_state["img_idx"] = 0

uploaded_file = st.file_uploader("📂 Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, header=2).dropna(axis=1, how='all')
        time_column = df.iloc[:, 0]
        time_seconds = convert_time_column(time_column)
        start_col_idx = df.columns.get_loc("A1")
        data_columns = df.columns[start_col_idx:]
        total_columns = len(data_columns)

        st.success(f"✅ File loaded. Found {total_columns} data columns for analysis.")

        analysis_mode = st.radio("Select Analysis Type", ["General Sliding Window", "Detect Saturation"])
        num_best = st.slider("🔢 Number of Best Fits to Display", min_value=1, max_value=5, value=3)
        min_segment_ratio = st.slider("📏 Minimum segment length (% of total points)", min_value=5, max_value=100, value=30, step=1)
        selected_columns = st.multiselect(
            "🔢 Select columns to analyze",
            options=list(data_columns),
            default=list(data_columns)
        )
        
        if st.button("🚀 Run Analysis"):
            result_dict = {}
            plot_paths = []
            progress_bar = st.progress(0)

            for i, col in enumerate(selected_columns):
                y_values = pd.to_numeric(df[col], errors='coerce')
                total_points = len(time_seconds)
                min_points = max(2, int(total_points * (min_segment_ratio / 100)))

                if analysis_mode == "General Sliding Window":
                    top_results = analyze_column(time_seconds, y_values, top_n=num_best)
                    cutoff_idx = None
                elif analysis_mode == "Detect Saturation":
                    top_results, cutoff_idx = analyze_column_with_saturation_cutoff(
                        time_seconds, y_values,
                        top_n=num_best,
                        min_points=min_points,
                        max_points=None
                    )

                if top_results[0][3]:
                    visualize_best_fits(time_seconds, y_values, top_results, label=col, num_best=num_best, cutoff_idx=cutoff_idx)
                    plot_path = os.path.join("plots", f"{col}_best{num_best}_fits.png")
                    plot_paths.append(plot_path)

                row = []
                for result in top_results:
                    row += [result[0], result[1], result[2]]
                result_dict[col] = row

                progress_bar.progress((i + 1) / len(selected_columns))

            row_labels = []
            for i in range(1, num_best + 1):
                row_labels += [f"Best {i} 시간", f"Best {i} r²", f"Best {i} slope"]

            result_df = pd.DataFrame(result_dict, index=row_labels)
            clean_result_df = result_df.astype(str)

            st.session_state["result_df"] = clean_result_df
            st.session_state["plot_paths"] = plot_paths
            st.session_state["analysis_done"] = True
            st.success("✅ Analysis complete.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# 결과 인터페이스 버튼 눌렀을 때 초기화 되는 것을 방지하기 위해 세션 상태확인인
if st.session_state["analysis_done"]:
    result_df = st.session_state["result_df"]
    plot_paths = st.session_state["plot_paths"]

    st.markdown("### 📋 Summary of Best Fits")
    st.dataframe(result_df, use_container_width=True)

    buffer = BytesIO()
    result_df.to_excel(buffer, index=True)
    st.download_button(
        label="📥 Download Excel Summary",
        data=buffer.getvalue(),
        file_name=f"best_fit_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if plot_paths:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for path in plot_paths:
                if os.path.exists(path):
                    zf.write(path, arcname=os.path.basename(path))

        st.download_button(
            label="📦 Download All Plots as ZIP",
            data=zip_buffer.getvalue(),
            file_name=f"plots_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip",
            mime="application/zip"
        )

        st.markdown("### 🖼 Preview Plots One-by-One")

        col1, col2, col3 = st.columns([1, 6, 1])

        with col1:
            if st.button("◀️ Prev") and st.session_state["img_idx"] > 0:
                st.session_state["img_idx"] -= 1

        with col3:
            if st.button("Next ▶️") and st.session_state["img_idx"] < len(plot_paths) - 1:
                st.session_state["img_idx"] += 1

        idx = st.session_state["img_idx"]
        current_img_path = plot_paths[idx]

        with open(current_img_path, "rb") as f:
            img_data = f.read()
            b64_encoded = base64.b64encode(img_data).decode("utf-8")
            st.markdown(
                f'<img src="data:image/png;base64,{b64_encoded}" style="width: 75%; display: block; margin: 0 auto;"/>',
                unsafe_allow_html=True
            )
        st.caption(f"{os.path.basename(current_img_path)} ({idx+1}/{len(plot_paths)})")
