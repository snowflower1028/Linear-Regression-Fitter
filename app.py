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
from streamlit_ui import run_carousel_ui

st.set_page_config(page_title="Linear Fit Analyzer", layout="wide")
st.title("📈 Linear Regression Analyzer with Saturation Detection")
st.markdown(
    """
    This app analyzes time series data to find linear regions and visualize the results.
    
    `version 1.1.0 (2025-04-14, Minsoo Lee, Seoul National University, College of Pharmacy, WLab(Prof. Wooin Lee))`

    - Upload an Excel file with time series data.    
    """
)

# Initialize session state variables
for key in ["analysis_done", "result_df", "plot_paths", "img_idx", "df", "time_seconds", "top_results_dict"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_done" else False

uploaded_file = st.file_uploader("📂 Upload Excel File (.xlsx)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, header=2).dropna(axis=1, how='all')
        time_column = df.iloc[:, 0]
        time_seconds = convert_time_column(time_column)
        start_col_idx = df.columns.get_loc("A1")
        data_columns = df.columns[start_col_idx:]
        total_columns = len(data_columns)

        st.session_state["df"] = df
        st.session_state["time_seconds"] = time_seconds

        # 파일 인식 성공시 다음으로 넘어감. 실패시 Exception으로 넘어감 당연함.
        st.success(f"✅ File loaded. Found {total_columns} data columns for analysis.")

        analysis_mode = st.radio("Select Analysis Type", ["range", "saturation"])
        num_best = st.slider("Number of Best Fits to Display", min_value=1, max_value=5, value=3)
        selected_columns = st.multiselect(
            "🔢 Select columns to analyze",
            options=list(data_columns),
            default=list(data_columns)
        )
        min_segment_ratio = st.slider("📏 Minimum segment length (% of total points)", min_value=5, max_value=100, value=30, step=1)

        # Push the button to start analysis
        # if로 안 해놨더니 그냥 막 돌아가버림.
        if st.button("🚀 Run Analysis"):
            result_dict = {}
            plot_paths = []
            top_results_dict = {}
            progress_bar = st.progress(0)

            for i, col in enumerate(selected_columns):
                y_values = pd.to_numeric(df[col], errors='coerce')
                total_points = len(time_seconds)
                min_points = max(2, int(total_points * (min_segment_ratio / 100)))

                if analysis_mode == "range":
                    top_results = analyze_column(time_seconds, y_values, min_points, top_n=num_best)
                    cutoff_idx = None
                else:
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

                top_results_dict[col] = (top_results, cutoff_idx)
                progress_bar.progress((i + 1) / len(selected_columns))

            row_labels = []
            for i in range(1, num_best + 1):
                row_labels += [f"Best {i} 시간", f"Best {i} r²", f"Best {i} slope"]

            result_df = pd.DataFrame(result_dict, index=row_labels)
            clean_result_df = result_df.astype(str)

            st.session_state["result_df"] = clean_result_df
            st.session_state["plot_paths"] = plot_paths
            st.session_state["top_results_dict"] = top_results_dict
            st.session_state["analysis_done"] = True
            st.success("✅ Analysis complete.")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# 결과 인터페이스 버튼 눌렀을 때 초기화 되는 것을 방지하기 위해 세션 상태확인
if st.session_state["analysis_done"]:
    result_df = st.session_state["result_df"]
    plot_paths = st.session_state["plot_paths"]
    df = st.session_state["df"]
    time_seconds = st.session_state["time_seconds"]
    top_results_dict = st.session_state["top_results_dict"]

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

    # Plotly를 사용한 interactive chart를 표시. stream_ui.py에서 실행됨
    st.markdown("### 🖼 Interactive Plot Viewer")
    run_carousel_ui(df, time_seconds, top_results_dict, num_best=num_best)
