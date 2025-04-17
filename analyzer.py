# analyzer.py
# This script analyzes time series data to find linear regions and visualize the results.
# It includes functions to convert time formats, detect linear regions, and plot the results.

import os
from datetime import datetime, timedelta
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def time_to_seconds_from_any(val, base=None) -> float:
    """
    엑셀에서 시간을 자꾸 이상한걸로 바꿔서 다시 바꿔주는 함수.
    Supports datetime, timedelta, string (HH:MM:SS), float (days), and int (seconds).
    :param val: The time value to convert.
    :param base: Optional base time in seconds for relative conversion. (default: None))
    :return: Time in seconds.
    """
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
    """
    Convert a time series column to seconds.
    처음엔 그냥 바꾸면 되는 줄 알았음.
    :param time_series: A pandas Series containing time values.
    :return: A pandas Series with time values converted to seconds.
    """
    base_seconds = time_to_seconds_from_any(time_series.iloc[0])
    return time_series.apply(lambda x: time_to_seconds_from_any(x, base=base_seconds))


def detect_linear_region(x, y, window=5, slope_threshold=0.05) -> int:
    """
    포화 구간 탐지하는 함수.
    일정 구간을 잡고 그 구간에서 선형 회귀를 돌려서 기울기를 구함.
    기울기가 threshold보다 작아지면 그 지점부터 포화 구간으로 판단.
    :param x: Time series data (numpy array).
    :param y: Corresponding values (numpy array).
    :param window: Size of the sliding window for linear regression.
    :param slope_threshold: Slope threshold for detecting linear regions.
    :return: The index of the first point where the slope is below the threshold.
    """
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


def analyze_column(time_seconds, y_values, min_points, top_n=3) -> List[dict]:
    """
    기본 슬라이딩 윈도우 분석.
    :param time_seconds: Time series data (pandas Series).
    :param y_values: Corresponding values (pandas Series).
    :param top_n: Number of top results to return.
    :return: List of top results with their R^2, slope, and intercept.
    """
    results = []
    for i in range(len(time_seconds)):
        for j in range(i + min_points, len(time_seconds)):
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


def _compute_segment(args):
    x_win, y_win, start_idx, end_idx = args
    # 여기로는 이미 NaN이 없는 세그먼트만 들어온다.
    model = LinearRegression().fit(x_win, y_win)
    r2 = r2_score(y_win, model.predict(x_win))
    slope = model.coef_[0] * 60
    intercept = model.intercept_
    return {
        "r2": r2,
        "slope": slope,
        "intercept": intercept,
        "start_idx": start_idx,
        "end_idx": end_idx
    }

def analyze_column_optimized(time_seconds: pd.Series,
                             y_values: pd.Series,
                             min_points: int,
                             top_n: int = 3):
    """
    NaN을 제거하고 ThreadPoolExecutor로 병렬화한 슬라이딩 윈도우 분석.
    :return: [["t0-t1", r2, "slopex+intercept", 원본 dict], ...]
    """
    n = len(time_seconds)
    tasks = []

    # 1) (x_win, y_win, start, end) 에 NaN 검사 추가
    for window_size in range(min_points, n + 1):
        for start in range(0, n - window_size + 1):
            end = start + window_size
            x_win = time_seconds.iloc[start:end].values.reshape(-1, 1)
            y_win = y_values.iloc[start:end].values

            # NaN 포함된 세그먼트는 건너뛴다
            if np.isnan(x_win).any() or np.isnan(y_win).any():
                continue

            tasks.append((x_win, y_win, start, end - 1))

    # 2) 병렬 실행 (한 번만 ThreadPoolExecutor 생성)
    results = []
    with ThreadPoolExecutor() as executor:
        for res in executor.map(_compute_segment, tasks):
            results.append(res)

    # 3) r2 기준 상위 top_n 결과 선택
    top_results = sorted(results, key=lambda x: -x["r2"])[:top_n]

    # 4) 기존 formatter와 동일하게 문자열화
    formatted = []
    for res in top_results:
        t0 = int(time_seconds.iloc[res["start_idx"]])
        t1 = int(time_seconds.iloc[res["end_idx"]])
        range_txt = f"{str(timedelta(seconds=t0))}-{str(timedelta(seconds=t1))}"
        slope_expr = f"{int(res['slope'])}x+{int(res['intercept'])}"
        formatted.append([range_txt, round(res["r2"], 4), slope_expr, res])

    # 5) 모자란 슬롯은 N/A로 채우기
    while len(formatted) < top_n:
        formatted.append(["N/A", "N/A", "N/A", None])

    return formatted

def calculate_r2(x, y):
    model = LinearRegression().fit(x, y)
    return {
        "r2": r2_score(y, model.predict(x)),
        "slope": model.coef_[0] * 60,
        "intercept": model.intercept_,
        "start_idx": x[0][0],
        "end_idx": x[-1][0]
    }


def analyze_column_prefix(time_seconds: pd.Series,
                          y_values: pd.Series,
                          min_points: int,
                          top_n: int = 3):
    # 1) 누적합 계산 (numpy array가 더 빠릅니다)
    x = time_seconds.to_numpy()
    y = y_values.to_numpy()
    Sx  = np.cumsum(x)
    Sy  = np.cumsum(y)
    Sxx = np.cumsum(x * x)
    Syy = np.cumsum(y * y)
    Sxy = np.cumsum(x * y)

    def window_stats(i, j):
        """i..j 구간에 대한 회귀계수와 R² 계산 (O(1))."""
        L = j - i + 1
        sum_x  = Sx[j]  - (Sx[i-1]  if i>0 else 0)
        sum_y  = Sy[j]  - (Sy[i-1]  if i>0 else 0)
        sum_xx = Sxx[j] - (Sxx[i-1] if i>0 else 0)
        sum_yy = Syy[j] - (Syy[i-1] if i>0 else 0)
        sum_xy = Sxy[j] - (Sxy[i-1] if i>0 else 0)

        denom = L*sum_xx - sum_x*sum_x
        if denom == 0:
            return None  # 직선이 불가능한 경우
        m = (L*sum_xy - sum_x*sum_y) / denom
        b = (sum_y - m*sum_x) / L

        # R² 계산
        num = (L*sum_xy - sum_x*sum_y)**2
        den = denom * (L*sum_yy - sum_y*sum_y)
        r2 = num/den if den>0 else 0

        return {"r2": r2, "slope": m*60, "intercept": b,
                "start_idx": i, "end_idx": j}

    # 2) 모든 구간에 대해 O(n²) 반복
    #    top_n 유지 위해 min-heap 사용 (크기 top_n)
    heap = []
    n = len(x)
    for i in range(n):
        # j 최소 i+min_points-1 부터 시작
        for j in range(i + min_points - 1, n):
            stats = window_stats(i, j)
            if not stats: 
                continue
            # heap 에 (r2, stats)로 pushpop
            if len(heap) < top_n:
                heapq.heappush(heap, (stats["r2"], stats))
            else:
                heapq.heappushpop(heap, (stats["r2"], stats))

    # 3) 힙에서 상위 N개 뽑아서 정렬
    top = sorted(heap, key=lambda x: -x[0])
    formatted = []
    for r2, res in top:
        t0 = str(timedelta(seconds=int(x[res["start_idx"]])))
        t1 = str(timedelta(seconds=int(x[res["end_idx"]])))
        formatted.append([f"{t0}-{t1}", round(r2,4),
                          f"{int(res['slope'])}x+{int(res['intercept'])}", res])

    # 부족분은 N/A로 채움
    while len(formatted) < top_n:
        formatted.append(["N/A","N/A","N/A", None])

    return formatted


def analyze_column_with_saturation_cutoff(time_seconds, y_values, top_n=3, min_points=10, max_points=15, slope_threshold=0.05) -> Tuple[List[dict], int]:
    """
    포화 구간 탐지 후 슬라이딩 윈도우 분석.
    :param time_seconds: Time series data (pandas Series).
    :param y_values: Corresponding values (pandas Series).
    :param top_n: Number of top results to return.
    :param min_points: 해당 개수 이상 탐색
    :param max_points: 구간 최대 크기기
    :param slope_threshold: Slope threshold for detecting linear regions.
    :return: List of top results with their R^2, slope, and intercept., cutoff index.
    """
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
    """
    Visualize the best fits for the given time series data.
    :param time_seconds: Time series data (pandas Series).
    :param y_values: Corresponding values (pandas Series).
    :param best_results: List of best results with their R^2, slope, and intercept.
    :param label: Label for the plot title.
    :param out_dir: Output directory for saving the plot.
    :param cutoff_idx: Index of the cutoff point for saturation detection.
    :param num_best: Number of best-fit segments to show (up to 5).
    """
    os.makedirs(out_dir, exist_ok=True)

    x_all = time_seconds.to_numpy()
    y_all = y_values.to_numpy()
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    x_line = np.linspace(x_min, x_max, 200)

    colors = ['skyblue', 'cornflowerblue', 'mediumseagreen', 'orange', 'orchid']
    labels = ['Best 1', 'Best 2', 'Best 3', 'Best 4', 'Best 5']

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

        plt.scatter(
            x_fit, y_fit,
            color=colors[i % len(colors)],
            s=40,
            edgecolors='black',
            label=f"{labels[i % len(labels)]} Region ({range_txt})",
            zorder=2 + i
        )
        plt.plot(
            x_line, y_line,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{labels[i % len(labels)]} Fit: {slope_expr}, R²={r2}",
            zorder=3 + i
        )

    if cutoff_idx is not None and 0 <= cutoff_idx < len(time_seconds):
        cutoff_time = time_seconds.iloc[cutoff_idx]
        plt.axvline(
            x=cutoff_time,
            color='purple',
            linestyle='--',
            linewidth=2,
            label=f"Saturation Cutoff @ {str(timedelta(seconds=int(cutoff_time)))}"
        )

    plt.xlabel("Time (sec)")
    plt.ylabel("Value")
    plt.title(f"{label} - Best {num_best} Fit(s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label}_best{num_best}_fits.png"))
    plt.close()


if __name__ == "__main__":
    import time

    # 1) 현재 폴더에서 첫 번째 Excel 파일 자동 탐색
    folder_path = os.path.dirname(__file__)
    excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    if not excel_files:
        raise FileNotFoundError("No Excel files found in the current folder.")
    uploaded_file = os.path.join(folder_path, excel_files[0])
    print(f"Using file: {uploaded_file}")

    # 2) 데이터 로드
    df = pd.read_excel(uploaded_file, header=2).dropna(axis=1, how='all')
    time_column = df.iloc[:, 0]
    time_seconds = convert_time_column(time_column)
    start_col_idx = df.columns.get_loc("A1")
    data_columns = df.columns[start_col_idx:]

    # 3) 파라미터 설정
    min_points = 10
    top_n = 3

    # 4) 원본 analyze_column 전체 컬럼 루프
    t0 = time.perf_counter()
    results_old = {}
    for col in data_columns:
        y_series = pd.to_numeric(df[col], errors='coerce')
        results_old[col] = analyze_column(time_seconds, y_series, min_points, top_n=top_n)
    dur_old = time.perf_counter() - t0
    print(f"\n--- Original analyze_column on {len(data_columns)} columns ---")
    print(f"Total time: {dur_old:.3f} sec")

    # (옵션) 첫 번째 컬럼 결과 샘플
    first_col = data_columns[0]
    print(f"Sample result for '{first_col}':\n{results_old[first_col]}")

    # 5) 최적화 analyze_column_optimized 전체 컬럼 루프
    t1 = time.perf_counter()
    results_opt = {}
    for col in data_columns:
        y_series = pd.to_numeric(df[col], errors='coerce')
        results_opt[col] = analyze_column_optimized(time_seconds, y_series, min_points, top_n=top_n)
    dur_opt = time.perf_counter() - t1
    print(f"\n--- Optimized analyze_column_optimized on {len(data_columns)} columns ---")
    print(f"Total time: {dur_opt:.3f} sec")

    # (옵션) 첫 번째 컬럼 결과 샘플
    print(f"Sample optimized result for '{first_col}':\n{results_opt[first_col]}")

    # 6) 최적화 analyze_column_prefix 전체 컬럼 루프
    t2 = time.perf_counter()
    results_pre = {}
    for col in data_columns:
        y_series = pd.to_numeric(df[col], errors='coerce')
        results_pre[col] = analyze_column_prefix(time_seconds, y_series, min_points, top_n=top_n)
    dur_pre = time.perf_counter() - t2
    print(f"\n--- Optimized analyze_column_prefix on {len(data_columns)} columns ---")
    print(f"Total time: {dur_pre:.3f} sec")

    # (옵션) 첫 번째 컬럼 결과 샘플
    print(f"Sample optimized result for '{first_col}':\n{results_opt[first_col]}")

    # 6) 속도 비교
    print(f"\n--- Speed Comparison ---")
    print(f"Original analyze_column: {dur_old:.3f} sec")
    print(f"Optimized analyze_column_optimized: {dur_opt:.3f} sec")
    print(f"Optimized analyze_column_prefix: {dur_pre:.3f} sec")
