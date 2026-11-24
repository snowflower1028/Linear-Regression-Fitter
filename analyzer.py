# analyzer.py
# This script analyzes time series data to find linear regions and visualize the results.
# It includes functions to convert time formats, detect linear regions, and plot the results.

import os
from datetime import datetime, timedelta, time as clock_time
from numbers import Real
import re
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


TIME_UNITS = {
    "Excel days (1 = 24 hours)": "excel_days",
    "Seconds": "seconds",
    "Minutes": "minutes",
    "Hours": "hours",
}
_TIME_FACTORS = {"excel_days": 86400.0, "seconds": 1.0,
                 "minutes": 60.0, "hours": 3600.0}


def time_to_seconds_from_any(val, base=None, numeric_unit="excel_days") -> float:
    """Convert a scalar time; invalid/missing values return NaN.

    Numeric values (including numeric strings) use numeric_unit. The default
    preserves existing Excel serial-day inputs. Duration strings are MM:SS or
    HH:MM:SS, with optional fractional seconds and no 24-hour limit.
    """
    if numeric_unit not in _TIME_FACTORS:
        raise ValueError(f"Unsupported numeric time unit: {numeric_unit}")
    try:
        if val is None or pd.isna(val) or isinstance(val, (bool, np.bool_)):
            return np.nan
        if isinstance(val, datetime):
            seconds = (val.toordinal() * 86400.0 + val.hour * 3600.0
                       + val.minute * 60.0 + val.second + val.microsecond / 1e6)
        elif isinstance(val, timedelta):
            seconds = val.total_seconds()
        elif isinstance(val, clock_time):
            seconds = (val.hour * 3600.0 + val.minute * 60.0
                       + val.second + val.microsecond / 1e6)
        elif isinstance(val, str) and ":" in val:
            match = re.fullmatch(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", val.strip())
            if not match:
                return np.nan
            hours, minutes, sec = match.groups()
            if float(sec) >= 60 or (hours is not None and int(minutes) >= 60):
                return np.nan
            seconds = int(hours or 0) * 3600.0 + int(minutes) * 60.0 + float(sec)
        elif isinstance(val, (Real, str)):
            seconds = float(val) * _TIME_FACTORS[numeric_unit]
        else:
            return np.nan
        seconds = seconds - base if base is not None else seconds
        return float(seconds) if np.isfinite(seconds) else np.nan
    except (ValueError, TypeError, OverflowError):
        return np.nan


def convert_time_column(time_series, numeric_unit="excel_days"):
    """Return elapsed seconds, preserving row positions and validating times.

    Measurement gaps are handled by the regression functions. A missing or
    invalid timestamp is rejected so it cannot silently change the time scale.
    """
    if len(time_series) == 0:
        raise ValueError("The time column is empty.")
    converted = time_series.apply(
        lambda val: time_to_seconds_from_any(val, numeric_unit=numeric_unit)
    ).astype(float)
    invalid = np.flatnonzero(~np.isfinite(converted.to_numpy()))
    if len(invalid):
        rows = ", ".join(str(i + 1) for i in invalid[:10])
        raise ValueError(f"Invalid or missing time at data row(s): {rows}. "
                         "Use MM:SS, HH:MM:SS, or numbers in the selected unit.")
    if np.any(np.diff(converted.to_numpy()) < 0):
        raise ValueError("Time values must be in nondecreasing order.")
    return converted - converted.iloc[0]


def read_measurements(source, header_row=3):
    """Read an XLSX with a selectable 1-based header row and time in column 1."""
    if header_row < 1:
        raise ValueError("Header row must be at least 1.")
    frame = pd.read_excel(source, header=header_row - 1).dropna(axis=1, how="all")
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("Provide a time column and at least one measurement column.")
    return frame


def summarize_results(result_dict, num_best):
    """Build the shared UI/XLSX summary without converting coefficients to text."""
    records = {}
    for name, results in result_dict.items():
        values = []
        for region, score, equation, stats in results:
            values.extend([region, stats["r2"] if stats else np.nan, equation,
                           stats["slope"] if stats else np.nan,
                           stats["intercept"] if stats else np.nan])
        records[name] = values
    labels = []
    for i in range(1, num_best + 1):
        labels.extend([f"Best {i} Region", f"Best {i} R²", f"Best {i} Equation (t_min)",
                       f"Best {i} Slope (value/min)", f"Best {i} Intercept"])
    return pd.DataFrame(records, index=labels)


def _rank_segment(stats):
    # On exact R² ties prefer more points, then the earlier interval.
    return (stats["r2"], stats["end_idx"] - stats["start_idx"] + 1,
            -stats["start_idx"], -stats["end_idx"])


def _format_results(time_seconds, results, top_n):
    formatted = []
    for stats in sorted(results, key=_rank_segment, reverse=True)[:top_n]:
        t0 = timedelta(seconds=float(time_seconds.iloc[stats["start_idx"]]))
        t1 = timedelta(seconds=float(time_seconds.iloc[stats["end_idx"]]))
        equation = f"{stats['slope']:.12g} * t_min {stats['intercept']:+.12g}"
        formatted.append([f"{t0}-{t1}", round(stats["r2"], 4), equation, stats])
    return formatted + [["N/A", "N/A", "N/A", None] for _ in range(top_n - len(formatted))]


def _validate_regression_input(time_seconds, y_values, min_points, top_n):
    if len(time_seconds) != len(y_values):
        raise ValueError("Time and measurement columns must have the same length.")
    if min_points < 2 or top_n < 1:
        raise ValueError("min_points must be at least 2 and top_n at least 1.")


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
    for i in range(len(x) - window + 1):
        xi = x[i:i + window].reshape(-1, 1)
        yi = y[i:i + window]
        if not (np.isfinite(xi).all() and np.isfinite(yi).all()):
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
    _validate_regression_input(time_seconds, y_values, min_points, top_n)
    results = []
    for i in range(len(time_seconds)):
        for j in range(i + min_points, len(time_seconds) + 1):
            x = time_seconds.iloc[i:j].to_numpy().reshape(-1, 1)
            y = y_values.iloc[i:j]
            if not (np.isfinite(x).all() and np.isfinite(y).all()) or np.ptp(x) == 0:
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

    return _format_results(time_seconds, results, top_n)


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
    _validate_regression_input(time_seconds, y_values, min_points, top_n)
    n = len(time_seconds)
    tasks = []

    # 1) (x_win, y_win, start, end) 에 NaN 검사 추가
    for window_size in range(min_points, n + 1):
        for start in range(0, n - window_size + 1):
            end = start + window_size
            x_win = time_seconds.iloc[start:end].values.reshape(-1, 1)
            y_win = y_values.iloc[start:end].values

            # NaN 포함된 세그먼트는 건너뛴다
            if not (np.isfinite(x_win).all() and np.isfinite(y_win).all()) or np.ptp(x_win) == 0:
                continue

            tasks.append((x_win, y_win, start, end - 1))

    # 2) 병렬 실행 (한 번만 ThreadPoolExecutor 생성)
    results = []
    with ThreadPoolExecutor() as executor:
        for res in executor.map(_compute_segment, tasks):
            results.append(res)

    return _format_results(time_seconds, results, top_n)


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
    """Rank contiguous finite windows, preserving original row indices.

    Invalid pairs never enter prefix sums. A separate count rejects any window
    containing a gap, so valid windows after that gap remain usable.
    """
    _validate_regression_input(time_seconds, y_values, min_points, top_n)
    x = pd.to_numeric(time_seconds, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return _format_results(time_seconds, [], top_n)
    x_origin, y_origin = x[valid][0], y[valid][0]
    x_safe = np.where(valid, x - x_origin, 0.0)
    y_safe = np.where(valid, y - y_origin, 0.0)
    def prefix(values):
        return np.concatenate(([0.0], np.cumsum(values)))
    sx, sy = prefix(x_safe), prefix(y_safe)
    sxx, syy = prefix(x_safe * x_safe), prefix(y_safe * y_safe)
    sxy, bad = prefix(x_safe * y_safe), prefix(~valid)
    heap = []
    for i in range(len(x)):
        for j in range(i + min_points, len(x) + 1):
            if bad[j] != bad[i]:
                continue
            length = j - i
            sum_x, sum_y = sx[j] - sx[i], sy[j] - sy[i]
            denom = length * (sxx[j] - sxx[i]) - sum_x * sum_x
            if denom <= 0:
                continue
            cross = length * (sxy[j] - sxy[i]) - sum_x * sum_y
            y_var = length * (syy[j] - syy[i]) - sum_y * sum_y
            slope = cross / denom
            intercept = y_origin + (sum_y - slope * sum_x) / length - slope * x_origin
            score = float(np.clip(cross * cross / (denom * y_var), 0.0, 1.0)) if y_var > 0 else 1.0
            stats = {"r2": score, "slope": slope * 60.0, "intercept": intercept,
                     "start_idx": i, "end_idx": j - 1}
            if not np.isfinite([score, slope, intercept]).all():
                continue
            # start/end make the key unique; a dict is never compared on ties.
            entry = (*_rank_segment(stats), stats)
            if len(heap) < top_n:
                heapq.heappush(heap, entry)
            else:
                heapq.heappushpop(heap, entry)
    return _format_results(time_seconds, [entry[-1] for entry in heap], top_n)


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
    _validate_regression_input(time_seconds, y_values, min_points, top_n)
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
            if not (np.isfinite(x_win).all() and np.isfinite(y_win).all()) or np.ptp(x_win) == 0:
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

    return _format_results(time_seconds, results, top_n), cutoff


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
    df = read_measurements(uploaded_file, header_row=3)
    time_column = df.iloc[:, 0]
    time_seconds = convert_time_column(time_column)
    data_columns = df.columns[1:]

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
