import io
import os
from pathlib import Path
import tempfile
import unittest
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.linear_model import LinearRegression

from analyzer import (
    time_to_seconds_from_any, convert_time_column, read_measurements,
    analyze_column, analyze_column_optimized, analyze_column_prefix,
    analyze_column_with_saturation_cutoff, summarize_results, visualize_best_fits,
)


class TimeInputTests(unittest.TestCase):
    def test_numeric_seconds(self):
        self.assertEqual(time_to_seconds_from_any(60, numeric_unit="seconds"), 60)

    def test_numeric_minutes(self):
        self.assertEqual(time_to_seconds_from_any(1.5, numeric_unit="minutes"), 90)

    def test_numeric_hours(self):
        self.assertEqual(time_to_seconds_from_any(0.5, numeric_unit="hours"), 1800)

    def test_excel_days_remain_default(self):
        self.assertEqual(time_to_seconds_from_any(0.5), 43200)

    def test_numpy_numbers_and_numeric_text(self):
        for value in (np.int64(60), np.float64(60), "60"):
            with self.subTest(value=value):
                self.assertEqual(time_to_seconds_from_any(value, numeric_unit="seconds"), 60)

    def test_mm_ss(self):
        self.assertEqual(time_to_seconds_from_any("0:01"), 1)
        self.assertEqual(time_to_seconds_from_any("1:30"), 90)

    def test_hh_mm_ss(self):
        self.assertEqual(time_to_seconds_from_any("1:02:03"), 3723)

    def test_fractional_seconds(self):
        self.assertEqual(time_to_seconds_from_any(" 0:01.25 "), 1.25)

    def test_long_durations(self):
        self.assertEqual(time_to_seconds_from_any("25:00:00"), 90000)
        self.assertEqual(time_to_seconds_from_any("120:00"), 7200)

    def test_excel_clock_and_timedelta(self):
        self.assertEqual(time_to_seconds_from_any(time(0, 1, 30)), 90)
        self.assertEqual(time_to_seconds_from_any(timedelta(days=2, seconds=3)), 172803)

    def test_datetime_crosses_midnight(self):
        times = pd.Series([datetime(2026, 11, 24, 23, 59), datetime(2026, 11, 25, 0, 1)])
        np.testing.assert_array_equal(convert_time_column(times), [0, 120])

    def test_invalid_scalars_return_nan(self):
        for value in (None, pd.NaT, np.nan, np.inf, True, "", "abc", "1:60", "0:60:00"):
            with self.subTest(value=value):
                self.assertTrue(np.isnan(time_to_seconds_from_any(value)))

    def test_unknown_unit_is_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            time_to_seconds_from_any(1, numeric_unit="auto")

    def test_relative_origin_and_index(self):
        result = convert_time_column(pd.Series([5., 6., 7.], index=[10, 20, 30]), "minutes")
        np.testing.assert_array_equal(result, [0, 60, 120])
        self.assertEqual(result.index.tolist(), [10, 20, 30])

    def test_empty_time_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            convert_time_column(pd.Series([], dtype=float))

    def test_missing_first_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "row.*1"):
            convert_time_column(pd.Series([np.nan, 1, 2]), "seconds")

    def test_missing_later_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "row.*2"):
            convert_time_column(pd.Series([0, "bad", 2]), "seconds")

    def test_decreasing_times_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "order"):
            convert_time_column(pd.Series([0, 2, 1]), "seconds")


class RegressionTests(unittest.TestCase):
    def test_exact_r2_ties_do_not_compare_dicts(self):
        x, y = pd.Series([0., 60., 120., 180.]), pd.Series([0., 1., 2., 3.])
        result = analyze_column_prefix(x, y, 2, top_n=3)
        self.assertEqual([(r[3]["start_idx"], r[3]["end_idx"]) for r in result], [(0, 3), (0, 2), (1, 3)])
        self.assertEqual(result, analyze_column_prefix(x, y, 2, top_n=3))

    def test_gap_excludes_crossing_windows_but_preserves_later_indices(self):
        x, y = pd.Series(np.arange(7.) * 60), pd.Series([0., 1., np.nan, 3., 4., 5., 6.])
        result = analyze_column_prefix(x, y, 3, top_n=2)
        self.assertEqual((result[0][3]["start_idx"], result[0][3]["end_idx"]), (3, 6))
        for row in result:
            self.assertGreater(row[3]["start_idx"], 2)
            self.assertAlmostEqual(row[3]["slope"], 1)

    def test_nonfinite_measurements_do_not_poison_later_windows(self):
        x, y = pd.Series([0., 60., 120., 180.]), pd.Series([np.inf, 1., 2., 3.])
        self.assertEqual(analyze_column_prefix(x, y, 3, 1)[0][3]["start_idx"], 1)

    def test_all_missing_returns_placeholders(self):
        result = analyze_column_prefix(pd.Series([0., 60., 120.]), pd.Series([np.nan] * 3), 2, 3)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(r[3] is None for r in result))

    def test_only_gap_crossing_window_returns_placeholder(self):
        self.assertIsNone(analyze_column_prefix(pd.Series([0., 60., 120.]), pd.Series([0., np.nan, 2.]), 3, 1)[0][3])

    def test_nonfinite_time_is_excluded_without_reindexing(self):
        result = analyze_column_prefix(pd.Series([np.nan, 60., 120., 180.]), pd.Series([0., 1., 2., 3.]), 3, 1)
        self.assertEqual(result[0][3]["start_idx"], 1)

    def test_flat_measurements_have_perfect_constant_fit(self):
        stats = analyze_column_prefix(pd.Series([0., 60., 120.]), pd.Series([2.] * 3), 3, 1)[0][3]
        self.assertEqual(stats["r2"], 1)
        self.assertEqual(stats["slope"], 0)
        self.assertEqual(stats["intercept"], 2)

    def test_no_time_variation_has_no_regression(self):
        self.assertIsNone(analyze_column_prefix(pd.Series([0., 0., 0.]), pd.Series([1., 2., 3.]), 2, 1)[0][3])

    def test_last_point_is_included_by_all_implementations(self):
        for fn in (analyze_column, analyze_column_optimized, analyze_column_prefix):
            with self.subTest(fn=fn.__name__):
                stats = fn(pd.Series([0., 60., 120.]), pd.Series([0., .2, .4]), 3, 1)[0][3]
                self.assertEqual(stats["end_idx"], 2)
                self.assertAlmostEqual(stats["slope"], .2)

    def test_saturation_path_preserves_small_coefficients(self):
        result, _ = analyze_column_with_saturation_cutoff(pd.Series([0., 60., 120.]), pd.Series([.125, .325, .525]), top_n=1, min_points=3)
        self.assertIn("0.2 * t_min", result[0][2])
        self.assertIn("+0.125", result[0][2])

    def test_coefficients_match_independent_least_squares(self):
        rng = np.random.default_rng(2026)
        x = pd.Series(np.arange(14.) * 60)
        y = pd.Series(-0.03 * x + 10 + rng.normal(0, .5, len(x)))
        y.iloc[4] = np.nan
        for _, _, _, stats in analyze_column_prefix(x, y, 4, top_n=5):
            i, j = stats["start_idx"], stats["end_idx"] + 1
            model = LinearRegression().fit(x.iloc[i:j].to_numpy().reshape(-1, 1), y.iloc[i:j])
            self.assertAlmostEqual(stats["slope"], model.coef_[0] * 60, places=9)
            self.assertAlmostEqual(stats["intercept"], model.intercept_, places=9)
            self.assertAlmostEqual(stats["r2"], model.score(x.iloc[i:j].to_numpy().reshape(-1, 1), y.iloc[i:j]), places=9)

    def test_small_negative_coefficients_are_not_zeroed(self):
        result = analyze_column_prefix(pd.Series([0., 60., 120.]), pd.Series([-.125, -.325, -.525]), 3, 1)
        self.assertIn("-0.2 * t_min -0.125", result[0][2])

    def test_invalid_lengths_and_settings_are_rejected(self):
        for x, y, points, top in [([0], [1, 2], 2, 1), ([0, 1], [1, 2], 1, 1), ([0, 1], [1, 2], 2, 0)]:
            with self.subTest(points=points, top=top):
                with self.assertRaises(ValueError):
                    analyze_column_prefix(pd.Series(x), pd.Series(y), points, top)


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.x = pd.Series([0., 60., 120.])
        self.y = pd.Series([.125, .325, .525])
        self.result = analyze_column_prefix(self.x, self.y, 3, 1)

    def test_summary_keeps_numeric_coefficients(self):
        summary = summarize_results({"Sample": self.result}, 1)
        self.assertAlmostEqual(summary.loc["Best 1 Slope (value/min)", "Sample"], .2)
        self.assertAlmostEqual(summary.loc["Best 1 Intercept", "Sample"], .125)

    def test_xlsx_round_trip_keeps_numbers_and_units(self):
        buf = io.BytesIO()
        summarize_results({"Sample": self.result}, 1).to_excel(buf)
        sheet = load_workbook(io.BytesIO(buf.getvalue())).active
        self.assertEqual(sheet['B5'].data_type, 'n')
        self.assertAlmostEqual(sheet['B5'].value, .2)
        self.assertAlmostEqual(sheet['B6'].value, .125)
        self.assertIn("t_min", sheet['B4'].value)

    def test_input_header_row_is_selectable_and_a1_is_not_required(self):
        for offset in (0, 2):
            with self.subTest(header_row=offset + 1):
                buf = io.BytesIO()
                pd.DataFrame({'Elapsed': [0, 60, 120], 'Sample': [.125, .325, .525]}).to_excel(buf, index=False, startrow=offset)
                buf.seek(0)
                frame = read_measurements(buf, header_row=offset + 1)
                self.assertEqual(frame.columns.tolist(), ['Elapsed', 'Sample'])
                self.assertEqual(len(frame), 3)

    def test_input_requires_measurement_column(self):
        buf = io.BytesIO()
        pd.DataFrame({'Time': [0, 1]}).to_excel(buf, index=False)
        buf.seek(0)
        with self.assertRaises(ValueError):
            read_measurements(buf, header_row=1)

    def test_static_and_interactive_plots_match_numeric_fit(self):
        from streamlit_ui import display_best_fits_plotly
        fig = display_best_fits_plotly(self.x, self.y, self.result, "Sample", num_best=1)
        fit_trace = fig.data[2]
        np.testing.assert_allclose(np.asarray(fit_trace.y), .2 * np.asarray(fit_trace.x) / 60 + .125)
        self.assertIn("0.2 * t_min", fit_trace.name)
        with tempfile.TemporaryDirectory() as tmp:
            visualize_best_fits(self.x, self.y, self.result, 'Sample', out_dir=tmp, num_best=1)
            self.assertGreater((Path(tmp) / 'Sample_best1_fits.png').stat().st_size, 1000)


if __name__ == '__main__':
    unittest.main()
