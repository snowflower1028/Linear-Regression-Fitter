import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os

import pandas as pd


class StreamlitTests(unittest.TestCase):
    def setUp(self):
        from streamlit.testing.v1 import AppTest
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # Isolate generated plot files from the source checkout.
        self.old_cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.old_cwd)
        self.app_file = str(Path(__file__).resolve().parents[1] / 'app.py')
        self.AppTest = AppTest

    def workbook(self, frame):
        buf = io.BytesIO()
        frame.to_excel(buf, index=False, startrow=2)
        buf.seek(0)
        return buf

    def test_upload_analyze_export_and_unit_change(self):
        frame = pd.DataFrame({'Time': [0, 60, 120, 180], 'Sample': [.125, .325, .525, .725]})
        with patch('streamlit.file_uploader', return_value=self.workbook(frame)):
            app = self.AppTest.from_file(self.app_file, default_timeout=30).run()
            app.selectbox[0].select('Seconds').run()
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.session_state.analysis_done)
            self.assertAlmostEqual(app.session_state.result_df.loc['Best 1 Slope (value/min)', 'Sample'], .2)
            self.assertEqual(len(app.get('download_button')), 2)
            app.selectbox[0].select('Minutes').run()
            self.assertFalse(app.session_state.analysis_done)
            self.assertEqual(len(app.get('download_button')), 0)

    def test_missing_measurements_warn_and_still_analyze(self):
        frame = pd.DataFrame({'Time': [0, 60, 120, 180], 'Sample': [None, .325, .525, .725]})
        with patch('streamlit.file_uploader', return_value=self.workbook(frame)):
            app = self.AppTest.from_file(self.app_file, default_timeout=30).run()
            app.selectbox[0].select('Seconds').run()
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.session_state.analysis_done)
            self.assertTrue(any('missing/non-finite' in w.value for w in app.warning))
            self.assertAlmostEqual(app.session_state.result_df.loc['Best 1 Slope (value/min)', 'Sample'], .2)

    def test_invalid_time_displays_error_and_no_export(self):
        frame = pd.DataFrame({'Time': [0, 'invalid', 120], 'Sample': [1, 2, 3]})
        with patch('streamlit.file_uploader', return_value=self.workbook(frame)):
            app = self.AppTest.from_file(self.app_file, default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('Invalid or missing time' in e.value for e in app.error))
            self.assertFalse(app.session_state.analysis_done)
            self.assertEqual(len(app.get('download_button')), 0)

    def test_no_selected_columns_does_not_report_success(self):
        frame = pd.DataFrame({'Time': [0, 60, 120], 'Sample': [1, 2, 3]})
        with patch('streamlit.file_uploader', return_value=self.workbook(frame)):
            app = self.AppTest.from_file(self.app_file, default_timeout=30).run()
            app.multiselect[0].set_value([]).run()
            app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(app.session_state.analysis_done)
            self.assertTrue(any('Select at least one' in w.value for w in app.warning))


class DesktopWorkerTests(unittest.TestCase):
    def test_worker_applies_time_unit_and_shared_summary(self):
        from pyqt_app import AnalysisThread
        frame = pd.DataFrame({'Time': [0., 60., 120.], 'Sample': [.125, .325, .525]})
        thread = AnalysisThread(frame, 'range', 3, 30, ['Sample'], numeric_unit='seconds')
        results, errors = [], []
        thread.result_ready.connect(lambda *args: results.append(args))
        thread.analysis_failed.connect(errors.append)
        with patch('pyqt_app.visualize_best_fits'):
            thread.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0][0].loc['Best 1 Slope (value/min)', 'Sample'], .2)
        self.assertEqual(results[0][2].tolist(), [0, 60, 120])

    def test_worker_reports_invalid_time(self):
        from pyqt_app import AnalysisThread
        frame = pd.DataFrame({'Time': [0, 'bad', 120], 'Sample': [1, 2, 3]})
        thread = AnalysisThread(frame, 'range', 1, 30, ['Sample'], numeric_unit='seconds')
        results, errors = [], []
        thread.result_ready.connect(lambda *args: results.append(args))
        thread.analysis_failed.connect(errors.append)
        thread.run()
        self.assertEqual(results, [])
        self.assertEqual(len(errors), 1)
        self.assertIn('Invalid or missing time', errors[0])


if __name__ == '__main__':
    unittest.main()
