import sys
import os
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import zipfile

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QRadioButton,
    QGroupBox, QButtonGroup, QTableWidget, QTableWidgetItem, QHeaderView,
    QProgressBar, QListWidget, QListWidgetItem, QMessageBox, QSpinBox, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWebEngineWidgets import QWebEngineView

from analyzer import (
    convert_time_column, analyze_column, analyze_column_with_saturation_cutoff,
    visualize_best_fits
)
from streamlit_ui import display_best_fits_plotly

# -----------------------------
# 백그라운드 분석 스레드
# -----------------------------
class AnalysisThread(QThread):
    # (result_df, top_results_dict, time_seconds)를 main thread에 전달
    result_ready = pyqtSignal(object, object, object)
    progress_changed = pyqtSignal(int)

    def __init__(self, df, analysis_mode, num_best, min_ratio, selected_cols):
        super().__init__()
        self.df = df
        self.analysis_mode = analysis_mode  # "range" 또는 "saturation"
        self.num_best = num_best
        self.min_ratio = min_ratio  # 최소 구간 길이 (%)
        self.selected_cols = selected_cols

    def run(self):
        try:
            # 첫 번째 열은 시간 데이터로 가정
            time_series = self.df.iloc[:, 0]
            time_seconds = convert_time_column(time_series)

            result_dict = {}
            top_results_dict = {}
            total_cols = len(self.selected_cols)

            for i, col in enumerate(self.selected_cols):
                y_values = pd.to_numeric(self.df[col], errors='coerce')
                total_points = len(time_seconds)
                min_points = max(2, int(total_points * (self.min_ratio / 100)))

                if self.analysis_mode == "range":
                    best_results = analyze_column(time_seconds, y_values, min_points, top_n=self.num_best)
                    cutoff_idx = None
                else:
                    best_results, cutoff_idx = analyze_column_with_saturation_cutoff(
                        time_seconds, y_values,
                        top_n=self.num_best,
                        min_points=min_points,
                        max_points=None
                    )

                # 각 열에 대해 플롯 파일을 ./plots 폴더에 저장
                if best_results and best_results[0][3]:
                    visualize_best_fits(time_seconds, y_values, best_results, label=col,
                                         num_best=self.num_best, cutoff_idx=cutoff_idx)

                # 결과 테이블에 표시할 데이터 생성
                row = []
                for res in best_results:
                    row += [res[0], res[1], res[2]]
                result_dict[col] = row
                top_results_dict[col] = (best_results, cutoff_idx)
                self.progress_changed.emit(int((i + 1) / total_cols * 100))

            row_labels = []
            for i in range(1, self.num_best + 1):
                row_labels += [f"Best {i} 시간", f"Best {i} r²", f"Best {i} slope"]
            result_df = pd.DataFrame(result_dict, index=row_labels)
            self.result_ready.emit(result_df, top_results_dict, time_seconds)
        except Exception as e:
            print("Error during analysis:", e)

# -----------------------------
# 메인 윈도우 클래스 (Modern & Simple 디자인)
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linear Regression Analyzer")
        self.resize(1200, 800)
        self.df = None              # 업로드된 DataFrame
        self.time_seconds = None    # 시간 데이터 (pandas Series)
        self.result_df = None       # 분석 결과 요약 DataFrame
        self.top_results_dict = {}  # {컬럼: (best_results, cutoff_idx)}
        self.analysis_thread = None

        self.initUI()

    def initUI(self):
        self.stacked = QStackedWidget()

        # --- 분석 전 페이지 (파일 업로드 및 옵션 설정) ---
        self.page_before = QWidget()
        layout_before = QVBoxLayout(self.page_before)
        layout_before.setContentsMargins(20, 20, 20, 20)
        layout_before.setSpacing(15)

        # 제목 및 설명 (좌측 정렬)
        title_label = QLabel("<h2>Linear Regression Analyzer</h2>"
                             "<p>version 1.1.0 (2025-04-14, Minsoo Lee, Seoul National University, College of Pharmacy, "
                             "WLab(Prof. Wooin Lee))</p>"
                             "<p>Upload an Excel file and configure analysis options.</p>")
        title_label.setAlignment(Qt.AlignLeft)
        layout_before.addWidget(title_label)

        # 파일 업로드 영역 (좌측 정렬)
        file_upload_layout = QHBoxLayout()
        self.file_label = QLabel("Upload Excel file (.xlsx)")
        self.file_label.setAlignment(Qt.AlignLeft)
        self.file_btn = QPushButton("Upload")
        self.file_btn.clicked.connect(self.on_upload_file)
        file_upload_layout.addWidget(self.file_label)
        file_upload_layout.addWidget(self.file_btn)
        layout_before.addLayout(file_upload_layout)

        options_group = QGroupBox("Analysis Setting")
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(10)

        # 분석 모드 선택
        self.radio_range = QRadioButton("Range")
        self.radio_saturation = QRadioButton("Saturation")
        self.radio_range.setChecked(True)
        self.mode_group = QButtonGroup()
        self.mode_group.addButton(self.radio_range)
        self.mode_group.addButton(self.radio_saturation)
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.radio_range)
        mode_layout.addWidget(self.radio_saturation)
        options_layout.addLayout(mode_layout)

        # Number of Best Fits (SpinBox)
        fits_layout = QHBoxLayout()
        fits_layout.addWidget(QLabel("Number of Best Fits:"))
        self.spin_best = QSpinBox()
        self.spin_best.setRange(1, 5)
        self.spin_best.setValue(3)
        fits_layout.addWidget(self.spin_best)
        options_layout.addLayout(fits_layout)

        # Minimum segment length (SpinBox)
        min_seg_layout = QHBoxLayout()
        min_seg_layout.addWidget(QLabel("Minimum segment length (%):"))
        self.spin_min = QSpinBox()
        self.spin_min.setRange(5, 100)
        self.spin_min.setValue(30)
        min_seg_layout.addWidget(self.spin_min)
        options_layout.addLayout(min_seg_layout)

        # 분석 대상 열 선택
        options_layout.addWidget(QLabel("Select columns to analyze:"))
        self.list_columns = QListWidget()
        self.list_columns.setSelectionMode(self.list_columns.MultiSelection)
        options_layout.addWidget(self.list_columns)

        layout_before.addWidget(options_group)
        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.clicked.connect(self.on_run_analysis)
        layout_before.addWidget(self.run_btn, alignment=Qt.AlignLeft)

        self.stacked.addWidget(self.page_before)

        # --- 분석 후 페이지 (결과 요약, 다운로드, 인터랙티브 플롯) ---
        self.page_after = QWidget()
        layout_after = QVBoxLayout(self.page_after)
        layout_after.setContentsMargins(20, 20, 20, 20)
        layout_after.setSpacing(15)

        # 분석 완료 제목은 메인 화면 제목과 동일하게
        self.label_complete = QLabel("<h2>Linear Regression Analyzer</h2>")
        layout_after.addWidget(self.label_complete, alignment=Qt.AlignLeft)

        self.table_summary = QTableWidget()
        layout_after.addWidget(self.table_summary)

        download_layout = QHBoxLayout()
        self.btn_download_excel = QPushButton("Download Excel Summary")
        self.btn_download_excel.clicked.connect(self.on_download_excel)
        self.btn_download_zip = QPushButton("Download All Plots as ZIP")
        self.btn_download_zip.clicked.connect(self.on_download_zip)
        download_layout.addWidget(self.btn_download_excel)
        download_layout.addWidget(self.btn_download_zip)
        layout_after.addLayout(download_layout)

        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Select column to visualize:"))
        self.combo_columns = QComboBox()
        self.combo_columns.currentIndexChanged.connect(self.on_combo_changed)
        combo_layout.addWidget(self.combo_columns)
        layout_after.addLayout(combo_layout)

        self.web_view = QWebEngineView()
        layout_after.addWidget(self.web_view)

        # "Run Analysis Again" 버튼을 좌우 꽉 채우도록 설정
        self.btn_rerun = QPushButton("Run Analysis Again")
        from PyQt5.QtWidgets import QSizePolicy
        self.btn_rerun.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_rerun.clicked.connect(self.on_rerun_analysis)
        layout_after.addWidget(self.btn_rerun)

        self.stacked.addWidget(self.page_after)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.stacked)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def on_upload_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx)")
        if file_name:
            try:
                df = pd.read_excel(file_name, header=2).dropna(axis=1, how='all')
                self.df = df
                # 파일 이름을 label에 표시하고 버튼 텍스트를 변경
                base_name = os.path.basename(file_name)
                self.file_label.setText(f"{base_name} loaded")
                self.file_btn.setText("Re-upload")
                # 업로드된 데이터의 컬럼 목록을 리스트 위젯에 채우기
                self.list_columns.clear()
                for col in df.columns:
                    item = QListWidgetItem(col)
                    self.list_columns.addItem(item)
                QMessageBox.information(self, "Success", f"File loaded successfully:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def on_run_analysis(self):
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Please upload an Excel file first.")
            return
        analysis_mode = "range" if self.radio_range.isChecked() else "saturation"
        num_best = self.spin_best.value()
        min_ratio = self.spin_min.value()
        selected_items = self.list_columns.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one column.")
            return
        selected_cols = [item.text() for item in selected_items]
        self.analysis_thread = AnalysisThread(
            df=self.df,
            analysis_mode=analysis_mode,
            num_best=num_best,
            min_ratio=min_ratio,
            selected_cols=selected_cols
        )
        self.analysis_thread.progress_changed.connect(self.on_progress_changed)
        self.analysis_thread.result_ready.connect(self.on_analysis_complete)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.analysis_thread.start()

    def on_progress_changed(self, val):
        self.progress_bar.setValue(val)

    def on_analysis_complete(self, result_df, top_results_dict, time_seconds):
        self.result_df = result_df
        self.top_results_dict = top_results_dict
        self.time_seconds = time_seconds
        self.show_result_table(result_df)
        self.combo_columns.clear()
        for col in self.top_results_dict.keys():
            self.combo_columns.addItem(col)
        if self.combo_columns.count() > 0:
            self.update_interactive_plot(self.combo_columns.currentText())
        self.stacked.setCurrentWidget(self.page_after)
        self.progress_bar.setVisible(False)

    def show_result_table(self, df: pd.DataFrame):
        self.table_summary.clear()
        self.table_summary.setRowCount(df.shape[0])
        self.table_summary.setColumnCount(df.shape[1])
        self.table_summary.setHorizontalHeaderLabels(df.columns.tolist())
        self.table_summary.setVerticalHeaderLabels(df.index.astype(str).tolist())
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                item = QTableWidgetItem(str(df.iloc[r, c]))
                self.table_summary.setItem(r, c, item)
        header = self.table_summary.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)

    def on_combo_changed(self):
        selected_col = self.combo_columns.currentText()
        self.update_interactive_plot(selected_col)

    def update_interactive_plot(self, col):
        if col not in self.top_results_dict:
            return
        best_results, cutoff_idx = self.top_results_dict[col]
        y_values = pd.to_numeric(self.df[col], errors="coerce")
        html_str = display_best_fits_plotly(self.time_seconds, y_values, best_results, label=col,
                                             cutoff_idx=cutoff_idx, num_best=self.spin_best.value()).to_html(include_plotlyjs='cdn')
        self.web_view.setHtml(html_str)

    def on_download_excel(self):
        if self.result_df is None:
            QMessageBox.warning(self, "Warning", "No results to download.")
            return
        default_filename = f"best_fit_summary_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Excel File", default_filename, "Excel Files (*.xlsx)")
        if file_name:
            try:
                self.result_df.to_excel(file_name, index=True)
                QMessageBox.information(self, "Success", f"Excel summary saved:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save Excel:\n{str(e)}")

    def on_download_zip(self):
        default_filename = f"plots_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.zip"
        file_name, _ = QFileDialog.getSaveFileName(self, "Save ZIP File", default_filename, "ZIP Files (*.zip)")
        if file_name:
            try:
                plots_dir = os.path.join(os.getcwd(), "plots")
                if not os.path.exists(plots_dir):
                    QMessageBox.warning(self, "Warning", f"Plots directory not found:\n{plots_dir}")
                    return
                with zipfile.ZipFile(file_name, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
                    for foldername, subfolders, filenames in os.walk(plots_dir):
                        for filename in filenames:
                            file_path = os.path.join(foldername, filename)
                            arcname = os.path.relpath(file_path, plots_dir)
                            zipf.write(file_path, arcname)
                QMessageBox.information(self, "Success", f"ZIP saved:\n{file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save ZIP:\n{str(e)}")

    def on_rerun_analysis(self):
        self.stacked.setCurrentWidget(self.page_before)

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    style_sheet = """
    QWidget { 
        background-color: #f8f8f8; 
        font-family: 'Segoe UI', sans-serif; 
        font-size: 12pt;
    }
    QPushButton { 
        background-color: #3e8ef7; 
        color: white; 
        border: none; 
        border-radius: 4px; 
        padding: 8px 16px;
    }
    QPushButton:hover { 
        background-color: #6ab7ff; 
    }
    QGroupBox { 
        border: 1px solid #cccccc; 
        margin-top: 10px;
        border-radius: 4px;
    }
    QGroupBox::title { 
        subcontrol-origin: margin; 
        subcontrol-position: top center; 
        padding: 0 3px;
    }
    QTableWidget {
        background: white;
    }
    QProgressBar {
        text-align: center;
        height: 20px;
    }
    """
    app.setStyleSheet(style_sheet)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
