# Linear-Regression-Fitter

This program performs **linear regression** between time and multiple data columns across various subranges to find the **best-fitting linear segments** within the dataset.

이 프로그램은 시간과 관측값 간의 선형 회귀를 다양한 구간에서 수행하여, 가장 피팅이 잘 되는 구간을 자동으로 탐색합니다.

➡️ [Can be used on Streamlit Cloud](https://linear-regression-fitter-9a38ezlgjpskkj9fcwvxji.streamlit.app/)

---

## 📁 Input Format

Upload an `.xlsx` file with the following format:

| Time  | A1      | A2      | ...   |
|-------|---------|---------|-------|
| 0:00  | 123980  | 213900  | ...   |
| 1:00  | 192388  | 221980  | ...   |
| ...   | ...     | ...     | ...   |

- **Time** doesn't have to be in `hh:mm:ss` format (e.g. `0:00:00`, `0:01:00`), but follow the format when error occured.
- Data columns (A1, A2, ...) will be detected automatically.

---

## ⚙️ Data Analysis Settings

You can customize the following options before running the analysis:

### 🔹 Number of Best Fits to Display
- Choose how many best-fitting segments to return.
- For example, selecting "3" will return **Top 1, Top 2, and Top 3** best fits based on R² score.

### 🔹 Minimum Analysis Window (Sliding Window)
- Define the **minimum size of the segment** (sliding window) used for fitting.
- This is specified as a percentage of total timepoints (e.g., 30% of 40 points = at least 12 points per segment).
- 💡 *We recommend at least **10 points** or **20%** of your data to ensure statistical significance.*

### 🔹 Columns to Analyze
- You may select **one or multiple** columns to analyze.
- All eligible columns will be detected automatically from your uploaded Excel file.

---

## 📋 Results & Download

Once analysis is complete, download options will be shown:

- 📥 **Download Excel Summary**:  
  Outputs an `.xlsx` file with regression summary including:
  - Time range used
  - Best-fit R² values
  - Fitting equations (`slope * time + intercept`)

- 📦 **Download All Plots (ZIP)**:  
  Contains plots for each selected column showing:
  - All data points  
  - Best-fit regression line(s)  
  - Highlighted subset used for fitting  
  - (if enabled) Saturation cutoff point

- 🖼 **Interactive Visualization**
  - From **v1.1.0**, you can now preview best fitting results using **interactive Plotly charts** directly in the app.
  - Visualize top 1 to 5 regression results per column.
  - Hover to inspect data points and regression lines.
  - Saturation cutoff lines will also be marked if applicable.

To use:
1. Complete the analysis with your desired settings.
2. Scroll to "🖼 Interactive Plot Viewer".
3. Use the slider to select a data column and interactively explore its best fits.

---

## 💻 Deployments

This app is available as:
- ✅ A web app via [Streamlit Cloud](https://linear-regression-fitter-9a38ezlgjpskkj9fcwvxji.streamlit.app/)
- ✅ A Windows desktop app (generated via [Nativefier](https://drive.google.com/file/d/1Dfah4LvvVWkrbT37a9C-1f3nhCMH-me8/view?usp=sharing))

---

## 🧪 Technologies Used

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Plotly](https://plotly.com/)
- [Nativefier](https://github.com/nativefier/nativefier) (for Windows desktop app)

---

## 👤 Author

**Minsoo Lee, PharmD, RPh**  
Seoul National University, College of Pharmacy  
WLab (Prof. Wooin Lee)

---

## 📄 License

MIT License
