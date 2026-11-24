# Linear-Regression-Fitter

Search contiguous intervals in time-series measurements, rank their linear fits,
and export the results from Streamlit or the PyQt desktop source.

**Source version 1.3.1 (2026-11-24)** fixes equal-R² ranking, missing measurements,
time input, and coefficient display/export. See [validation](docs/validation.md).

## Run from source

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
# Desktop alternative:
python pyqt_app.py
```

[Streamlit Cloud](https://linear-regression-fitter-9a38ezlgjpskkj9fcwvxji.streamlit.app/)
uses the repository source; the live deployment must be checked separately after a push.
The [historical Windows executable](https://drive.google.com/file/d/1psZ2h_yWe0FFY0abXpzv0f4ywYzIf89P/view?usp=drive_link)
predates 1.3.1 and does **not** include these fixes. Use the current source until a new executable is built.

## Excel input

Upload an `.xlsx` file. Set **Header row** to the 1-based row containing column
names (default **3**, matching older workbooks; choose **1** for the table below).
The first column is time; all subsequent nonempty columns are measurements.
Column names such as `A1` are no longer required.

| Time | Sample A | Sample B |
|---|---:|---:|
| 0:00 | 0.125 | 1.20 |
| 1:00 | 0.325 | 1.10 |
| 2:00 | 0.525 | 1.00 |
| 3:00 | 0.725 | 0.90 |

Text times above are **MM:SS**: `1:00` is 60 seconds.
Three fields are **HH:MM:SS**: `1:00:00` is one hour.
Fractional seconds and durations longer than 24 hours are supported.
Excel time/datetime cells and timedeltas are also accepted.

For numbers or numeric strings, choose **Numeric time unit**:

| Setting | Interpretation of numeric 1 |
|---|---|
| Excel days (default) | 86,400 seconds, preserving legacy Excel serial values |
| Seconds | 1 second |
| Minutes | 60 seconds |
| Hours | 3,600 seconds |

The first timestamp becomes elapsed time zero. Invalid/missing timestamps or
decreasing times produce an error rather than a guessed conversion.
The selected numeric unit does not change the meaning of colon-separated text.

## Analysis

- **Range** searches intervals with at least the selected minimum number of rows.
- **Saturation** first applies the existing slope-threshold cutoff, then searches
  the retained prefix.
- Missing, nonnumeric or infinite measurements invalidate any interval containing
  them. They are never filled with zero or silently joined across a gap.
  Valid intervals after a gap remain searchable at their original row positions.
- On exact R² ties, prefer more points, then the earlier interval.
- If no interval qualifies, the result is `N/A`.
- A two-point fit has no residual degrees of freedom; choose the minimum length
  according to the experiment. R² alone does not establish statistical significance.

## Results and units

Plots use **elapsed seconds**. Every fit equation explicitly uses **t_min**
(elapsed minutes), so `0.2 * t_min +0.125` describes the example above.

The on-screen summary and Excel export include region, R², equation, numeric
**Slope (value/min)** and numeric **Intercept** rows.
Equations display 12 significant digits; numeric coefficient rows retain the
computed floating-point values in Excel. Small/negative coefficients are not
converted to integers. The static PNG and Plotly curves use the numeric coefficients.

The web app offers Excel and plot ZIP downloads and an interactive plot selector.
Changing input files, header row, time unit or analysis settings invalidates old
results, so the previous export cannot be mistaken for a new analysis.

This is a general linear-interval tool. It does not automatically transform
concentrations to log remaining fraction or calculate CLint.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the four reported defects, later valid intervals after missing
data, last-point inclusion, independent scikit-learn agreement, Excel numeric
round trips, static/interactive plots, Streamlit input-to-export flow, and the
PyQt analysis worker. Details and execution limits are in
[docs/validation.md](docs/validation.md).

## Author

Minsoo Lee · Seoul National University College of Pharmacy · WLab (Prof. Wooin Lee)

## License

MIT License
