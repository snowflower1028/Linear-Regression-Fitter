# Validation of version 1.3.1

Validated on 2026-11-24 with Python 3.12.14 on Windows.

## Reported defects and fixes

| Case | Previous behavior | Fixed behavior |
|---|---|---|
| Equal R², time `[0,60,120,180]`, values `[0,1,2,3]`, min 2, top 3 | Heap compared dictionaries and raised TypeError | Unique scalar ranking keys; prefer longer, earlier intervals on exact ties |
| Missing value `[0,NaN,2]` | NaN contaminated cumulative sums and formatting failed | Reject windows containing non-finite pairs; return N/A if no interval qualifies |
| Gap followed by valid measurements | Cumulative NaN could invalidate every later window | Valid later intervals remain searchable with original row indices |
| Numeric time `60` intended as seconds | Treated as 60 Excel days | Explicit unit selection; seconds converts to 60 s, legacy Excel-days remains the documented default |
| Text `0:01` | Unsupported | Parsed as MM:SS, one second; `0:01:00` is HH:MM:SS, 60 seconds |
| Slope `0.2`, intercept `0.125` | Equation became `0x+0` | Equation `0.2 * t_min +0.125`; numeric coefficients preserved in Excel |

Related corrections include last-point inclusion, an explicit Excel header row,
measurement columns without an A1 naming requirement, and invalidation of stale
web exports after settings change.

## Reproduce

From the repository root, after installing `requirements.txt`:

```bash
python -m unittest discover -s tests -v
```

**Result: 42 tests passed, 0 failures, 0 errors.**

- 18 time parsing/unit/validation tests.
- 13 regression tests, including comparison with independent scikit-learn fits.
- 5 workbook/export/plot tests, including numeric Excel round trips.
- 4 Streamlit AppTest scenarios: input-to-export, missing measurements, invalid
  time and empty column selection. Uploaded XLSX bytes are supplied locally by
  a patched uploader; analysis, widgets, session state, plotting and export
  generation run through the actual app script.
- 2 PyQt analysis-worker tests for unit conversion/summary and error signaling.
  These execute `AnalysisThread.run()` directly; the desktop window is not opened.

## Tested versions

| Package | Version |
|---|---|
| NumPy | 2.3.5 |
| pandas | 3.0.1 |
| scikit-learn | 1.9.1 |
| Matplotlib | 3.11.2 |
| Streamlit | 1.67.0 |
| Plotly | 7.1.0 |
| openpyxl | 3.1.5 |
| PyQt5 | 5.15.11 |
| PyQtWebEngine | 5.15.7 |

This records the tested environment, not validation of every version allowed
by the minimum requirements. Streamlit emitted its existing
`use_container_width` deprecation notice and the expected AppTest bare-context
warning; all tests completed successfully.

## Scope

The tests validate software behavior on synthetic data. They do not establish
scientific suitability of an R²-selected interval for a particular assay or
validate a CLint calculation. No original research data are included.

The historical downloadable Windows executable has not been rebuilt and still
predates these source fixes. A deployed Streamlit instance must be checked
separately; a source commit alone does not prove the deployed version.
