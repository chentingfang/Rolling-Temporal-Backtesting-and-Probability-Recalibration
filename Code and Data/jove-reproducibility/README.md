# JoVE reproducibility package

This package accompanies *Rolling Temporal Backtesting and Probability Recalibration for Financial Distress Early Warning in Listed Companies*. It runs the manuscript's retrospective workflow on 21,245 archived one- and two-year forecasts for 3,377 anonymized firms, with target years 2020–2024. The CSV contains the observed outcomes and predictions from logistic regression, SVM, random forest, and XGBoost. The workflow starts from these frozen predictions.

## Run the analysis

Use Python 3.12. The reference run used Python 3.12.10 on Windows with the versions pinned in `requirements.txt`. Run the commands from the folder that contains `reproduce.py`.

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -v
.\.venv\Scripts\python.exe reproduce.py
```

**macOS and Linux**

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -v
.venv/bin/python reproduce.py
```

The default run uses base seed `20260719` and 2,000 bootstrap replicates per horizon for paired model comparisons. It verifies the bundled CSV's SHA-256 before analysis and creates `results/`. Use an empty or new directory with `--output` for another run. The commands below assume that `python` points to the configured environment.

```bash
python reproduce.py --output results_repeat
python reproduce.py --bootstrap 20 --no-figures --output results_smoke
```

Use the smoke run to check installation and CSV export. The default seed and replicate count are required for the manuscript regression checks. See `python reproduce.py --help` for options. The Windows helper `run_reproduction.ps1` accepts `-PythonExecutable` and writes to a timestamped directory.

NumPy and pandas perform the calculations. Matplotlib produces figures. The pipeline reads and writes CSV and JSON, so it runs without OpenPyXL or a validation workbook. Once the dependencies are installed, the analysis runs offline.

## Input and integrity

The sole analysis input, [anonymized_prediction_archive.csv](data/anonymized_prediction_archive.csv), is a byte-for-byte copy of the anonymized CSV supplied with the revision materials. See the [data dictionary](data/README.md) for the schema and cohort counts.

```text
0cafa218445c8b71804448767afce09d5698a50b73eda52e4c6d284f039c8e39
```

The release checksum is also recorded in [SHA256SUMS](SHA256SUMS). To verify it manually on Windows, run `Get-FileHash -Algorithm SHA256 -LiteralPath .\data\anonymized_prediction_archive.csv`. On Linux, run `sha256sum -c SHA256SUMS`. On macOS, run `shasum -a 256 -c SHA256SUMS`.

