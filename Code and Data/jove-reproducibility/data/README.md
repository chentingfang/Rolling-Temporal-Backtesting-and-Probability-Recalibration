# Anonymized prediction archive

`anonymized_prediction_archive.csv` records frozen model predictions and outcomes for the manuscript's retrospective evaluation. The archive was supplied with the manuscript revision as `Anonymized Predictive Data Archive.csv`. The packaged copy preserves the supplied bytes. Firm identifiers are stable analysis identifiers that link a firm's records across years and horizons.

Each record is one firm at one forecast horizon and target year. The composite key is `(firm_id, horizon, year)`. The four model streams share the same outcome and record key.

| Field | Type and valid values | Meaning |
| --- | --- | --- |
| `firm_id` | String, e.g. `F00001` | Stable anonymized firm identifier |
| `feature_year` | Integer | Feature reference year. The archive satisfies `year - feature_year = horizon`. |
| `year` | Integer, 2020–2024 | Target year of the forecast and outcome |
| `horizon` | Integer, 1 or 2 | Forecast lead in years |
| `y` | Integer, 0 or 1 | Annual outcome. A value of 1 denotes ST or *ST status in the target year. |
| `p_<model>` | Finite numeric value in [0, 1] | Archived model probability |
| `threshold_<model>` | Finite numeric value in [0, 1] | Archived decision cutoff, constant within a model, horizon, and target year |
| `yhat_<model>` | Integer, 0 or 1 | Archived binary warning. It equals 1 when the probability meets or exceeds the cutoff. |
| `score_consensus` | Numeric | Additional field preserved from the supplied archive and unused by the reproducibility workflow |

The four model suffixes are `logistic` for logistic regression, `svm` for support vector machine, `rf` for random forest, and `xgb` for XGBoost.

| Target year | One-year records | One-year events | Two-year records | Two-year events |
| --- | ---: | ---: | ---: | ---: |
| 2020 | 2,261 | 54 | 2,026 | 52 |
| 2021 | 2,024 | 31 | 1,935 | 31 |
| 2022 | 2,031 | 18 | 1,962 | 17 |
| 2023 | 2,299 | 14 | 1,887 | 12 |
| 2024 | 2,599 | 22 | 2,221 | 20 |
| Total | 11,214 | 139 | 10,031 | 132 |

There are 21,245 records and 3,377 distinct firms. Event counts refer to positive records and can include the same firm across years or horizons.
