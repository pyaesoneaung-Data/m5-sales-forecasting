# M5 Sales Forecasting

## Business Problem

This project builds an end-to-end time-series forecasting workflow for Walmart M5 sales data. The goal is to understand demand patterns and compare forecasting methods that can support inventory planning, staffing decisions, and operational monitoring.


## Public Portfolio Note

Dataset files are not included in this repository. Download the Walmart M5 Forecasting Accuracy data separately from Kaggle and place the CSV files in `data/raw/` before running the pipeline.

## Dataset

The project uses the Walmart M5 Forecasting Accuracy dataset from Kaggle: <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>.

Raw files expected in `data/raw/`:

- `calendar.csv`
- `sell_prices.csv`
- `sales_train_validation.csv`
- `sales_train_evaluation.csv`
- `sample_submission.csv`

Raw and processed datasets are excluded from GitHub.

## Scope

The current modeling scope is a manageable prototype for store `CA_1`. Sales are aggregated to total daily store sales before modeling. This is not an all-store pipeline and does not produce an official M5 competition submission.

## Workflow

1. Inspect raw data quality and dataset relationships.
2. Transform `CA_1` item-level wide sales into a long-format Parquet table.
3. Explore sales trends, seasonality, product groups, events, SNAP days, prices, and intermittent demand.
4. Aggregate `CA_1` sales by date for fair store-level model comparison.
5. Evaluate baseline, traditional time-series, and machine-learning models with rolling 28-day validation.
6. Export model metrics, figures, and a final 28-day store-level forecast.

## Repository Structure

```text
data/raw/                 Raw M5 CSV files, ignored by Git
data/processed/           Processed Parquet files, ignored by Git
notebooks/                Data wrangling, EDA, and modeling notebooks
src/                      Reusable data and modeling pipeline scripts
outputs/figures/          Generated EDA and modeling charts
outputs/metrics/          Model comparison tables and evaluation summary
outputs/forecasts/        Final forecast CSV output
presentation/             Supervisor or stakeholder presentation materials
```

## Methods Compared

- Naive forecast
- Seasonal naive with 7-day lag
- Seasonal naive with 28-day lag
- Holt-Winters Exponential Smoothing with weekly seasonality
- Random Forest regression with leakage-safe lag, rolling, calendar, event, and SNAP features

## Validation Strategy

The modeling notebook uses three rolling-origin validation folds with a 28-day forecast horizon. Validation dates always occur after training dates, and no random train-test split is used. Random Forest validation forecasts are recursive, so actual validation sales are never inserted into future validation features.

## Metrics

- MAE: mean absolute error
- RMSE: root mean squared error
- RMSSE: root mean squared scaled error, scaled using only each fold's training data

## Actual Best Model Result

Random Forest had the lowest average RMSSE across validation folds:

| Model | Average MAE | Average RMSE | Average RMSSE |
| --- | ---: | ---: | ---: |
| Random Forest | 266.02 | 329.87 | 0.3821 |
| Holt-Winters | 284.71 | 377.23 | 0.4370 |
| Seasonal Naive 28 | 379.76 | 491.70 | 0.5696 |
| Seasonal Naive 7 | 457.65 | 589.61 | 0.6828 |
| Naive | 1438.65 | 1588.99 | 1.8415 |

The strongest baseline was Seasonal Naive 28. Random Forest improved average RMSSE by 32.91% versus that baseline.

## Key EDA Findings

- The CA_1 daily series covers 2011-01-29 through 2016-04-24 with 1,913 daily observations.
- Sunday had the highest average daily sales, while Wednesday had the lowest.
- `FOODS` was the highest-sales category, while `HOBBIES` was the lowest.
- `FOODS_3` was the highest-sales department, while `HOBBIES_2` was the lowest.
- Average daily sales were higher on SNAP-active days than SNAP-inactive days.
- Event days had lower average daily sales than non-event days in this CA_1 aggregate view.
- 63.95% of item-day records had zero sales, showing substantial intermittent demand.
- The top-selling item was `FOODS_3_090`.

## Business Recommendations

- Use daily store-level forecasts to support near-term inventory and staffing plans.
- Plan weekly operations around stronger weekend demand, especially Sunday.
- Give extra planning attention to high-volume food departments.
- Monitor event-day and SNAP-day forecast errors separately because those periods behave differently.
- Treat zero-heavy items carefully at lower levels of granularity because intermittent demand can make item forecasts unstable.
- Track forecast residuals after deployment and refresh validation as new data becomes available.
- Expand the pipeline beyond `CA_1` to item and store levels before making broader network-wide decisions.

## Limitations

- The current model targets aggregated daily sales for one store only.
- Aggregation hides item-level stockout, substitution, and intermittent-demand behavior.
- Future event and SNAP calendars are needed for operational machine-learning forecasts beyond observed dates.
- The final forecast is a prototype output, not a Kaggle submission.

## Environment Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run Notebooks

Open the notebooks with the `.venv` kernel and run them in order:

1. `notebooks/01_data_wrangling.ipynb`
2. `notebooks/02_eda.ipynb`
3. `notebooks/03_feature_engineering_modeling.ipynb`

## Run Scripts

```powershell
.\.venv\Scripts\python.exe src\data_pipeline.py
.\.venv\Scripts\python.exe src\modeling_pipeline.py
```

## Final Report

- Final supervisor report: [reports/M5_Sales_Forecasting_Final_Report.pdf](reports/M5_Sales_Forecasting_Final_Report.pdf)

## Expected Generated Outputs

- `data/processed/ca1_sales_long.parquet`
- `outputs/metrics/model_comparison.csv`
- `outputs/metrics/model_comparison_summary.csv`
- `outputs/metrics/model_evaluation_summary.md`
- `outputs/forecasts/ca1_28_day_forecast.csv`
- EDA and modeling PNG files in `outputs/figures/`

## Git and Data Notes

The repository is configured to ignore raw datasets, processed datasets, virtual environments, notebook checkpoints, model binaries, and Kaggle credential files. Do not commit raw M5 CSVs, processed Parquet files, or credentials.
