# Presentation Script: M5 Walmart Sales Forecasting

## Slide 1: M5 Walmart Sales Forecasting

Good morning. Today I will present my M5 Walmart Sales Forecasting project. This is an end-to-end time-series forecasting workflow, starting from raw sales data and moving through data wrangling, exploratory analysis, feature engineering, model validation, and a final 28-day forecast. The goal is not just to build a model, but to show a reproducible process that can support retail planning decisions. I focused on store CA_1 as a manageable prototype, so the results are specific to that store and should not be treated as an all-store M5 forecast.

Transition: I will start with the business problem and the scope of the project.

## Slide 2: Business Problem and Scope

The business problem is to predict the next 28 days of sales so the business can plan inventory and staffing more effectively. The M5 dataset includes sales history, calendar information, events, SNAP indicators, and prices. For this project, I modeled CA_1 only, and the target was total daily store sales. The data period used for modeling runs from January 29, 2011 through April 24, 2016, giving 1,913 daily observations. This scope keeps the project realistic and reproducible while still demonstrating the complete forecasting workflow.

Transition: Next, I will show how the data moved through the pipeline.

## Slide 3: Data Pipeline

The pipeline begins with the raw M5 files, then moves into data wrangling, exploratory data analysis, feature engineering, model comparison, and the final 28-day forecast. During wrangling, I transformed CA_1 item-level sales into a long-format dataset with 5,832,737 item-day records. For modeling, those item-level records were aggregated into total daily store sales. Processing stayed chronological throughout, and I avoided random train-test splitting because that would not reflect a real forecasting situation. I also used Parquet for efficient storage and type preservation.

Transition: Once the data was prepared, I explored the main sales patterns.

## Slide 4: Key EDA Findings

The EDA showed several patterns that are relevant for forecasting. Weekly seasonality was clear: Sunday had the highest average daily sales, while Wednesday had the lowest. Product mix also mattered. FOODS was the highest-sales category, and FOODS_3 was the highest-sales department. SNAP-active days had higher average sales than inactive days. I also found that 63.95% of item-day records had zero sales, which shows substantial intermittent demand at the item level. These are observed associations from the data, not causal claims.

Transition: Those patterns informed the model features and validation design.

## Slide 5: Models and Validation

I compared five approaches: a naive forecast, seasonal naive with a 7-day lag, seasonal naive with a 28-day lag, Holt-Winters exponential smoothing, and Random Forest. Every model used the same three rolling-origin validation folds, each with a 28-day forecast horizon. This keeps the comparison fair and prevents future-data leakage. The metrics were MAE, RMSE, and RMSSE, where lower values are better. RMSSE is especially useful because it scales errors relative to historical changes in the training data.

Transition: Now I will summarize the model performance and forecast output.

## Slide 6: Model Performance and Forecast

Random Forest performed best on validation, with average MAE of 266.02, RMSE of 329.87, and RMSSE of 0.3821. Holt-Winters was the second-best model, with RMSSE of 0.4370. The strongest baseline was Seasonal Naive 28, with RMSSE of 0.5696. Random Forest improved RMSSE by 32.91% compared with that strongest baseline. For the final exported 28-day forecast, the project uses Holt-Winters because it can forecast beyond the observed period without needing future SNAP or event values, while the Random Forest recursive setup depends on those future inputs.

Transition: I will close with recommendations and limitations.

## Slide 7: Recommendations and Limitations

The main recommendation is to use the CA_1 forecast for short-term inventory planning and staffing decisions, especially around stronger weekend demand. The business should also monitor SNAP and event periods separately because they behave differently in the data. High-volume food departments deserve particular attention. At the same time, this is only a CA_1 aggregate store-level model. It does not capture item-level demand, substitutions, or stockouts. Before using this for network-wide decisions, the pipeline should be expanded to all stores and products and monitored regularly for forecast error.

Transition: That completes the presentation. I am happy to discuss the methodology, validation choices, or how this could be scaled.

## Likely Supervisor Questions and Concise Answers

1. Why did you model only CA_1?
   CA_1 was selected as a manageable prototype so the full pipeline could be built and validated before scaling to all stores.

2. Why not use a random train-test split?
   Random splitting would leak future patterns into training. A time-series forecast must train on earlier dates and validate on later dates.

3. Why was Random Forest best but Holt-Winters used for the final forecast?
   Random Forest was best during validation, but future event and SNAP values are unavailable in the current final-forecast setup. Holt-Winters can produce a true future forecast from the sales series alone.

4. Does the project prove that SNAP or events cause sales changes?
   No. The analysis shows associations only. Causal claims would require a different study design.

5. What is the most important next step?
   Expand the pipeline to item-store level forecasts and test it across all stores before using it for broader operational decisions.
