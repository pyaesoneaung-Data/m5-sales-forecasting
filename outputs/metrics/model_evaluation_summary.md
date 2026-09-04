# Model Evaluation Summary

## Forecasting Objective
Forecast total daily unit sales for Walmart M5 store CA_1 over a 28-day horizon.

## Scope
This is a CA_1 store-level prototype using aggregated daily sales. It is not an official all-store M5 competition forecast.

## Validation Method
Three rolling-origin validation folds were used. Validation dates always occur after training dates, and no random split was used.

- Fold 1: train 2011-01-29 to 2016-01-31, validate 2016-02-01 to 2016-02-28
- Fold 2: train 2011-01-29 to 2016-02-28, validate 2016-02-29 to 2016-03-27
- Fold 3: train 2011-01-29 to 2016-03-27, validate 2016-03-28 to 2016-04-24

## Models Compared
- Naive
- Seasonal Naive 7
- Seasonal Naive 28
- Holt-Winters
- Random Forest

## Metric Definitions
- MAE: average absolute forecast error.
- RMSE: square root of average squared forecast error.
- RMSSE: scaled RMSE using only each fold's training data for the denominator.

## Actual Model Results
- Random Forest: MAE 266.02, RMSE 329.87, RMSSE 0.3821
- Holt-Winters: MAE 284.71, RMSE 377.23, RMSSE 0.4370
- Seasonal Naive 28: MAE 379.76, RMSE 491.70, RMSSE 0.5696
- Seasonal Naive 7: MAE 457.65, RMSE 589.61, RMSSE 0.6828
- Naive: MAE 1438.65, RMSE 1588.99, RMSSE 1.8415

## Best-Performing Model
Random Forest performed best with average RMSSE 0.3821.

## Improvement Over Strongest Baseline
The strongest baseline was Seasonal Naive 28 with average RMSSE 0.5696. The best model improved RMSSE by 32.91%.

## Important Features or Seasonal Patterns
- lag_28: 0.5538
- lag_7: 0.1880
- lag_14: 0.0643
- lag_1: 0.0448
- rolling_mean_7: 0.0323
- day_of_week: 0.0242
- rolling_mean_28: 0.0178
- snap_CA: 0.0152

## Limitations
- The workflow models only CA_1 aggregated store sales.
- Item-level differences are hidden by aggregation.
- Future event and SNAP inputs are needed to deploy recursive ML forecasts beyond observed dates.

## Recommended Next Steps
- Expand validation to item and store levels.
- Add future calendar inputs for operational ML forecasting.
- Monitor forecast error by weekday, event period, and demand level.
- Compare additional models after the leakage-safe pipeline is stable.
