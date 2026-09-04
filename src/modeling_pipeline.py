"""Run leakage-safe store-level forecasting models for CA_1 total daily sales."""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

HORIZON = 28
N_FOLDS = 3
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_28",
    "day_of_week",
    "month",
    "year",
    "is_weekend",
    "event_indicator",
    "snap_CA",
]


def get_project_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def load_daily_sales(data_path: Path) -> pd.DataFrame:
    """Load CA_1 transformed data and aggregate to total daily sales."""
    if not data_path.exists():
        raise FileNotFoundError(f"Missing processed dataset: {data_path}")

    print("Loading transformed CA_1 dataset...")
    raw_df = pd.read_parquet(data_path)
    raw_df["date"] = pd.to_datetime(raw_df["date"])
    raw_df["event_indicator"] = raw_df[["event_name_1", "event_name_2"]].notna().any(axis=1).astype(int)

    daily_sales = (
        raw_df.groupby("date", as_index=False)
        .agg(
            sales=("sales", "sum"),
            event_indicator=("event_indicator", "max"),
            snap_CA=("snap_CA", "max"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )
    expected_dates = pd.date_range(daily_sales["date"].min(), daily_sales["date"].max(), freq="D")
    if len(expected_dates.difference(daily_sales["date"])) > 0:
        raise ValueError("Daily sales series contains missing dates.")
    if daily_sales["date"].duplicated().any():
        raise ValueError("Daily sales series contains duplicate dates.")

    print(f"Daily observations: {len(daily_sales):,}")
    print(f"Date range: {daily_sales['date'].min().date()} to {daily_sales['date'].max().date()}")
    return daily_sales


def create_rolling_folds(data: pd.DataFrame, horizon: int = HORIZON, n_folds: int = N_FOLDS) -> list[dict]:
    """Create rolling-origin validation folds with validation after training."""
    if len(data) < horizon * (n_folds + 1):
        n_folds = max(1, len(data) // horizon - 1)
    folds = []
    for fold_number in range(n_folds, 0, -1):
        val_start_idx = len(data) - horizon * fold_number
        val_end_idx = val_start_idx + horizon
        folds.append(
            {
                "fold": len(folds) + 1,
                "train": data.iloc[:val_start_idx].copy(),
                "valid": data.iloc[val_start_idx:val_end_idx].copy(),
            }
        )
    return folds


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return root mean squared error."""
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmsse(y_true: np.ndarray, y_pred: np.ndarray, train_values: pd.Series) -> float:
    """Return RMSSE using a denominator calculated from training data only."""
    diffs = train_values.diff(1).dropna()
    denominator = np.mean(np.square(diffs))
    if denominator <= 0:
        return np.nan
    return np.sqrt(np.mean(np.square(y_true - y_pred)) / denominator)


def evaluate_forecast(model_name: str, fold_number: int, train: pd.DataFrame, valid: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Calculate metrics for one validation forecast."""
    y_true = valid["sales"].to_numpy()
    y_pred = np.clip(np.asarray(predictions, dtype=float), 0, None)
    return {
        "Model": model_name,
        "Fold": fold_number,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "RMSSE": rmsse(y_true, y_pred, train["sales"]),
    }


def naive_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Forecast using the most recent observed value."""
    return np.repeat(train["sales"].iloc[-1], horizon)


def seasonal_naive_forecast(train: pd.DataFrame, horizon: int, seasonality: int) -> np.ndarray:
    """Forecast using the most recent seasonal cycle."""
    history = train["sales"].to_numpy()
    return np.array([history[-seasonality + (step % seasonality)] for step in range(horizon)])


def holt_winters_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Forecast with additive Holt-Winters trend and weekly seasonality."""
    series = train.set_index("date")["sales"].asfreq("D")
    model = ExponentialSmoothing(
        series,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    )
    return model.fit(optimized=True).forecast(horizon).to_numpy()


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add date-derived model features."""
    result = data.copy()
    result["day_of_week"] = result["date"].dt.dayofweek
    result["month"] = result["date"].dt.month
    result["year"] = result["date"].dt.year
    result["is_weekend"] = result["day_of_week"].isin([5, 6]).astype(int)
    return result


def make_supervised_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe lag and shifted rolling features."""
    result = add_calendar_features(data).sort_values("date").reset_index(drop=True)
    shifted_sales = result["sales"].shift(1)
    for lag in [1, 7, 14, 28]:
        result[f"lag_{lag}"] = result["sales"].shift(lag)
    for window in [7, 28]:
        result[f"rolling_mean_{window}"] = shifted_sales.rolling(window).mean()
        result[f"rolling_std_{window}"] = shifted_sales.rolling(window).std()
    return result.dropna(subset=FEATURE_COLUMNS + ["sales"]).reset_index(drop=True)


def make_random_forest() -> RandomForestRegressor:
    """Create a reproducible Random Forest model with manageable runtime."""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def feature_row_from_history(history: list[float], calendar_row: pd.Series) -> dict:
    """Build one recursive forecast feature row from available history only."""
    return {
        "lag_1": history[-1],
        "lag_7": history[-7],
        "lag_14": history[-14],
        "lag_28": history[-28],
        "rolling_mean_7": float(np.mean(history[-7:])),
        "rolling_mean_28": float(np.mean(history[-28:])),
        "rolling_std_7": float(np.std(history[-7:], ddof=1)),
        "rolling_std_28": float(np.std(history[-28:], ddof=1)),
        "day_of_week": calendar_row["date"].dayofweek,
        "month": calendar_row["date"].month,
        "year": calendar_row["date"].year,
        "is_weekend": int(calendar_row["date"].dayofweek in [5, 6]),
        "event_indicator": int(calendar_row["event_indicator"]),
        "snap_CA": int(calendar_row["snap_CA"]),
    }


def random_forest_recursive_forecast(train: pd.DataFrame, future_calendar: pd.DataFrame, horizon: int) -> tuple[np.ndarray, RandomForestRegressor]:
    """Forecast recursively without inserting actual validation sales into history."""
    train_features = make_supervised_features(train)
    model = make_random_forest()
    model.fit(train_features[FEATURE_COLUMNS], train_features["sales"])

    history = train["sales"].astype(float).tolist()
    predictions = []
    for _, calendar_row in future_calendar.head(horizon).iterrows():
        feature_row = pd.DataFrame([feature_row_from_history(history, calendar_row)], columns=FEATURE_COLUMNS)
        prediction = max(float(model.predict(feature_row)[0]), 0.0)
        predictions.append(prediction)
        history.append(prediction)
    return np.array(predictions), model


def compare_models(daily_sales: pd.DataFrame, folds: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run validation forecasts and return fold metrics, summaries, and predictions."""
    metric_rows = []
    prediction_frames = []
    baseline_models = {
        "Naive": lambda train, horizon: naive_forecast(train, horizon),
        "Seasonal Naive 7": lambda train, horizon: seasonal_naive_forecast(train, horizon, 7),
        "Seasonal Naive 28": lambda train, horizon: seasonal_naive_forecast(train, horizon, 28),
    }

    for fold in folds:
        fold_number = fold["fold"]
        train = fold["train"]
        valid = fold["valid"]
        horizon = valid.shape[0]
        print(f"Evaluating fold {fold_number}...")

        for model_name, forecast_function in baseline_models.items():
            predictions = forecast_function(train, horizon)
            metric_rows.append(evaluate_forecast(model_name, fold_number, train, valid, predictions))
            prediction_frames.append(valid[["date", "sales"]].assign(Model=model_name, Fold=fold_number, predicted_sales=np.clip(predictions, 0, None)))

        hw_predictions = holt_winters_forecast(train, horizon)
        metric_rows.append(evaluate_forecast("Holt-Winters", fold_number, train, valid, hw_predictions))
        prediction_frames.append(valid[["date", "sales"]].assign(Model="Holt-Winters", Fold=fold_number, predicted_sales=np.clip(hw_predictions, 0, None)))

        rf_predictions, _ = random_forest_recursive_forecast(train, valid[["date", "event_indicator", "snap_CA"]], horizon)
        metric_rows.append(evaluate_forecast("Random Forest", fold_number, train, valid, rf_predictions))
        prediction_frames.append(valid[["date", "sales"]].assign(Model="Random Forest", Fold=fold_number, predicted_sales=rf_predictions))

    metrics_table = pd.DataFrame(metric_rows).sort_values(["Fold", "RMSSE"]).reset_index(drop=True)
    predictions_table = pd.concat(prediction_frames, ignore_index=True)
    predictions_table["residual"] = predictions_table["sales"] - predictions_table["predicted_sales"]
    summary_table = (
        metrics_table.groupby("Model", as_index=False)
        .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), RMSSE=("RMSSE", "mean"))
        .sort_values("RMSSE")
        .reset_index(drop=True)
    )
    return metrics_table, summary_table, predictions_table


def save_model_figures(predictions_table: pd.DataFrame, summary_table: pd.DataFrame, feature_importance: pd.DataFrame, figures_dir: Path) -> None:
    """Save essential modeling figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    final_fold = predictions_table["Fold"].max()
    final_fold_predictions = predictions_table[predictions_table["Fold"].eq(final_fold)]
    best_model = summary_table.iloc[0]["Model"]

    fig, ax = plt.subplots(figsize=(12, 5))
    actual_final = final_fold_predictions.drop_duplicates("date")[["date", "sales"]]
    ax.plot(actual_final["date"], actual_final["sales"], label="Actual", color="black", linewidth=2)
    for model_name, model_data in final_fold_predictions.groupby("Model"):
        ax.plot(model_data["date"], model_data["predicted_sales"], label=model_name, linewidth=1.5)
    ax.set_title("Actual vs Predicted Sales, Final Validation Fold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total daily sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.savefig(figures_dir / "model_actual_vs_predicted_final_fold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=summary_table, x="Model", y="RMSSE", color="#2F6F9F", ax=ax)
    ax.set_title("Average RMSSE by Model")
    ax.set_xlabel("Model")
    ax.set_ylabel("Average RMSSE")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(figures_dir / "model_rmsse_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_residuals = predictions_table[predictions_table["Model"].eq(best_model)]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0, color="black", linewidth=1)
    ax.scatter(best_residuals["date"], best_residuals["residual"], color="#D98E04", alpha=0.8)
    ax.set_title(f"Residuals for Best Model: {best_model}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Actual - predicted sales")
    fig.autofmt_xdate()
    fig.savefig(figures_dir / "model_best_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=feature_importance.head(10).sort_values("importance"), x="importance", y="feature", color="#2F6F9F", ax=ax)
    ax.set_title("Random Forest Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.savefig(figures_dir / "model_random_forest_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_feature_importance_model(daily_sales: pd.DataFrame) -> tuple[RandomForestRegressor, pd.DataFrame]:
    """Train Random Forest on all supervised rows and return feature importances."""
    training = make_supervised_features(daily_sales)
    model = make_random_forest()
    model.fit(training[FEATURE_COLUMNS], training["sales"])
    importance = (
        pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return model, importance


def final_forecast(daily_sales: pd.DataFrame, summary_table: pd.DataFrame) -> pd.DataFrame:
    """Create the final 28-day non-negative forecast with a model label."""
    best_model = summary_table.iloc[0]["Model"]
    if best_model == "Naive":
        model_name = best_model
        predictions = naive_forecast(daily_sales, HORIZON)
    elif best_model == "Seasonal Naive 7":
        model_name = best_model
        predictions = seasonal_naive_forecast(daily_sales, HORIZON, 7)
    elif best_model == "Seasonal Naive 28":
        model_name = best_model
        predictions = seasonal_naive_forecast(daily_sales, HORIZON, 28)
    elif best_model == "Holt-Winters":
        model_name = best_model
        predictions = holt_winters_forecast(daily_sales, HORIZON)
    else:
        fallback = summary_table[summary_table["Model"].ne("Random Forest")].iloc[0]["Model"]
        print("Random Forest was best, but future event and SNAP values are unavailable for recursive future forecasting.")
        print(f"Using strongest directly forecastable model for final forecast: {fallback}")
        model_name = fallback
        if fallback == "Holt-Winters":
            predictions = holt_winters_forecast(daily_sales, HORIZON)
        elif fallback == "Naive":
            predictions = naive_forecast(daily_sales, HORIZON)
        elif fallback == "Seasonal Naive 7":
            predictions = seasonal_naive_forecast(daily_sales, HORIZON, 7)
        elif fallback == "Seasonal Naive 28":
            predictions = seasonal_naive_forecast(daily_sales, HORIZON, 28)
        else:
            raise ValueError(f"Unsupported fallback model: {fallback}")

    forecast_dates = pd.date_range(daily_sales["date"].max() + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    return pd.DataFrame(
        {
            "date": forecast_dates,
            "predicted_sales": np.clip(predictions, 0, None),
            "model": model_name,
        }
    )


def write_model_evaluation_summary(
    daily_sales: pd.DataFrame,
    folds: list[dict],
    summary_table: pd.DataFrame,
    feature_importance: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write a Markdown model evaluation summary using generated results."""
    best = summary_table.iloc[0]
    strongest_baseline = summary_table[
        summary_table["Model"].isin(["Naive", "Seasonal Naive 7", "Seasonal Naive 28"])
    ].iloc[0]
    improvement = (
        (strongest_baseline["RMSSE"] - best["RMSSE"]) / strongest_baseline["RMSSE"] * 100
        if strongest_baseline["RMSSE"] != 0
        else np.nan
    )
    fold_lines = [
        f"- Fold {fold['fold']}: train {fold['train']['date'].min().date()} to {fold['train']['date'].max().date()}, "
        f"validate {fold['valid']['date'].min().date()} to {fold['valid']['date'].max().date()}"
        for fold in folds
    ]
    model_lines = [
        f"- {row.Model}: MAE {row.MAE:.2f}, RMSE {row.RMSE:.2f}, RMSSE {row.RMSSE:.4f}"
        for row in summary_table.itertuples(index=False)
    ]
    feature_lines = [
        f"- {row.feature}: {row.importance:.4f}"
        for row in feature_importance.head(8).itertuples(index=False)
    ]

    content = "\n".join(
        [
            "# Model Evaluation Summary",
            "",
            "## Forecasting Objective",
            "Forecast total daily unit sales for Walmart M5 store CA_1 over a 28-day horizon.",
            "",
            "## Scope",
            "This is a CA_1 store-level prototype using aggregated daily sales. It is not an official all-store M5 competition forecast.",
            "",
            "## Validation Method",
            "Three rolling-origin validation folds were used. Validation dates always occur after training dates, and no random split was used.",
            "",
            *fold_lines,
            "",
            "## Models Compared",
            "- Naive",
            "- Seasonal Naive 7",
            "- Seasonal Naive 28",
            "- Holt-Winters",
            "- Random Forest",
            "",
            "## Metric Definitions",
            "- MAE: average absolute forecast error.",
            "- RMSE: square root of average squared forecast error.",
            "- RMSSE: scaled RMSE using only each fold's training data for the denominator.",
            "",
            "## Actual Model Results",
            *model_lines,
            "",
            "## Best-Performing Model",
            f"{best['Model']} performed best with average RMSSE {best['RMSSE']:.4f}.",
            "",
            "## Improvement Over Strongest Baseline",
            f"The strongest baseline was {strongest_baseline['Model']} with average RMSSE {strongest_baseline['RMSSE']:.4f}. "
            f"The best model improved RMSSE by {improvement:.2f}%.",
            "",
            "## Important Features or Seasonal Patterns",
            *feature_lines,
            "",
            "## Limitations",
            "- The workflow models only CA_1 aggregated store sales.",
            "- Item-level differences are hidden by aggregation.",
            "- Future event and SNAP inputs are needed to deploy recursive ML forecasts beyond observed dates.",
            "",
            "## Recommended Next Steps",
            "- Expand validation to item and store levels.",
            "- Add future calendar inputs for operational ML forecasting.",
            "- Monitor forecast error by weekday, event period, and demand level.",
            "- Compare additional models after the leakage-safe pipeline is stable.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    """Run model validation and final CA_1 store-level forecast."""
    project_root = get_project_root()
    data_path = project_root / "data" / "processed" / "ca1_sales_long.parquet"
    metrics_dir = project_root / "outputs" / "metrics"
    forecasts_dir = project_root / "outputs" / "forecasts"
    figures_dir = project_root / "outputs" / "figures"

    daily_sales = load_daily_sales(data_path)
    folds = create_rolling_folds(daily_sales)
    metrics_table, summary_table, predictions_table = compare_models(daily_sales, folds)
    _, feature_importance = train_feature_importance_model(daily_sales)
    forecast = final_forecast(daily_sales, summary_table)
    summary_path = write_model_evaluation_summary(
        daily_sales,
        folds,
        summary_table,
        feature_importance,
        metrics_dir / "model_evaluation_summary.md",
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(metrics_dir / "model_comparison.csv", index=False)
    summary_table.to_csv(metrics_dir / "model_comparison_summary.csv", index=False)
    forecast.to_csv(forecasts_dir / "ca1_28_day_forecast.csv", index=False)
    save_model_figures(predictions_table, summary_table, feature_importance, figures_dir)

    print("Best model by average RMSSE:", summary_table.iloc[0]["Model"])
    print(f"Model evaluation summary saved to {summary_path}")
    print("Saved model outputs.")


if __name__ == "__main__":
    main()
