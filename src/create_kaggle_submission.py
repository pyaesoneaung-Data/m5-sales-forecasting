"""Create an honest Seasonal Naive 28 Kaggle submission for M5.

This script builds item-store forecasts in the exact M5 sample-submission
format. It does not use aggregate CA_1 model outputs and does not submit to
Kaggle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FORECAST_COLUMNS = [f"F{i}" for i in range(1, 29)]
ID_COLUMN = "id"


def get_project_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def require_files(paths: list[Path]) -> None:
    """Fail clearly if any required input file is missing."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files: " + ", ".join(missing))


def read_inputs(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read validation sales, evaluation sales, and the sample submission."""
    raw_dir = project_root / "data" / "raw"
    validation_path = raw_dir / "sales_train_validation.csv"
    evaluation_path = raw_dir / "sales_train_evaluation.csv"
    sample_path = raw_dir / "sample_submission.csv"

    require_files([validation_path, evaluation_path, sample_path])

    print("Reading raw M5 files...")
    validation_sales = pd.read_csv(validation_path)
    evaluation_sales = pd.read_csv(evaluation_path)
    sample_submission = pd.read_csv(sample_path)
    return validation_sales, evaluation_sales, sample_submission


def last_28_day_columns(sales: pd.DataFrame) -> list[str]:
    """Return the final 28 daily sales columns from a wide M5 sales table."""
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    if len(day_columns) < 28:
        raise ValueError("Sales table must contain at least 28 day columns.")
    return day_columns[-28:]


def create_seasonal_naive_forecast(sales: pd.DataFrame, id_suffix: str) -> pd.DataFrame:
    """Create a Seasonal Naive 28 forecast for one M5 split.

    The final 28 observed daily sales values are reused as the next 28-day
    forecast for each item-store series.
    """
    forecast_source_columns = last_28_day_columns(sales)
    forecast = sales[[ID_COLUMN, *forecast_source_columns]].copy()
    forecast[ID_COLUMN] = forecast[ID_COLUMN].str.replace(
        "_validation",
        id_suffix,
        regex=False,
    )
    forecast = forecast.rename(
        columns=dict(zip(forecast_source_columns, FORECAST_COLUMNS, strict=True))
    )
    return forecast[[ID_COLUMN, *FORECAST_COLUMNS]]


def align_to_sample_submission(
    sample_submission: pd.DataFrame,
    validation_forecast: pd.DataFrame,
    evaluation_forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Combine forecasts and align them to sample-submission row order."""
    combined_forecast = pd.concat(
        [validation_forecast, evaluation_forecast],
        ignore_index=True,
    )
    if combined_forecast[ID_COLUMN].duplicated().any():
        duplicate_count = int(combined_forecast[ID_COLUMN].duplicated().sum())
        raise ValueError(f"Combined forecast contains {duplicate_count} duplicate IDs.")

    aligned = sample_submission[[ID_COLUMN]].merge(
        combined_forecast,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )
    return aligned


def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> dict[str, object]:
    """Validate the generated submission against M5 format requirements."""
    expected_columns = [ID_COLUMN, *FORECAST_COLUMNS]
    prediction_values = submission[FORECAST_COLUMNS]

    checks = {
        "identical_row_count": len(submission) == len(sample_submission),
        "exact_columns": list(submission.columns) == expected_columns,
        "no_missing_ids": submission[ID_COLUMN].notna().all(),
        "no_duplicate_ids": not submission[ID_COLUMN].duplicated().any(),
        "no_nan_predictions": not prediction_values.isna().any().any(),
        "all_predictions_numeric": all(
            pd.api.types.is_numeric_dtype(prediction_values[column])
            for column in FORECAST_COLUMNS
        ),
        "all_predictions_non_negative": bool((prediction_values.to_numpy() >= 0).all()),
        "every_sample_id_matched_once": submission[ID_COLUMN].equals(sample_submission[ID_COLUMN]),
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Submission validation failed: " + ", ".join(failed))
    return checks


def save_submission(submission: pd.DataFrame, output_path: Path) -> Path:
    """Save the validated submission CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return output_path


def print_submission_report(submission: pd.DataFrame, output_path: Path) -> None:
    """Print a concise validation and output summary."""
    predictions = submission[FORECAST_COLUMNS]
    print(f"Output shape: {submission.shape}")
    print(f"Column count: {submission.shape[1]}")
    print(f"Missing-value count: {int(submission.isna().sum().sum())}")
    print(f"Duplicate-ID count: {int(submission[ID_COLUMN].duplicated().sum())}")
    print(f"Minimum prediction: {float(np.nanmin(predictions.to_numpy()))}")
    print(f"Maximum prediction: {float(np.nanmax(predictions.to_numpy()))}")
    print(f"File size: {output_path.stat().st_size / (1024 ** 2):.2f} MB")
    print("First five rows:")
    print(submission.head().to_string(index=False))


def main() -> None:
    """Create and validate the M5 Seasonal Naive 28 submission file."""
    project_root = get_project_root()
    output_path = project_root / "outputs" / "forecasts" / "m5_seasonal_naive_28_submission.csv"

    validation_sales, evaluation_sales, sample_submission = read_inputs(project_root)
    validation_forecast = create_seasonal_naive_forecast(validation_sales, "_validation")
    evaluation_forecast = create_seasonal_naive_forecast(evaluation_sales, "_evaluation")
    submission = align_to_sample_submission(
        sample_submission,
        validation_forecast,
        evaluation_forecast,
    )

    validate_submission(submission, sample_submission)
    save_submission(submission, output_path)
    print_submission_report(submission, output_path)
    print("Submission validation passed")


if __name__ == "__main__":
    main()
