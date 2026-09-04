"""Build the CA_1 transformed sales dataset for the M5 project."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILES = {
    "calendar": "calendar.csv",
    "prices": "sell_prices.csv",
    "sales": "sales_train_validation.csv",
}


def get_project_root() -> Path:
    """Return the repository root based on this script location."""
    return Path(__file__).resolve().parents[1]


def require_raw_files(raw_dir: Path) -> dict[str, Path]:
    """Return raw file paths, failing clearly if any required file is missing."""
    paths = {name: raw_dir / filename for name, filename in RAW_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required raw files: " + ", ".join(missing))
    return paths


def load_raw_data(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load calendar, price, and validation sales data."""
    print("Loading raw M5 files...")
    calendar = pd.read_csv(paths["calendar"])
    prices = pd.read_csv(paths["prices"])
    sales = pd.read_csv(paths["sales"])
    return calendar, prices, sales


def validate_inputs(calendar: pd.DataFrame, prices: pd.DataFrame, sales: pd.DataFrame) -> list[str]:
    """Validate basic quality conditions before transformation."""
    messages: list[str] = []
    day_columns = [column for column in sales.columns if column.startswith("d_")]
    if not day_columns:
        raise ValueError("No sales day columns beginning with 'd_' were found.")

    sales_numeric = all(pd.api.types.is_numeric_dtype(sales[column]) for column in day_columns)
    sales_non_negative = sales[day_columns].min().min() >= 0
    price_keys_unique = not prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any()
    calendar_keys_unique = not calendar.duplicated(["d"]).any()

    if not sales_numeric:
        raise TypeError("One or more sales day columns are not numeric.")
    if not sales_non_negative:
        raise ValueError("Negative sales values were found.")
    if not price_keys_unique:
        raise ValueError("sell_prices contains duplicate store_id + item_id + wm_yr_wk keys.")
    if not calendar_keys_unique:
        raise ValueError("calendar contains duplicate d keys.")

    event_columns = [column for column in calendar.columns if column.startswith("event_")]
    messages.append(f"Sales day columns: {len(day_columns)}")
    messages.append(f"Missing event values: {int(calendar[event_columns].isna().sum().sum()) if event_columns else 0}")
    messages.append(f"Missing sell prices in price table: {int(prices['sell_price'].isna().sum())}")
    return messages


def build_ca1_long(calendar: pd.DataFrame, prices: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """Transform CA_1 sales to long format and merge calendar and price data."""
    print("Preparing CA_1 prototype...")
    calendar = calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"])

    identifier_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [column for column in sales.columns if column.startswith("d_")]

    ca1_sales_wide = sales.loc[
        sales["store_id"].eq("CA_1"),
        identifier_columns + day_columns,
    ].copy()
    print(f"CA_1 wide rows: {ca1_sales_wide.shape[0]:,}")

    ca1_sales_long = ca1_sales_wide.melt(
        id_vars=identifier_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )
    del ca1_sales_wide
    gc.collect()

    rows_before_calendar = ca1_sales_long.shape[0]
    ca1_sales_calendar = ca1_sales_long.merge(calendar, on="d", how="left", validate="many_to_one")
    del ca1_sales_long
    gc.collect()
    if ca1_sales_calendar.shape[0] != rows_before_calendar:
        raise ValueError("Calendar merge changed row count.")

    rows_before_prices = ca1_sales_calendar.shape[0]
    ca1_sales_merged = ca1_sales_calendar.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
        validate="many_to_one",
    )
    del ca1_sales_calendar
    gc.collect()
    if ca1_sales_merged.shape[0] != rows_before_prices:
        raise ValueError("Price merge changed row count.")

    ca1_sales_merged["price_available"] = ca1_sales_merged["sell_price"].notna().astype(np.int8)
    ca1_sales_merged = ca1_sales_merged.sort_values(["item_id", "date"]).reset_index(drop=True)

    duplicate_item_dates = ca1_sales_merged.duplicated(["item_id", "date"]).sum()
    if duplicate_item_dates:
        raise ValueError(f"Found {duplicate_item_dates:,} duplicate item_id + date rows.")
    if not ca1_sales_merged["store_id"].eq("CA_1").all():
        raise ValueError("Non-CA_1 rows were found in the prototype output.")

    print(f"CA_1 long rows: {ca1_sales_merged.shape[0]:,}")
    return ca1_sales_merged


def save_processed_data(data: pd.DataFrame, output_path: Path) -> Path:
    """Save the transformed CA_1 data as Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    return output_path


def main() -> None:
    """Run the CA_1 data transformation pipeline."""
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    output_path = project_root / "data" / "processed" / "ca1_sales_long.parquet"

    paths = require_raw_files(raw_dir)
    calendar, prices, sales = load_raw_data(paths)
    for message in validate_inputs(calendar, prices, sales):
        print(message)
    ca1_data = build_ca1_long(calendar, prices, sales)
    save_processed_data(ca1_data, output_path)


if __name__ == "__main__":
    main()
