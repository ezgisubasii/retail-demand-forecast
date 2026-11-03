"""Build a long-format subset of the M5 data (one store, one department).

Output is data/processed/sales.parquet with one row per item per day.
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"

STORE = "CA_1"
DEPT = "FOODS_3"
START_DATE = "2013-05-23"  # keep roughly the last 3 years


def main() -> None:
    # 1. Read the wide sales table and keep only our store and department
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    sales = sales[(sales["store_id"] == STORE) & (sales["dept_id"] == DEPT)]

    # 2. Wide -> long: one column per day becomes one row per day
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long = sales.melt(id_vars=id_cols, var_name="d", value_name="sales")

    # 3. Add the real date and calendar events
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    calendar = calendar[["d", "date", "wm_yr_wk", "event_name_1", "event_type_1", "snap_CA"]]
    long = long.merge(calendar, on="d", how="left")
    long = long[long["date"] >= START_DATE]

    # 4. Add the weekly sell price of each item in this store
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[prices["store_id"] == STORE]
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # 5. Small dtypes and save
    long = long.drop(columns=["d", "dept_id", "cat_id", "state_id"])
    for col in ["id", "item_id", "store_id", "event_name_1", "event_type_1"]:
        long[col] = long[col].astype("category")
    long["sales"] = long["sales"].astype("int16")
    long = long.sort_values(["id", "date"]).reset_index(drop=True)

    OUT.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT / "sales.parquet", index=False)
    print(f"series={long['id'].nunique()} rows={len(long):,} "
          f"dates={long['date'].min().date()}..{long['date'].max().date()}")


if __name__ == "__main__":
    main()
