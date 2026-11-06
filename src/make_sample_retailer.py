#I wanted to see how onboarding copes with data that feels like it came from a real customer, so this script takes one store (TX_1)
#and one department (FOODS_1) from M5 and roughs it up: their own column names, their own formats, and the kind of problems real exports have
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "onboarding" / "retailer_x"

STORE = "TX_1"
DEPT = "FOODS_1"
START_DATE = "2014-05-23"  #two years of history: plenty for the model without making the files huge


def main():
    rng = np.random.default_rng(42)

    #Start from the clean M5 sales and turn them into one row per item and day
    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    sales = sales[(sales["store_id"] == STORE) & (sales["dept_id"] == DEPT)]
    long = sales.melt(id_vars=["item_id", "store_id"], value_vars=[c for c in sales.columns if c.startswith("d_")],
                      var_name="d", value_name="units")
    calendar = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])[["d", "date", "wm_yr_wk"]]
    long = long.merge(calendar, on="d")
    long = long[long["date"] >= START_DATE]

    #The customer sends prices per week, so each price gets the date its week starts on
    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[(prices["store_id"] == STORE) & prices["item_id"].str.startswith(DEPT)]
    week_start = calendar.groupby("wm_yr_wk")["date"].min().rename("week_start")
    prices = prices.merge(week_start, on="wm_yr_wk")
    prices = prices[prices["week_start"] >= START_DATE]

    #Problem 1: their export job fell over on 3 days, so those days are simply gone
    missing_days = pd.to_datetime(["2015-03-14", "2015-11-02", "2016-02-19"])
    long = long[~long["date"].isin(missing_days)]

    #Problem 2: returns ended up in the sales file as negative units
    returns = long[long["units"] > 0].sample(25, random_state=1).index
    long.loc[returns, "units"] = -long.loc[returns, "units"]

    #Problem 3: some rows got exported twice
    long = pd.concat([long, long.sample(60, random_state=2)])

    #Problem 4: 12 items never made it into the price list
    no_price_items = rng.choice(prices["item_id"].unique(), 12, replace=False)
    prices = prices[~prices["item_id"].isin(no_price_items)]

    #Problem 5: someone typed 0 for a few prices
    zero_rows = prices.sample(8, random_state=3).index
    prices.loc[zero_rows, "sell_price"] = 0.0

    #Now dress it up the way this customer would: their column names, day.month.year dates and decimal commas in prices
    sales_export = pd.DataFrame({
        "SKU": long["item_id"].str.replace("FOODS_1_", "F1-"),
        "Store": "Houston-01",
        "Day": long["date"].dt.strftime("%d.%m.%Y"),
        "Units": long["units"],
    }).sample(frac=1, random_state=4)  #real exports are almost never sorted

    price_export = pd.DataFrame({
        "SKU": prices["item_id"].str.replace("FOODS_1_", "F1-"),
        "Store": "Houston-01",
        "WeekStart": prices["week_start"].dt.strftime("%Y-%m-%d"),
        "Price": prices["sell_price"].map(lambda p: f"{p:.2f}".replace(".", ",")),
    })

    OUT.mkdir(parents=True, exist_ok=True)
    sales_export.to_csv(OUT / "transactions_export.csv", index=False, sep=";")
    price_export.to_csv(OUT / "price_list.csv", index=False, sep=";")
    print(f"transactions_export.csv: {len(sales_export):,} rows")
    print(f"price_list.csv: {len(price_export):,} rows")


if __name__ == "__main__":
    main()
