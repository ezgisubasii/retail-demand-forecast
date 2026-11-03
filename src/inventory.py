#Stock level recommendation: how much stock covers the next 14 days with a target service level
#Compares the classic safety stock formula with a LightGBM quantile model
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest import DATA, make_cutoffs
from train_lgbm import load_tables

warnings.filterwarnings("ignore", category=RuntimeWarning)  #mean of empty windows

LEAD_TIME = 7       #days until an order arrives
REVIEW_PERIOD = 7   #we place an order once a week
PERIOD = LEAD_TIME + REVIEW_PERIOD  #today's order has to cover 14 days
SERVICE_LEVELS = [0.80, 0.90, 0.95]

FEATURES = [
    "mean_7", "mean_28", "mean_56", "mean_112",
    "std_28", "zero_share_28",
    "price_vs_avg", "snap_days", "event_days",
]


def stock_features_at(day: int, sales, price, calendar) -> pd.DataFrame:
    """One row per item: what we know on `day` and the real demand of the next 14 days."""
    s = sales.values
    p = price.values
    future = s[:, day + 1: day + PERIOD + 1]

    table = pd.DataFrame({
        "id": sales.index,
        "mean_7": np.nanmean(s[:, day - 6: day + 1], axis=1),
        "mean_28": np.nanmean(s[:, day - 27: day + 1], axis=1),
        "mean_56": np.nanmean(s[:, day - 55: day + 1], axis=1),
        "mean_112": np.nanmean(s[:, day - 111: day + 1], axis=1),
        "std_28": np.nanstd(s[:, day - 27: day + 1], axis=1),
        "zero_share_28": np.nanmean(s[:, day - 27: day + 1] == 0, axis=1),
        #planned price of the next 14 days compared to the last 112 days
        "price_vs_avg": np.nanmean(p[:, day + 1: day + PERIOD + 1], axis=1)
                        / np.nanmean(p[:, day - 111: day + 1], axis=1),
        "snap_days": calendar["snap_CA"].iloc[day + 1: day + PERIOD + 1].sum(),
        "event_days": calendar["event_name_1"].iloc[day + 1: day + PERIOD + 1].notna().sum(),
        "demand": np.nansum(future, axis=1),
    })
    #Keep items that were on the shelf for all 14 days
    on_shelf = np.isfinite(future).all(axis=1)
    return table[on_shelf]


def make_stock_training_set(cutoff_day: int, sales, price, calendar) -> pd.DataFrame:
    """Repeat the situation every 7 days in the past; no target goes past the cutoff."""
    tables = []
    for day in range(cutoff_day - PERIOD, 112, -7):
        tables.append(stock_features_at(day, sales, price, calendar))
    return pd.concat(tables)


def train_stock_model(train: pd.DataFrame, service_level: float) -> lgb.LGBMRegressor:
    """Quantile model: predicts the stock that is enough in `service_level` of the cases."""
    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=service_level,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        verbose=-1,
    )
    model.fit(train[FEATURES], train["demand"])
    return model


def classic_stock_level(table: pd.DataFrame, z: float) -> pd.Series:
    """Textbook formula: expected demand + z * daily std * sqrt(days)."""
    expected = table["mean_28"] * PERIOD
    safety_stock = z * table["std_28"] * np.sqrt(PERIOD)
    return expected + safety_stock


def evaluate(demand: pd.Series, stock_level) -> dict:
    """How often the stock was enough, and how much extra stock we kept."""
    demand = np.asarray(demand)
    stock_level = np.asarray(stock_level)
    return {
        "covered": round(float(np.mean(demand <= stock_level)), 3),
        "excess_stock": round(float(np.clip(stock_level - demand, 0, None).sum() / demand.sum()), 3),
    }


def main():
    sales, price, calendar = load_tables()
    df = pd.read_parquet(DATA)
    cutoffs = make_cutoffs(df[df["sell_price"].notna()])
    results = []

    for cutoff in cutoffs:
        cutoff_day = sales.columns.get_loc(cutoff)
        train = make_stock_training_set(cutoff_day, sales, price, calendar)
        test = stock_features_at(cutoff_day, sales, price, calendar)

        #Classic formula with z = 1.645 (should give 95%) and z = 3.0
        for z in [1.645, 3.0]:
            stock_level = classic_stock_level(test, z)
            results.append({"cutoff": cutoff.date(), "method": f"classic z={z}",
                            **evaluate(test["demand"], stock_level)})

        #LightGBM quantile model for each target service level
        for service_level in SERVICE_LEVELS:
            model = train_stock_model(train, service_level)
            stock_level = model.predict(test[FEATURES])
            results.append({"cutoff": cutoff.date(), "method": f"lgbm target={service_level:.0%}",
                            **evaluate(test["demand"], stock_level)})

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    print("Average over the 3 periods:")
    print(results_df.groupby("method", sort=False)[["covered", "excess_stock"]].mean().round(3))


if __name__ == "__main__":
    main()
