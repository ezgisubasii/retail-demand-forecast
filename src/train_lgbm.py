#Backtest of a LightGBM model against the 28 days average baseline
#Every forecast only uses what is known on the cutoff day (direct multi-horizon)
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest import DATA, HORIZON, make_cutoffs, mae, wape

warnings.filterwarnings("ignore", category=RuntimeWarning)  #mean of empty windows

FEATURES = [
    "horizon", "day_of_week",
    "mean_7", "mean_28", "mean_56", "mean_112",
    "zero_share_28", "same_weekday_mean",
    "sell_price", "price_vs_avg",
    "snap_CA", "has_event",
]


def load_tables():
    """Load the data as wide tables: one row per item, one column per day."""
    df = pd.read_parquet(DATA)
    price = df.pivot(index="id", columns="date", values="sell_price")
    sales = df.pivot(index="id", columns="date", values="sales").astype(float)
    #Off-shelf days are not real zero demand, so we mark them as missing
    sales = sales.where(price.notna())
    calendar = df.drop_duplicates("date").set_index("date").loc[sales.columns]
    return sales, price, calendar


def features_at(day: int, sales, price, calendar) -> pd.DataFrame:
    """Build rows for the 28 days after `day`, using only data up to `day`."""
    s = sales.values
    p = price.values

    #What we know about each item on the cutoff day
    mean_7 = np.nanmean(s[:, day - 6: day + 1], axis=1)
    mean_28 = np.nanmean(s[:, day - 27: day + 1], axis=1)
    mean_56 = np.nanmean(s[:, day - 55: day + 1], axis=1)
    mean_112 = np.nanmean(s[:, day - 111: day + 1], axis=1)
    zero_share_28 = np.nanmean(s[:, day - 27: day + 1] == 0, axis=1)
    avg_price_112 = np.nanmean(p[:, day - 111: day + 1], axis=1)
    last_4_weeks = sales.columns[day - 27: day + 1]

    rows = []
    for horizon in range(1, HORIZON + 1):
        target = day + horizon
        if target >= s.shape[1]:
            break
        weekday = sales.columns[target].day_of_week
        #Average of the same weekday in the last 4 weeks
        same_weekday = [i for i, d in enumerate(last_4_weeks) if d.day_of_week == weekday]
        same_weekday_mean = np.nanmean(s[:, day - 27 + np.array(same_weekday)], axis=1)

        rows.append(pd.DataFrame({
            "id": sales.index,
            "date": sales.columns[target],
            "horizon": horizon,
            "day_of_week": weekday,
            "mean_7": mean_7,
            "mean_28": mean_28,
            "mean_56": mean_56,
            "mean_112": mean_112,
            "zero_share_28": zero_share_28,
            "same_weekday_mean": same_weekday_mean,
            "sell_price": p[:, target],
            "price_vs_avg": p[:, target] / avg_price_112,
            "snap_CA": calendar["snap_CA"].iloc[target],
            "has_event": int(pd.notna(calendar["event_name_1"].iloc[target])),
            "sales": s[:, target],
        }))
    table = pd.concat(rows)
    #Keep only days where the item was on the shelf
    return table[table["sales"].notna()]


def make_training_set(cutoff_day: int, sales, price, calendar) -> pd.DataFrame:
    """Repeat the cutoff situation every 7 days in the past to create training rows."""
    tables = []
    #The first origin is 28 days before the cutoff, so no target is after the cutoff
    for day in range(cutoff_day - HORIZON, 112, -7):
        tables.append(features_at(day, sales, price, calendar))
    return pd.concat(tables)


def train_model(train: pd.DataFrame) -> lgb.LGBMRegressor:
    """Train a LightGBM model on the training set."""
    model = lgb.LGBMRegressor(
        #poisson predicts the mean, so daily forecasts add up to the right weekly and store totals
        #(l1 predicts the median, which looked better on daily WAPE but was 25% too low in total)
        objective="poisson",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        verbose=-1,
    )
    model.fit(train[FEATURES], train["sales"])
    return model


def main():
    sales, price, calendar = load_tables()
    df = pd.read_parquet(DATA)
    cutoffs = make_cutoffs(df[df["sell_price"].notna()])
    results = []

    for cutoff in cutoffs:
        cutoff_day = sales.columns.get_loc(cutoff)
        train = make_training_set(cutoff_day, sales, price, calendar)
        test = features_at(cutoff_day, sales, price, calendar)

        model = train_model(train)
        test["lgbm"] = model.predict(test[FEATURES])
        #mean_28 on the cutoff day is exactly the baseline forecast
        test["baseline"] = test["mean_28"].fillna(0)

        #Same forecasts summed to item x week and to the store total per day
        test["week"] = (test["horizon"] - 1) // 7
        weekly = test.groupby(["id", "week"], observed=True)[["sales", "lgbm", "baseline"]].sum()
        store = test.groupby("date")[["sales", "lgbm", "baseline"]].sum()

        for name in ["baseline", "lgbm"]:
            results.append({
                "cutoff": cutoff.date(),
                "model": name,
                "item_day_wape": round(wape(test["sales"], test[name]), 3),
                "item_week_wape": round(wape(weekly["sales"], weekly[name]), 3),
                "store_day_wape": round(wape(store["sales"], store[name]), 3),
                "item_day_mae": round(mae(test["sales"], test[name]), 3),
                #1.0 means the total forecast equals total sales
                "total_vs_actual": round(test[name].sum() / test["sales"].sum(), 3),
            })
        print("done:", cutoff.date())

    pd.set_option("display.width", 200)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print()
    print("Average over the 3 periods:")
    print(results_df.groupby("model", sort=False).mean(numeric_only=True).round(3))

    #Which features the last model used the most
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(importance)


if __name__ == "__main__":
    main()
