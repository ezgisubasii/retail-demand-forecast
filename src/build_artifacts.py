#Train the models once and save their outputs, so the MCP server only has to read files
#In a real setup this would be a nightly batch job
import numpy as np
import pandas as pd

from backtest import DATA
from inventory import FEATURES as STOCK_FEATURES
from inventory import SERVICE_LEVELS, make_stock_training_set, stock_features_at, train_stock_model
from train_lgbm import FEATURES, features_at, load_tables, make_training_set, train_model

TODAY = pd.Timestamp("2016-04-24")  #the "current day" of the demo (last backtest cutoff)
OUT = DATA.parent


def build_forecasts(today_day: int, sales, price, calendar) -> pd.DataFrame:
    """Daily forecast for the next 28 days of every item."""
    model = train_model(make_training_set(today_day, sales, price, calendar))
    table = features_at(today_day, sales, price, calendar)
    table["forecast"] = model.predict(table[FEATURES]).clip(min=0)
    return table[["id", "date", "horizon", "forecast", "mean_28", "sell_price"]]


def build_stock_levels(today_day: int, sales, price, calendar) -> pd.DataFrame:
    """Stock level that covers the next 14 days, for every item and service level."""
    train = make_stock_training_set(today_day, sales, price, calendar)
    table = stock_features_at(today_day, sales, price, calendar)
    columns = []
    for service_level in SERVICE_LEVELS:
        model = train_stock_model(train, service_level)
        column = f"stock_{int(service_level * 100)}"
        table[column] = model.predict(table[STOCK_FEATURES]).clip(min=0)
        columns.append(column)

    #Separate quantile models can cross (90% below 80%), so force them to be increasing
    table[columns] = np.maximum.accumulate(table[columns].values, axis=1)
    return table.drop(columns=["demand"])  #the real future demand stays hidden


def main():
    sales, price, calendar = load_tables()
    today_day = sales.columns.get_loc(TODAY)

    forecasts = build_forecasts(today_day, sales, price, calendar)
    forecasts.to_parquet(OUT / "forecasts.parquet", index=False)
    print("forecasts:", forecasts.shape)

    stock_levels = build_stock_levels(today_day, sales, price, calendar)
    stock_levels.to_parquet(OUT / "stock_levels.parquet", index=False)
    print("stock levels:", stock_levels.shape)


if __name__ == "__main__":
    main()
