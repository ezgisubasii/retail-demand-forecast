#Backtest of 28 days average baseline
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "processed" / "sales.parquet"
HORIZON = 28 #how many days ahead we forecast
N_FOLDS = 3 #how many periods we test on


def load_data() -> pd.DataFrame:
    """Load the sales data from parquet file."""
    df = pd.read_parquet(DATA)
    #No price means the item was not on the shelf, so these are not real zero demand days
    return df[df["sell_price"].notna()].reset_index(drop=True)


def make_cutoffs(df: pd.DataFrame) -> list:
    """Make cutoffs for backtesting, oldest first."""
    last_date = df["date"].max()
    return [last_date - pd.Timedelta(days=HORIZON * i) for i in range(N_FOLDS, 0, -1)]


def baseline_forecast(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Make a baseline forecast using the average of the last 28 days."""
    #Keep only the last 28 days of the training set
    recent = train[train["date"] > train["date"].max() - pd.Timedelta(days=HORIZON)]
    #Calculate the average sales for each series (id = item + store)
    avg_sales = recent.groupby("id", observed=True)["sales"].mean().reset_index()
    avg_sales = avg_sales.rename(columns={"sales": "forecast"})
    #Add the forecast to the test set
    forecast = test.merge(avg_sales, on="id", how="left")
    #Fill missing forecasts with 0 (for items not seen in the last 28 days)
    forecast["forecast"] = forecast["forecast"].fillna(0)
    return forecast


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Calculate Weighted Absolute Percentage Error."""
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)


def main():
    df = load_data()
    cutoffs = make_cutoffs(df)
    results = []

    for cutoff in cutoffs:
        train = df[df["date"] <= cutoff]
        test = df[(df["date"] > cutoff) & (df["date"] <= cutoff + pd.Timedelta(days=HORIZON))]

        #The forecast table already has both actual sales and the forecast
        forecast_df = baseline_forecast(train, test)

        #Calculate metrics
        mae_value = mae(forecast_df["sales"], forecast_df["forecast"])
        wape_value = wape(forecast_df["sales"], forecast_df["forecast"])

        results.append({
            "cutoff": cutoff.date(),
            "mae": round(mae_value, 3),
            "wape": round(wape_value, 3)
        })

    results_df = pd.DataFrame(results)
    print(results_df)
    print("mean wape:", round(results_df["wape"].mean(), 3))


if __name__ == "__main__":
    main()
