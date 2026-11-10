#Steps of the weekly forecast pipeline. Airflow runs each step as its own task
#Usage: python src/pipeline.py <prepare|check_data|evaluate|gate|publish>
import json
import os
import sys

import pandas as pd

import build_artifacts
import prepare_data
from backtest import DATA, HORIZON, make_cutoffs, wape
from train_lgbm import FEATURES, features_at, load_tables, make_training_set, train_model

METRICS = DATA.parent / "metrics.json"

#Rules a new model has to pass before its forecasts are published
TOTAL_MIN, TOTAL_MAX = 0.85, 1.15  #total forecast must be within 15% of real sales


def check_data():
    """Stop the pipeline if the prepared data has problems the model cannot handle."""
    df = pd.read_parquet(DATA)
    problems = []
    if df.empty:
        problems.append("sales table is empty")
    if df.duplicated(["id", "date"]).any():
        problems.append(f"{df.duplicated(['id', 'date']).sum()} duplicate item/day rows")
    days_per_item = df.groupby("id", observed=True)["date"].nunique()
    if days_per_item.nunique() > 1:
        problems.append("items have a different number of days (missing days)")
    if (df["sales"] < 0).any():
        problems.append(f"{(df['sales'] < 0).sum()} negative sales")

    if problems:
        raise ValueError("Data check failed: " + "; ".join(problems))
    print(f"Data check passed: {len(df):,} rows, {df['id'].nunique()} items, "
          f"{df['date'].min().date()} to {df['date'].max().date()}")


def evaluate():
    """Backtest the model on the latest 28 days and save the metrics."""
    objective = os.environ.get("MODEL_OBJECTIVE", "poisson")
    sales, price, calendar = load_tables()
    df = pd.read_parquet(DATA)
    cutoff = make_cutoffs(df[df["sell_price"].notna()])[-1]
    cutoff_day = sales.columns.get_loc(cutoff)

    model = train_model(make_training_set(cutoff_day, sales, price, calendar), objective)
    test = features_at(cutoff_day, sales, price, calendar)
    test["lgbm"] = model.predict(test[FEATURES])
    test["baseline"] = test["mean_28"].fillna(0)
    store = test.groupby("date")[["sales", "lgbm", "baseline"]].sum()

    metrics = {
        "objective": objective,
        "cutoff": str(cutoff.date()),
        "horizon_days": HORIZON,
        "item_day_wape": {"lgbm": round(wape(test["sales"], test["lgbm"]), 4),
                          "baseline": round(wape(test["sales"], test["baseline"]), 4)},
        "store_day_wape": {"lgbm": round(wape(store["sales"], store["lgbm"]), 4),
                           "baseline": round(wape(store["sales"], store["baseline"]), 4)},
        "total_vs_actual": round(float(test["lgbm"].sum() / test["sales"].sum()), 4),
    }
    METRICS.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def gate():
    """Publish only if the new model is better than the baseline and not biased."""
    metrics = json.loads(METRICS.read_text())
    failures = []
    if metrics["store_day_wape"]["lgbm"] >= metrics["store_day_wape"]["baseline"]:
        failures.append(f"store level WAPE {metrics['store_day_wape']['lgbm']} is not better "
                        f"than the baseline {metrics['store_day_wape']['baseline']}")
    if not TOTAL_MIN <= metrics["total_vs_actual"] <= TOTAL_MAX:
        failures.append(f"total forecast is {metrics['total_vs_actual']:.0%} of real sales "
                        f"(allowed {TOTAL_MIN:.0%} to {TOTAL_MAX:.0%})")

    if failures:
        raise ValueError("Validation gate failed, forecasts are NOT published: " + "; ".join(failures))
    print("Validation gate passed")


def publish():
    """Write the forecasts and stock levels that the MCP server reads."""
    build_artifacts.main()


STEPS = {
    "prepare": prepare_data.main,
    "check_data": check_data,
    "evaluate": evaluate,
    "gate": gate,
    "publish": publish,
}

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in STEPS:
        sys.exit(f"Usage: python src/pipeline.py <{'|'.join(STEPS)}>")
    STEPS[sys.argv[1]]()
