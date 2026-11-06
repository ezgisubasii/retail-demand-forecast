#The helpers behind retailer onboarding: look at the raw files, map them onto our data model and point out what's wrong with the data
#On purpose, nothing in here decides how to fix a problem. These functions say what they found, and the caller (in the end, a person) makes the call
from pathlib import Path

import numpy as np
import pandas as pd

#Our data model: the fields each table has to have, and the type each one needs to be
DATA_MODEL = {
    "sales": {
        "item_id": "text",
        "store_id": "text",
        "date": "date",
        "sales": "number",
    },
    "prices": {
        "item_id": "text",
        "store_id": "text",
        "week_start": "date",
        "sell_price": "number",
    },
}

DATE_FORMATS = ["%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"]
MIN_HISTORY_DAYS = 112  #the forecast model looks back 112 days, so anything shorter can't be used yet


def read_raw(path) -> pd.DataFrame:
    """Read a csv file as text only, with the separator detected automatically."""
    return pd.read_csv(path, sep=None, engine="python", dtype=str)


def to_number(values: pd.Series) -> pd.Series:
    """Convert text to numbers. Accepts both 2.24 and 2,24."""
    return pd.to_numeric(values.str.strip().str.replace(",", ".", regex=False), errors="coerce")


def guess_date_format(values: pd.Series):
    """Return the date format that parses the most values, or None."""
    sample = values.dropna().head(500)
    best_format, best_share = None, 0.0
    for fmt in DATE_FORMATS:
        share = pd.to_datetime(sample, format=fmt, errors="coerce").notna().mean()
        if share > best_share:
            best_format, best_share = fmt, share
    return best_format if best_share >= 0.95 else None


def to_date(values: pd.Series, date_format: str) -> pd.Series:
    """Convert text to dates with a known format."""
    return pd.to_datetime(values, format=date_format, errors="coerce")


def profile_file(path) -> dict:
    """Describe a raw file: size, and for each column its type, missing values and examples."""
    df = read_raw(path)
    columns = []
    for name in df.columns:
        values = df[name]
        info = {
            "column": name,
            "missing_share": round(float(values.isna().mean()), 4),
            "unique_values": int(values.nunique()),
            "examples": values.dropna().drop_duplicates().head(5).tolist(),
        }
        date_format = guess_date_format(values)
        numbers = to_number(values.dropna())
        if date_format:
            dates = to_date(values, date_format)
            info.update({"looks_like": "date", "date_format": date_format,
                         "min": str(dates.min().date()), "max": str(dates.max().date())})
        elif len(numbers) and numbers.notna().mean() >= 0.95:
            info.update({"looks_like": "number", "min": float(numbers.min()), "max": float(numbers.max()),
                         "uses_decimal_comma": bool(values.str.contains(",", regex=False).any())})
        else:
            info["looks_like"] = "text"
        columns.append(info)
    return {"file": Path(path).name, "rows": len(df), "columns": columns}


def apply_mapping(path, table: str, mapping: dict) -> tuple[pd.DataFrame, list]:
    """Rename and convert the raw columns into our data model. Returns the table and any problems.

    mapping: {model_field: raw_column}, plus "date_format" for the date field.
    """
    df = read_raw(path)
    problems = []
    out = pd.DataFrame(index=df.index)
    for field, field_type in DATA_MODEL[table].items():
        column = mapping.get(field)
        if column is None:
            problems.append(f"Field '{field}' is not mapped")
            continue
        if column not in df.columns:
            problems.append(f"Column '{column}' for field '{field}' is not in the file")
            continue
        if field_type == "date":
            date_format = mapping.get("date_format") or guess_date_format(df[column])
            out[field] = to_date(df[column], date_format) if date_format else pd.NaT
        elif field_type == "number":
            out[field] = to_number(df[column])
        else:
            out[field] = df[column].str.strip()
        failed = int(out[field].isna().sum() - df[column].isna().sum())
        if failed > 0:
            problems.append(f"{failed} values in '{column}' could not be read as {field_type}")
    return out, problems


def find_gaps(sales: pd.DataFrame, prices: pd.DataFrame) -> list[dict]:
    """Check mapped sales and prices tables for the problems that matter for forecasting."""
    issues = []

    def add(check, severity, count, detail, examples=None):
        if count:
            issues.append({"check": check, "severity": severity, "count": int(count),
                           "detail": detail, "examples": (examples or [])[:5]})

    #When a whole day is missing for the entire store, it's almost always a failed export, not a day nobody bought anything
    all_days = pd.date_range(sales["date"].min(), sales["date"].max())
    missing_days = all_days.difference(pd.DatetimeIndex(sales["date"].unique()))
    add("missing_days", "high", len(missing_days),
        "Days with no rows at all in the sales file",
        [str(d.date()) for d in missing_days])

    #The same item, store and day showing up more than once
    key = ["item_id", "store_id", "date"]
    duplicated = sales[sales.duplicated(key, keep=False)]
    identical = sales.duplicated(keep="first").sum()
    add("duplicate_rows", "high", duplicated.duplicated(key).sum(),
        f"Rows that repeat an item/store/day key ({identical} of them are exact copies)",
        duplicated.head(5).astype(str).values.tolist())

    #Negative sales are almost always returns sneaking into the sales file
    negative = sales[sales["sales"] < 0]
    add("negative_sales", "medium", len(negative),
        "Negative units, probably returns recorded in the sales file",
        negative.head(5).astype(str).values.tolist())

    #Items we have sales for but can't put a price on
    no_price = sorted(set(sales["item_id"]) - set(prices["item_id"]))
    add("items_without_price", "high", len(no_price),
        "Items that have sales but no price at all", no_price)

    bad_price = prices[prices["sell_price"] <= 0]
    add("zero_or_negative_prices", "medium", len(bad_price),
        "Price rows with a price of 0 or less", bad_price.head(5).astype(str).values.tolist())

    #Series too short to fill the model's lookback features
    history = sales.groupby("item_id")["date"].agg(lambda d: (d.max() - d.min()).days + 1)
    short = history[history < MIN_HISTORY_DAYS]
    add("short_history", "low", len(short),
        f"Items with less than {MIN_HISTORY_DAYS} days of history", short.index.tolist())

    #A long run of zeros after an item had been selling usually means an empty shelf, not that people stopped wanting it
    stockouts = []
    for item, group in sales.groupby("item_id"):
        units = group.groupby("date")["sales"].sum().reindex(all_days, fill_value=0).values
        if units[:-28].mean() >= 0.5 and (units[-28:] == 0).all():
            stockouts.append(item)
    add("possible_stockouts", "medium", len(stockouts),
        "Items that sold before but had zero sales in the last 28 days", stockouts)

    #Things the model uses that this customer didn't send us
    issues.append({"check": "missing_calendar", "severity": "low", "count": 1,
                   "detail": "No calendar file with holidays or events was found", "examples": []})
    return issues


def build_standard_table(sales: pd.DataFrame, prices: pd.DataFrame, duplicates: str = "keep_first",
                         negative_sales: str = "set_zero") -> pd.DataFrame:
    """Turn mapped tables into our model format (one row per item, store and day with a price).

    duplicates: "keep_first" or "sum". negative_sales: "set_zero", "keep" or "drop".
    These are business decisions, so they are arguments and not hidden defaults.
    """
    key = ["item_id", "store_id", "date"]
    if duplicates == "sum":
        sales = sales.groupby(key, as_index=False)["sales"].sum()
    else:
        sales = sales.drop_duplicates(key, keep="first")

    if negative_sales == "set_zero":
        sales = sales.assign(sales=sales["sales"].clip(lower=0))
    elif negative_sales == "drop":
        sales = sales[sales["sales"] >= 0]

    #Each day takes the price of the most recent week that started on or before it
    prices = prices[prices["sell_price"] > 0].sort_values("week_start")
    sales = sales.sort_values("date")
    table = pd.merge_asof(sales, prices, left_on="date", right_on="week_start",
                          by=["item_id", "store_id"], direction="backward")
    table = table.drop(columns="week_start")
    table["id"] = table["item_id"] + "_" + table["store_id"]
    return table.sort_values(["id", "date"]).reset_index(drop=True)
