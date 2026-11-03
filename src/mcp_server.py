#MCP server: lets an AI agent ask for demand forecasts and order recommendations
#Run src/build_artifacts.py first; this server only reads its output files
import math
from pathlib import Path

import pandas as pd
from mcp.server.mcpserver import MCPServer

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
TODAY = "2016-04-24"
STORE = "CA_1"
LEAD_TIME = 7
REVIEW_PERIOD = 7

forecasts = pd.read_parquet(PROCESSED / "forecasts.parquet")
stock_levels = pd.read_parquet(PROCESSED / "stock_levels.parquet")
#"FOODS_3_090_CA_1_evaluation" -> "FOODS_3_090"
forecasts["item_id"] = forecasts["id"].str.replace(f"_{STORE}_evaluation", "")
stock_levels["item_id"] = stock_levels["id"].str.replace(f"_{STORE}_evaluation", "")

mcp = MCPServer(
    name="retail-forecast",
    instructions=(
        f"Demand forecasts and order recommendations for store {STORE}, department FOODS_3 "
        f"(Walmart M5 data). Today is {TODAY}. Item ids look like FOODS_3_090. "
        "Use find_items if you do not know the exact id."
    ),
)


@mcp.tool()
def find_items(top: int = 10, sort_by: str = "sales") -> list[dict]:
    """List items of the store.

    sort_by: "sales" (best sellers first) or "stockout" (possible empty shelves first).
    """
    table = stock_levels.copy()
    table["possible_stockout"] = (table["zero_share_28"] == 1) & (table["mean_112"] >= 0.5)
    if sort_by == "stockout":
        table = table.sort_values(["possible_stockout", "mean_112"], ascending=False)
    else:
        table = table.sort_values("mean_28", ascending=False)
    return [
        {
            "item_id": row.item_id,
            "avg_daily_sales_last_28_days": round(row.mean_28, 2),
            "possible_stockout": bool(row.possible_stockout),
        }
        for row in table.head(top).itertuples()
    ]


@mcp.tool()
def get_forecast(item_id: str, days: int = 28) -> dict:
    """Daily sales forecast for one item for the next `days` days (max 28)."""
    rows = forecasts[forecasts["item_id"] == item_id].sort_values("date").head(days)
    if rows.empty:
        return {"error": f"Unknown item_id {item_id}. Use find_items to see valid ids."}
    return {
        "item_id": item_id,
        "store": STORE,
        "today": TODAY,
        "avg_daily_sales_last_28_days": round(float(rows["mean_28"].iloc[0]), 2),
        "total_forecast": round(float(rows["forecast"].sum()), 1),
        "daily": [
            {"date": str(r.date.date()), "forecast": round(float(r.forecast), 2)}
            for r in rows.itertuples()
        ],
    }


@mcp.tool()
def recommend_order(item_id: str, on_hand: int, service_level: int = 95) -> dict:
    """How many units to order today for one item.

    on_hand: units currently in the store.
    service_level: 80, 90 or 95 (% of periods where stock should be enough).
    """
    if service_level not in (80, 90, 95):
        return {"error": "service_level must be 80, 90 or 95"}
    rows = stock_levels[stock_levels["item_id"] == item_id]
    if rows.empty:
        return {"error": f"Unknown item_id {item_id}. Use find_items to see valid ids."}
    row = rows.iloc[0]

    target_stock = float(row[f"stock_{service_level}"])
    order_quantity = max(0, math.ceil(target_stock - on_hand))
    expected_demand = float(
        forecasts[forecasts["item_id"] == item_id].sort_values("date").head(LEAD_TIME + REVIEW_PERIOD)["forecast"].sum()
    )

    result = {
        "item_id": item_id,
        "today": TODAY,
        "covers_days": LEAD_TIME + REVIEW_PERIOD,
        "expected_demand": round(expected_demand, 1),
        "target_stock": round(target_stock, 1),
        "on_hand": on_hand,
        "order_quantity": order_quantity,
        "service_level": service_level,
        "backtest_note": "In backtests this model covered demand in 80.8%, 90.3% and 95.4% of cases for the 80/90/95 targets.",
    }
    #Zero sales for 4 weeks after selling well before usually means an empty shelf, not no demand
    if row["zero_share_28"] == 1 and row["mean_112"] >= 0.5:
        result["warning"] = (
            "No sales in the last 28 days although the item sold "
            f"{row['mean_112']:.1f} per day over the last 112 days. Possible stockout, check the shelf."
        )
    return result


if __name__ == "__main__":
    mcp.run()
