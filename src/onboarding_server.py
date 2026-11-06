#This is the MCP server an agent talks to when we onboard a new retailer: it can look at their files, try out a column mapping and build our standard table
#I keep it fenced in to data/onboarding/<retailer>/ so the agent can never wander off and read anything else
from pathlib import Path

from mcp.server.mcpserver import MCPServer

import onboarding as ob

ONBOARDING = (Path(__file__).resolve().parent.parent / "data" / "onboarding").resolve()

mcp = MCPServer(
    name="retailer-onboarding",
    instructions=(
        "Tools to onboard a new retailer's raw data into our data model. "
        f"Our data model: {ob.DATA_MODEL}. "
        "Profile the files first, propose a column mapping, check it, and ask the user before fixing any data problem."
    ),
)


def retailer_file(retailer: str, file_name: str) -> Path:
    """Return a path inside data/onboarding/<retailer>/ and refuse anything outside it."""
    path = (ONBOARDING / retailer / file_name).resolve()
    if ONBOARDING not in path.parents:
        raise ValueError("Only files under data/onboarding/<retailer>/ can be read")
    if not path.exists():
        raise ValueError(f"{retailer}/{file_name} does not exist")
    return path


@mcp.tool()
def list_files(retailer: str) -> dict:
    """List the raw files a retailer sent."""
    folder = (ONBOARDING / retailer).resolve()
    if ONBOARDING not in folder.parents or not folder.exists():
        return {"error": f"Unknown retailer {retailer}",
                "retailers": sorted(p.name for p in ONBOARDING.iterdir() if p.is_dir())}
    return {"retailer": retailer, "files": sorted(p.name for p in folder.glob("*.csv"))}


@mcp.tool()
def profile_file(retailer: str, file_name: str) -> dict:
    """Rows, columns, guessed types, date formats, ranges and example values of one raw file."""
    try:
        return ob.profile_file(retailer_file(retailer, file_name))
    except ValueError as error:
        return {"error": str(error)}


@mcp.tool()
def check_mapping(retailer: str, sales_file: str, sales_mapping: dict[str, str],
                  price_file: str, price_mapping: dict[str, str]) -> dict:
    """Apply a column mapping and report conversion problems and data gaps.

    sales_mapping keys: item_id, store_id, date, sales (and optional date_format).
    price_mapping keys: item_id, store_id, week_start, sell_price (and optional date_format).
    Values are column names in the raw files.
    """
    try:
        sales_path, price_path = retailer_file(retailer, sales_file), retailer_file(retailer, price_file)
    except ValueError as error:
        return {"error": str(error)}
    sales, sales_problems = ob.apply_mapping(sales_path, "sales", sales_mapping)
    prices, price_problems = ob.apply_mapping(price_path, "prices", price_mapping)
    if sales_problems or price_problems:
        return {"mapping_ok": False, "sales_problems": sales_problems, "price_problems": price_problems}
    return {
        "mapping_ok": True,
        "summary": {
            "sales_rows": len(sales),
            "items": int(sales["item_id"].nunique()),
            "stores": int(sales["store_id"].nunique()),
            "first_day": str(sales["date"].min().date()),
            "last_day": str(sales["date"].max().date()),
        },
        "issues": ob.find_gaps(sales, prices),
    }


@mcp.tool()
def build_standard_table(retailer: str, sales_file: str, sales_mapping: dict[str, str],
                         price_file: str, price_mapping: dict[str, str],
                         duplicates: str, negative_sales: str) -> dict:
    """Write the retailer's data in our model format to data/onboarding/<retailer>/standard.parquet.

    duplicates: "keep_first" or "sum". negative_sales: "set_zero", "keep" or "drop".
    Ask the user to choose these, they are business decisions.
    """
    if duplicates not in ("keep_first", "sum") or negative_sales not in ("set_zero", "keep", "drop"):
        return {"error": "duplicates must be keep_first or sum, negative_sales must be set_zero, keep or drop"}
    try:
        sales_path, price_path = retailer_file(retailer, sales_file), retailer_file(retailer, price_file)
    except ValueError as error:
        return {"error": str(error)}
    sales, sales_problems = ob.apply_mapping(sales_path, "sales", sales_mapping)
    prices, price_problems = ob.apply_mapping(price_path, "prices", price_mapping)
    if sales_problems or price_problems:
        return {"error": "Fix the mapping first", "sales_problems": sales_problems, "price_problems": price_problems}
    table = ob.build_standard_table(sales, prices, duplicates, negative_sales)
    out = ONBOARDING / retailer / "standard.parquet"
    table.to_parquet(out, index=False)
    return {
        "written_to": f"data/onboarding/{retailer}/standard.parquet",
        "rows": len(table),
        "series": int(table["id"].nunique()),
        "rows_without_price": int(table["sell_price"].isna().sum()),
        "decisions": {"duplicates": duplicates, "negative_sales": negative_sales},
    }


if __name__ == "__main__":
    mcp.run()
