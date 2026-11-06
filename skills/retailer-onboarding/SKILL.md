---
name: retailer-onboarding
description: Onboard a new retailer's raw sales and price files into our forecasting data model. Use when a customer sends new data files that need to be profiled, mapped to our columns and checked for problems before forecasting. Works with the retailer-onboarding MCP server.
---

# Retailer onboarding

Goal - turn a new retailer's raw export into our standard table and give the analyst a short report of what was found and what was decided.

The agent does the heavy lifting (profiling, mapping, checks). The analyst owns the judgment (how to treat each data problem). Never fix a data problem without asking.

## Steps

1. **List the files** with `list_files(retailer)`.

2. **Profile every file** with `profile_file`. For each file note the separator issues, date formats, decimal commas, value ranges and anything unusual (for example negative numbers in a sales column, a minimum price of 0).

3. **Propose a column mapping** to our data model.
   - sales table - `item_id`, `store_id`, `date`, `sales`
   - prices table - `item_id`, `store_id`, `week_start`, `sell_price`
   Explain why each raw column was chosen. If a field has no clear match, ask the analyst instead of guessing.

4. **Check the mapping** with `check_mapping`. If `mapping_ok` is false, fix the mapping and check again.

5. **Report the issues** returned by `check_mapping`, highest severity first. For each one give the count, two or three examples and the options.
   - `missing_days` - usually a failed export. Ask the retailer to resend these days. Do not fill them with zeros.
   - `duplicate_rows` - options `keep_first` (exact copies from a double export) or `sum` (several real transactions per day).
   - `negative_sales` - usually returns. Options `set_zero`, `keep` or `drop`.
   - `items_without_price` - these items cannot use price features. Ask the retailer for the price list.
   - `zero_or_negative_prices` - treated as missing prices.
   - `possible_stockouts` - zeros may be empty shelves, not missing demand. Flag them for the merchandiser.
   - `short_history` - too little history for the forecast features.
   - `missing_calendar` - ask for holidays and promotions if the retailer has them.

6. **Ask the analyst** for the `duplicates` and `negative_sales` decisions. Do not pick them yourself.

7. **Build the standard table** with `build_standard_table` using the analyst's decisions.

8. **Write the onboarding report** in this format

```
# Onboarding report - <retailer>

## Files
<file> - <rows> rows, <columns>, <date range>

## Column mapping
| Our field | Raw column | Note |

## Data issues
| Severity | Check | Count | Examples | Decision / next step |

## Decisions
duplicates = ..., negative_sales = ... (decided by <analyst>)

## Requests to the retailer
- ...

## Output
data/onboarding/<retailer>/standard.parquet - <rows> rows, <series> series, <rows without price> rows without price
```

## Rules

- Report numbers exactly as the tools return them.
- Separate facts (what the tools found) from suggestions (what you think should be done).
- If something does not fit any check above, describe it and ask.
