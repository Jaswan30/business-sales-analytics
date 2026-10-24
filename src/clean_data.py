
from pathlib import Path
import json
import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT_DIR / "data" / "raw" / "superstore_sales.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = OUTPUT_DIR / "clean_sales.csv"
REPORT_FILE = OUTPUT_DIR / "data_quality_report.json"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------
def load_data(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Place your CSV in data/raw/ and name it "
            "superstore_sales.csv."
        )

    # Read values as strings first so we can validate
    # and convert data types explicitly.
    try:
        df = pd.read_csv(file_path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, dtype=str, encoding="cp1252")
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, dtype=str, encoding="latin1")

    if df.empty:
        raise ValueError("The CSV file contains no data rows.")

    # Standardize column names.
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    if df.columns.duplicated().any():
        raise ValueError(
            "Duplicate column names found after standardization."
        )

    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


# --------------------------------------------------
# 3. STANDARDIZE COMMON COLUMN NAMES
# --------------------------------------------------
def standardize_columns(df):
    aliases = {
        "order_id": ["order_id", "orderid"],
        "order_date": ["order_date", "orderdate"],
        "ship_date": ["ship_date", "shipdate"],
        "customer_id": ["customer_id", "customerid"],
        "customer_name": ["customer_name", "customername"],
        "product_id": ["product_id", "productid"],
        "product_name": ["product_name", "productname"],
        "sub_category": ["sub_category", "subcategory"],
        "sales": ["sales", "revenue", "total_sales"],
        "quantity": ["quantity", "qty", "units_sold"],
        "discount": ["discount"],
        "profit": ["profit"],
        "category": ["category", "product_category"],
        "region": ["region", "sales_region"],
        "segment": ["segment", "customer_segment"],
        "state": ["state"],
        "city": ["city"],
        "ship_mode": ["ship_mode", "shipmode"],
    }

    rename_map = {}

    for canonical, possible_names in aliases.items():
        matches = [
            name for name in possible_names
            if name in df.columns
        ]

        if len(matches) > 1:
            raise ValueError(
                f"Multiple columns map to '{canonical}': {matches}"
            )

        if matches:
            rename_map[matches[0]] = canonical

    df = df.rename(columns=rename_map)

    required = [
        "order_id",
        "order_date",
        "sales",
        "quantity",
        "category",
        "profit",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Required columns are missing: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    return df


# --------------------------------------------------
# 4. CLEAN AND VALIDATE DATA
# --------------------------------------------------
def clean_data(df):
    report = {
        "source_file": RAW_FILE.name,
        "checks": {},
        "removed_rows": {},
        "notes": [
            "Negative profit is valid and is not removed.",
            "Rows are treated as exact duplicates only when "
            "all standardized source fields match.",
        ],
    }

    original_rows = len(df)

    # Normalize whitespace and empty strings.
    for col in df.columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("", np.nan)

    # Detect exact duplicate rows before transformations.
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    report["checks"]["exact_duplicate_rows"] = exact_duplicates
    report["removed_rows"]["exact_duplicates"] = exact_duplicates

    # Record missing values before removing incomplete records.
    missing_before = {
        col: int(count)
        for col, count in df.isna().sum().items()
    }

    report["checks"]["missing_values_by_column"] = missing_before

    # Convert dates and numeric columns.
    df["order_date"] = pd.to_datetime(
        df["order_date"], errors="coerce"
    )

    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(
            df["ship_date"], errors="coerce"
        )

    numeric_columns = [
        col for col in ["sales", "quantity", "discount", "profit"]
        if col in df.columns
    ]

    invalid_numeric_counts = {}

    for col in numeric_columns:
        before = df[col].notna()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        invalid_numeric_counts[col] = int(
            (before & df[col].isna()).sum()
        )

    report["checks"]["invalid_numeric_values"] = (
        invalid_numeric_counts
    )

    # Flag rows with missing critical fields or invalid dates.
    invalid_date = df["order_date"].isna()
    missing_order_id = df["order_id"].isna()
    invalid_sales = df["sales"].isna()
    invalid_quantity = df["quantity"].isna()

    # Superstore order dates should not be after ship dates.
    invalid_ship_sequence = pd.Series(
        False, index=df.index
    )

    if "ship_date" in df.columns:
        invalid_ship_sequence = (
            df["ship_date"].notna()
            & df["order_date"].notna()
            & (df["ship_date"] < df["order_date"])
        )

    # Non-positive sales or quantity are flagged for review.
    # These records are excluded from the standard sales dashboard.
    nonpositive_sales = df["sales"].notna() & (df["sales"] <= 0)
    nonpositive_quantity = (
        df["quantity"].notna() & (df["quantity"] <= 0)
    )

    invalid_mask = (
        invalid_date
        | missing_order_id
        | invalid_sales
        | invalid_quantity
        | nonpositive_sales
        | nonpositive_quantity
        | invalid_ship_sequence
    )

    report["checks"]["invalid_order_dates"] = int(
        invalid_date.sum()
    )
    report["checks"]["missing_order_ids"] = int(
        missing_order_id.sum()
    )
    report["checks"]["nonpositive_sales_rows"] = int(
        nonpositive_sales.sum()
    )
    report["checks"]["nonpositive_quantity_rows"] = int(
        nonpositive_quantity.sum()
    )
    report["checks"]["ship_date_before_order_date"] = int(
        invalid_ship_sequence.sum()
    )

    # Keep excluded rows countable and explain why they were removed.
    report["removed_rows"]["invalid_or_incomplete_records"] = int(
        invalid_mask.sum()
    )

    df = df.loc[~invalid_mask].copy()

    # Remove optional records that have no useful product/category
    # description from category-level analysis only if needed later;
    # do not delete rows merely because an optional field is missing.

    # Add useful reporting fields.
    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["year_month"] = df["order_date"].dt.strftime("%Y-%m")

    if "discount" in df.columns:
        df["discount"] = df["discount"].fillna(0)

    # Final validation summary.
    report["input_rows"] = original_rows
    report["clean_rows"] = len(df)
    report["total_rows_removed"] = original_rows - len(df)
    report["retention_rate_percent"] = round(
        len(df) / original_rows * 100, 2
    ) if original_rows else 0

    report["checks"]["negative_profit_rows_retained"] = int(
        (df["profit"] < 0).sum()
    )

    report["checks"]["remaining_missing_values_by_column"] = {
        col: int(count)
        for col, count in df.isna().sum().items()
    }

    return df, report


# --------------------------------------------------
# 5. SAVE OUTPUTS
# --------------------------------------------------
def main():
    try:
        raw_df = load_data(RAW_FILE)
        raw_df = standardize_columns(raw_df)

        clean_df, report = clean_data(raw_df)

        clean_df.to_csv(CLEAN_FILE, index=False)

        with open(REPORT_FILE, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, default=str)

        print("\nDATA CLEANING COMPLETED")
        print("-" * 40)
        print(f"Original rows : {report['input_rows']:,}")
        print(f"Clean rows    : {report['clean_rows']:,}")
        print(f"Removed rows  : {report['total_rows_removed']:,}")
        print(
            f"Retention     : "
            f"{report['retention_rate_percent']}%"
        )
        print(f"\nClean dataset : {CLEAN_FILE}")
        print(f"Quality report: {REPORT_FILE}")

        print("\nCleaned data preview:")
        print(clean_df.head().to_string(index=False))

    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        print(f"\nDATA CLEANING FAILED: {exc}")
        raise


if __name__ == "__main__":
    main()
