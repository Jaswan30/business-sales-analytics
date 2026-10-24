
from pathlib import Path
import sqlite3
import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = ROOT_DIR / "outputs" / "clean_sales.csv"
OUTPUT_DIR = ROOT_DIR / "outputs"
DB_FILE = OUTPUT_DIR / "sales.db"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. SQL ANALYSIS QUERIES
# --------------------------------------------------
QUERIES = {
    "monthly_performance": """
        SELECT
            year_month,
            ROUND(SUM(sales), 2) AS revenue,
            ROUND(SUM(profit), 2) AS profit,
            COUNT(*) AS transaction_lines,
            COUNT(DISTINCT order_id) AS orders
        FROM sales
        GROUP BY year_month
        ORDER BY year_month;
    """,

    "category_performance": """
        SELECT
            category,
            ROUND(SUM(sales), 2) AS revenue,
            ROUND(SUM(profit), 2) AS profit,
            ROUND(
                100.0 * SUM(profit) / NULLIF(SUM(sales), 0),
                2
            ) AS profit_margin_percent,
            COUNT(DISTINCT order_id) AS orders
        FROM sales
        GROUP BY category
        ORDER BY revenue DESC;
    """,

    "regional_performance": """
        SELECT
            region,
            ROUND(SUM(sales), 2) AS revenue,
            ROUND(SUM(profit), 2) AS profit,
            COUNT(DISTINCT order_id) AS orders
        FROM sales
        GROUP BY region
        ORDER BY revenue DESC;
    """,

    "top_products": """
        SELECT
            product_name,
            category,
            ROUND(SUM(sales), 2) AS revenue,
            ROUND(SUM(profit), 2) AS profit,
            SUM(quantity) AS units_sold
        FROM sales
        GROUP BY product_name, category
        ORDER BY revenue DESC
        LIMIT 10;
    """,

    "loss_making_products": """
        SELECT
            product_name,
            category,
            ROUND(SUM(sales), 2) AS revenue,
            ROUND(SUM(profit), 2) AS profit
        FROM sales
        GROUP BY product_name, category
        HAVING SUM(profit) < 0
        ORDER BY profit ASC
        LIMIT 10;
    """,

    "discount_analysis": """
        SELECT
            discount,
            COUNT(*) AS transaction_lines,
            ROUND(SUM(sales), 2) AS revenue,
            ROUND(SUM(profit), 2) AS profit
        FROM sales
        WHERE discount IS NOT NULL
        GROUP BY discount
        ORDER BY discount;
    """,

    "state_performance": """
        SELECT
            state,
            ROUND(SUM(sales), 2) AS revenue,
            ROUND(SUM(profit), 2) AS profit
        FROM sales
        GROUP BY state
        ORDER BY revenue DESC
        LIMIT 10;
    """
}


# --------------------------------------------------
# 3. LOAD CLEANED DATA
# --------------------------------------------------
def load_clean_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {INPUT_FILE}\n"
            "Run python src/clean_data.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "order_id",
        "year_month",
        "sales",
        "profit",
        "quantity",
        "category",
        "product_name",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Convert reporting measures to numeric types.
    for col in ["sales", "profit", "quantity", "discount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # SQLite cannot store pandas NaT/NaN reliably as desired.
    # Convert pandas missing values to Python None.
    df = df.astype(object).where(pd.notna(df), None)

    print(f"Loaded {len(df):,} cleaned rows for SQL analysis.")
    return df


# --------------------------------------------------
# 4. CREATE DATABASE
# --------------------------------------------------
def create_database(df):
    # Replace the project database each time the script runs.
    # This ensures it reflects the latest cleaned dataset.
    with sqlite3.connect(DB_FILE) as connection:
        df.to_sql("sales", connection, if_exists="replace", index=False)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sales_order_date
            ON sales(order_date)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_sales_category
            ON sales(category)
        """)

        connection.commit()

    print(f"SQLite database created: {DB_FILE}")


# --------------------------------------------------
# 5. RUN SQL QUERIES
# --------------------------------------------------
def run_analysis():
    with sqlite3.connect(DB_FILE) as connection:
        for report_name, query in QUERIES.items():
            # Some Superstore files may not include state or discount.
            # Skip those optional analyses when columns are unavailable.
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(sales)"
                ).fetchall()
            }

            if report_name == "state_performance" and "state" not in columns:
                print("Skipping state analysis: no state column.")
                continue

            if report_name == "discount_analysis" and "discount" not in columns:
                print("Skipping discount analysis: no discount column.")
                continue

            result = pd.read_sql_query(query, connection)

            output_file = OUTPUT_DIR / f"{report_name}.csv"
            result.to_csv(output_file, index=False)

            print(f"\n{report_name.replace('_', ' ').title()}")
            print(result.head(5).to_string(index=False))
            print(f"Saved: {output_file.name}")


# --------------------------------------------------
# 6. EXECUTION
# --------------------------------------------------
def main():
    try:
        df = load_clean_data()
        create_database(df)
        run_analysis()

        print("\n" + "=" * 45)
        print("SQL BUSINESS ANALYSIS COMPLETED SUCCESSFULLY")
        print("=" * 45)
        print(f"Database: {DB_FILE}")
        print(f"Reports : {OUTPUT_DIR}")

    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"\nSQL ANALYSIS FAILED: {exc}")
        raise


if __name__ == "__main__":
    main()
