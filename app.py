

from pathlib import Path

import json

import sqlite3

from io import BytesIO



import numpy as np

import pandas as pd

import plotly.express as px

import streamlit as st





# ============================================================

# 1. CONFIGURATION

# ============================================================

ROOT = Path(__file__).resolve().parent

OUTPUTS = ROOT / "outputs"

CLEAN_FILE = OUTPUTS / "clean_sales.csv"

QUALITY_FILE = OUTPUTS / "data_quality_report.json"

DB_FILE = OUTPUTS / "sales.db"



NAVY = "#172B4D"

BLUE = "#316FEA"

MUTED = "#64748B"

BORDER = "#E2E8F0"

BACKGROUND = "#F5F7FB"



st.set_page_config(

    page_title="Business Analytics | Sales Intelligence",

    page_icon=None,

    layout="wide",

    initial_sidebar_state="expanded",

)



st.markdown(

    f"""

    <style>
    .stApp {{
        background: {BACKGROUND};
        color: #18263A;
    }}
    [data-testid="stHeader"] {{
        background: rgba(245,247,251,0.95);
    }}
    [data-testid="stSidebar"] {{
        background: {NAVY};
    }}
    [data-testid="stSidebar"] * {{
        color: #F1F5F9;
    }}
    [data-testid="stSidebar"] [data-baseweb="select"] * {{
        color: #172B4D;
    }}
    [data-testid="stSidebar"] input {{
        color: #172B4D;
    }}
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }}
    .page-eyebrow {{
        color: {BLUE};
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }}
    .page-title {{
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.045em;
        color: {NAVY};
        margin: 0;
    }}
    .page-subtitle {{
        color: {MUTED};
        font-size: 0.94rem;
        margin-top: 0.45rem;
        margin-bottom: 1.6rem;
    }}
    .section-title {{
        font-size: 1.1rem;
        font-weight: 700;
        color: {NAVY};
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }}
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 18px 18px 14px 18px;
        box-shadow: 0 2px 5px rgba(15, 23, 42, 0.025);
    }}
    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-size: 0.82rem;
    }}
    div[data-testid="stMetricValue"] {{
        color: {NAVY};
        font-weight: 750;
        font-size: 1.7rem;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-color: {BORDER};
        border-radius: 12px;
    }}
    .source-note {{
        color: {MUTED};
        font-size: 0.78rem;
    }}
    div.stButton > button,
    div.stDownloadButton > button {{
        border-radius: 8px;
        font-weight: 600;
    }}
    hr {{
        border-color: {BORDER};
    }}

    /* Sidebar date picker: dark text on a white input */
    [data-testid="stSidebar"] [data-testid="stDateInput"] input {{
        background-color: #FFFFFF !important;
        color: #172B4D !important;
        -webkit-text-fill-color: #172B4D !important;
        opacity: 1 !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebar"] [data-testid="stDateInput"] button,
    [data-testid="stSidebar"] [data-testid="stDateInput"] svg {{
        color: #172B4D !important;
        fill: #172B4D !important;
    }}
    [data-testid="stSidebar"] [data-testid="stDateInput"] input::placeholder {{
        color: #64748B !important;
        -webkit-text-fill-color: #64748B !important;
        opacity: 1 !important;
    }}
    </style>

    """,

    unsafe_allow_html=True,

)



# Consistent Plotly style

CHART_LAYOUT = dict(

    template="plotly_white",

    paper_bgcolor="#FFFFFF",

    plot_bgcolor="#FFFFFF",

    font=dict(family="Arial, sans-serif", color=NAVY, size=12),

    margin=dict(l=18, r=18, t=55, b=20),

    legend=dict(orientation="h", yanchor="bottom", y=1.02),

)





# ============================================================

# 2. HELPERS

# ============================================================

def page_header(title, subtitle):

    st.markdown('<div class="page-eyebrow">BUSINESS INTELLIGENCE</div>',

                unsafe_allow_html=True)

    st.markdown(f'<h1 class="page-title">{title}</h1>',

                unsafe_allow_html=True)

    st.markdown(f'<div class="page-subtitle">{subtitle}</div>',

                unsafe_allow_html=True)





def normalize_name(name):

    return (

        str(name).strip().lower()

        .replace("%", " percent ")

        .replace("/", " ")

        .replace("-", " ")

        .replace("_", " ")

        .replace(".", " ")

    )





def standardize_headers(df):

    result = df.copy()

    result.columns = [

        "_".join(normalize_name(c).split())

        for c in result.columns

    ]



    if result.columns.duplicated().any():

        raise ValueError(

            "Two or more columns have the same name after "

            "standardization. Rename the duplicate columns in "

            "your source file and upload it again."

        )



    return result





ALIASES = {

    "date": [

        "order_date", "orderdate", "date", "transaction_date",

        "sales_date", "invoice_date", "created_at"

    ],

    "sales": [

        "sales", "revenue", "total_sales", "sales_amount",

        "amount", "total_amount", "net_sales", "turnover"

    ],

    "profit": [

        "profit", "net_profit", "profit_amount", "gross_profit"

    ],

    "quantity": [

        "quantity", "qty", "units_sold", "units", "volume"

    ],

    "order_id": [

        "order_id", "orderid", "transaction_id", "invoice_id",

        "order_number", "invoice_number"

    ],

    "category": [

        "category", "product_category", "product_type",

        "department", "item_category"

    ],

    "region": [

        "region", "sales_region", "territory", "market",

        "country", "state", "city"

    ],

    "product": [

        "product_name", "product", "item_name", "item",

        "product_title", "sku_name"

    ],

    "discount": ["discount", "discount_rate", "discount_percent"],

}





def guess_column(columns, field):

    normalized = {normalize_name(c).replace(" ", "_"): c for c in columns}



    for alias in ALIASES[field]:

        key = normalize_name(alias).replace(" ", "_")

        if key in normalized:

            return normalized[key]



    if field == "date":

        for col in columns:

            if "date" in normalize_name(col):

                return col



    return None





def read_uploaded_file(uploaded, sheet_name=None):

    raw = uploaded.getvalue()

    suffix = Path(uploaded.name).suffix.lower()



    if suffix == ".csv":

        errors = []

        for encoding in ["utf-8-sig", "cp1252", "latin1"]:

            try:

                return pd.read_csv(BytesIO(raw), encoding=encoding)

            except (UnicodeDecodeError, pd.errors.ParserError) as exc:

                errors.append(str(exc))



        raise ValueError("Unable to read the CSV. Check its format and encoding.")



    if suffix == ".xlsx":

        return pd.read_excel(

            BytesIO(raw),

            sheet_name=sheet_name,

            engine="openpyxl",

        )



    raise ValueError("Supported formats are CSV and Excel .xlsx.")





def load_project_data():

    if not CLEAN_FILE.exists():

        raise FileNotFoundError(

            "The cleaned project dataset is missing. "

            "Run py src/clean_data.py first."

        )



    df = pd.read_csv(CLEAN_FILE)

    df = standardize_headers(df)



    # Create a canonical mapping for the existing project dataset.

    mapping = {

        field: guess_column(df.columns, field)

        for field in ALIASES

    }

    return df, mapping





def clean_numeric(series):

    if pd.api.types.is_numeric_dtype(series):

        return pd.to_numeric(series, errors="coerce")



    text = series.astype(str).str.strip()

    text = text.str.replace(r"[$£€₹,\s]", "", regex=True)

    text = text.str.replace(r"^**\\(**(.*)**\\)**$", r"-\1", regex=True)

    text = text.replace(

        {"": np.nan, "nan": np.nan, "None": np.nan, "N/A": np.nan}

    )

    return pd.to_numeric(text, errors="coerce")





def quality_summary(df):

    return {

        "Rows": len(df),

        "Columns": len(df.columns),

        "Exact duplicate rows": int(df.duplicated().sum()),

        "Rows with at least one missing value": int(df.isna().any(axis=1).sum()),

        "Missing cells": int(df.isna().sum().sum()),

    }





def fmt_money(value, currency):

    return f"{currency}{value:,.2f}"





# ============================================================

# 3. SIDEBAR / NAVIGATION

# ============================================================

with st.sidebar:

    st.markdown("## SALES INTELLIGENCE")

    st.caption("Business performance workspace")

    st.divider()



    page = st.radio(

        "WORKSPACE",

        [

            "Overview",

            "Data Explorer",

            "Upload & Analyze",

            "Data Quality",

        ],

        label_visibility="visible",

    )



    st.divider()

    st.markdown("**Data source**")

    source = st.radio(

        "Choose the dataset",

        ["Project dataset", "Upload a file"],

        index=0,

        label_visibility="collapsed",

    )



    uploaded_file = None

    selected_sheet = None







    st.divider()

    st.caption("Python · Pandas · SQL · Plotly · Streamlit")





# ============================================================

# ============================================================
# 4. LOAD THE SELECTED DATA SOURCE
# ============================================================
source_label = "Project dataset"
mapping = {}
uploaded_file = None
selected_sheet = None

if source == "Upload a file":
    page_header(
        "Upload & Analyze",
        "Import a business dataset to explore its performance and data quality.",
    )
    st.info(
        "Upload a CSV or Excel workbook to analyze its performance. "
        "Your original file will not be modified."
    )

    st.markdown("#### Import dataset")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx"],
        help="Supported formats: CSV and Excel (.xlsx).",
        key="main_dataset_uploader",
    )

    if uploaded_file is None:
        st.caption(
            "Supported formats: .csv and .xlsx. "
            "For Excel workbooks, select the worksheet you want to analyze."
        )
        st.stop()

    source_label = uploaded_file.name

    if uploaded_file.name.lower().endswith(".xlsx"):
        try:
            workbook = pd.ExcelFile(
                BytesIO(uploaded_file.getvalue()),
                engine="openpyxl",
            )
            selected_sheet = st.selectbox(
                "Worksheet",
                options=workbook.sheet_names,
                key="main_dataset_sheet",
            )
        except Exception as exc:
            st.error(f"Unable to inspect workbook: {exc}")
            st.stop()

    try:
        raw_df = read_uploaded_file(uploaded_file, selected_sheet)
        if raw_df.empty:
            st.error("The selected file contains no data rows.")
            st.stop()

        df = standardize_headers(raw_df)
        mapping = {
            field: guess_column(df.columns, field)
            for field in ALIASES
        }

    except Exception as exc:
        st.error(f"Unable to load the uploaded dataset: {exc}")
        st.stop()

else:
    try:
        df, mapping = load_project_data()
    except Exception as exc:
        st.error(str(exc))
        st.stop()


# ============================================================
# 5. COLUMN MAPPING

# ============================================================

if (
    source == "Upload a file"
    and not (
        st.session_state.get("analysis_ready")
        and st.session_state.get("active_source_mode") == "Upload a file"
        and st.session_state.get("active_source") == source_label
    )
):

    page_header(

        "Upload & Analyze",

        "Map your source fields to business metrics before generating the analysis.",

    )



    st.markdown("#### Dataset preview")

    st.caption(

        f"Source: {source_label} · {len(df):,} rows · {len(df.columns)} columns"

    )

    st.dataframe(df.head(8), use_container_width=True, hide_index=True)



    st.markdown("#### Map your columns")

    st.write(

        "The application suggests matching columns automatically. "

        "Review the mappings below, especially when your file uses custom headings."

    )



    choices = [None] + list(df.columns)



    with st.form("column_mapping_form"):

        c1, c2, c3 = st.columns(3)



        with c1:

            date_col = st.selectbox(

                "Date column (required)",

                choices,

                index=choices.index(mapping["date"])

                if mapping.get("date") in choices else 0,

            )

            sales_col = st.selectbox(

                "Sales / revenue column (required)",

                choices,

                index=choices.index(mapping["sales"])

                if mapping.get("sales") in choices else 0,

            )

            profit_col = st.selectbox(

                "Profit column (optional)",

                choices,

                index=choices.index(mapping["profit"])

                if mapping.get("profit") in choices else 0,

            )



        with c2:

            category_col = st.selectbox(

                "Category column",

                choices,

                index=choices.index(mapping["category"])

                if mapping.get("category") in choices else 0,

            )

            region_col = st.selectbox(

                "Region column",

                choices,

                index=choices.index(mapping["region"])

                if mapping.get("region") in choices else 0,

            )

            product_col = st.selectbox(

                "Product column",

                choices,

                index=choices.index(mapping["product"])

                if mapping.get("product") in choices else 0,

            )



        with c3:

            quantity_col = st.selectbox(

                "Quantity column",

                choices,

                index=choices.index(mapping["quantity"])

                if mapping.get("quantity") in choices else 0,

            )

            order_col = st.selectbox(

                "Order ID column",

                choices,

                index=choices.index(mapping["order_id"])

                if mapping.get("order_id") in choices else 0,

            )

            discount_col = st.selectbox(

                "Discount column",

                choices,

                index=choices.index(mapping["discount"])

                if mapping.get("discount") in choices else 0,

            )



        currency = st.selectbox(

            "Currency display",

            ["$", "₹", "€", "£", "No symbol"],

            index=0,

        )



        submitted = st.form_submit_button(

            "Apply mapping and analyze",

            type="primary",

            use_container_width=True,

        )



    if submitted:

        if not date_col or not sales_col:

            st.error("Please select both a date column and a sales column.")

            st.stop()



        selected_mapping = {

            "date": date_col,

            "sales": sales_col,

            "profit": profit_col,

            "category": category_col,

            "region": region_col,

            "product": product_col,

            "quantity": quantity_col,

            "order_id": order_col,

            "discount": discount_col,

        }



        st.session_state["active_mapping"] = selected_mapping

        st.session_state["active_currency"] = currency

        st.session_state["active_source"] = source_label

        st.session_state["active_dataset"] = df.copy()

        st.session_state["analysis_ready"] = True

        st.session_state["active_source_mode"] = source

        st.rerun()



    st.stop()





# Restore the submitted upload mapping between Streamlit reruns.

if (

    st.session_state.get("analysis_ready")

    and st.session_state.get("active_source_mode") == "Upload a file"

):

    if source == "Upload a file" and uploaded_file is not None:

        if st.session_state.get("active_source") == uploaded_file.name:

            df = st.session_state["active_dataset"].copy()

            mapping = st.session_state["active_mapping"]



# The project dataset uses automatically detected canonical columns.

if source == "Project dataset":

    mapping = {

        field: guess_column(df.columns, field)

        for field in ALIASES

    }

    currency = "$"

    source_label = "Project dataset"

else:

    currency = st.session_state.get("active_currency", "$")

    source_label = st.session_state.get("active_source", source_label)



    if not st.session_state.get("analysis_ready"):

        page_header(

            "Upload & Analyze",

            "Import, map, and analyze your business data.",

        )

        st.info("Review your column mappings and select Apply mapping and analyze.")

        st.stop()



    mapping = st.session_state["active_mapping"]

    df = st.session_state["active_dataset"].copy()



# If a user switches the source, don't silently apply stale upload data.

if source == "Project dataset":

    st.session_state["active_source_mode"] = "Project dataset"





# ============================================================

# 6. BUILD A STANDARDIZED ANALYSIS DATAFRAME

# ============================================================

date_col = mapping.get("date")

sales_col = mapping.get("sales")

profit_col = mapping.get("profit")

category_col = mapping.get("category")

region_col = mapping.get("region")

product_col = mapping.get("product")

quantity_col = mapping.get("quantity")

order_col = mapping.get("order_id")

discount_col = mapping.get("discount")



if not date_col or not sales_col:

    page_header("Upload & Analyze", "Map the required business fields.")

    st.warning("A date column and sales column are required for this analysis.")

    st.stop()



analysis_df = pd.DataFrame(index=df.index)

analysis_df["date"] = pd.to_datetime(df[date_col], errors="coerce")

analysis_df["sales"] = clean_numeric(df[sales_col])



if profit_col:

    analysis_df["profit"] = clean_numeric(df[profit_col])



if quantity_col:

    analysis_df["quantity"] = clean_numeric(df[quantity_col])



if discount_col:

    analysis_df["discount"] = clean_numeric(df[discount_col])



analysis_df["category"] = (

    df[category_col].fillna("Unspecified").astype(str)

    if category_col else "All data"

)



analysis_df["region"] = (

    df[region_col].fillna("Unspecified").astype(str)

    if region_col else "All data"

)



analysis_df["product"] = (

    df[product_col].fillna("Unspecified").astype(str)

    if product_col else "Unspecified"

)



analysis_df["order_id"] = (

    df[order_col].astype(str)

    if order_col else None

)



analysis_df["source_row"] = df.index.astype(str)



analysis_df["year_month"] = analysis_df["date"].dt.strftime("%Y-%m")



# Preserve a quality snapshot before analytical exclusions.

missing_dates = int(analysis_df["date"].isna().sum())

invalid_sales = int(analysis_df["sales"].isna().sum())

duplicate_rows = int(df.duplicated().sum())

missing_cells = int(df.isna().sum().sum())

rows_before_analysis = len(analysis_df)



# Exclude rows that cannot contribute to date/revenue analysis.

valid_df = analysis_df.dropna(subset=["date", "sales"]).copy()

rows_excluded = rows_before_analysis - len(valid_df)



if valid_df.empty:

    st.error(

        "No rows have both a valid date and a numeric sales value. "

        "Check your column mapping and source data."

    )

    st.stop()





# ============================================================

# 7. FILTERS

# ============================================================

with st.sidebar:

    st.markdown("### ANALYSIS FILTERS")



    min_date = valid_df["date"].min().date()

    max_date = valid_df["date"].max().date()



    date_range = st.date_input(

        "Date range",

        value=(min_date, max_date),

        min_value=min_date,

        max_value=max_date,

    )



    categories = sorted(valid_df["category"].unique().tolist())

    regions = sorted(valid_df["region"].unique().tolist())



    chosen_categories = st.multiselect(

        "Category",

        categories,

        default=categories,

    )



    chosen_regions = st.multiselect(

        "Region",

        regions,

        default=regions,

    )



if isinstance(date_range, (tuple, list)) and len(date_range) == 2:

    start_date, end_date = date_range

else:

    start_date = min_date

    end_date = max_date



filtered = valid_df[

    valid_df["date"].dt.date.between(start_date, end_date)

    & valid_df["category"].isin(chosen_categories)

    & valid_df["region"].isin(chosen_regions)

].copy()



if filtered.empty:

    page_header("Sales Overview", "Business performance for the selected filters.")

    st.warning("No records match the selected filters. Widen the date range or selections.")

    st.stop()





# ============================================================

# 8. OVERVIEW

# ============================================================

if page == "Overview":

    page_header(

        "Sales Overview",

        "A consolidated view of revenue, profitability, and commercial performance.",

    )



    st.caption(

        f"Data source: {source_label}  |  "

        f"Reporting period: {start_date} to {end_date}  |  "

        f"{len(filtered):,} transaction rows"

    )



    revenue = filtered["sales"].sum()

    orders = (

        filtered["order_id"].nunique()

        if order_col else len(filtered)

    )

    average_order_value = revenue / orders if orders else 0



    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Total Revenue", fmt_money(revenue, "" if currency == "No symbol" else currency))

    k2.metric(

        "Total Profit",

        fmt_money(filtered["profit"].sum(), "" if currency == "No symbol" else currency)

        if "profit" in filtered else "Not available",

    )

    k3.metric("Orders" if order_col else "Transaction Rows", f"{orders:,.0f}")

    k4.metric(

        "Profit Margin",

        f"{(filtered['profit'].sum() / revenue * 100):.2f}%"

        if "profit" in filtered and revenue else "Not available",

    )



    st.markdown('<div class="section-title">Performance trends</div>',

                unsafe_allow_html=True)



    monthly = (

        filtered.groupby("year_month", as_index=False)

        .agg(Revenue=("sales", "sum"))

        .sort_values("year_month")

    )



    if "profit" in filtered:

        monthly_profit = (

            filtered.groupby("year_month", as_index=False)

            .agg(Profit=("profit", "sum"))

        )

        monthly = monthly.merge(monthly_profit, on="year_month", how="left")



    left, right = st.columns(2)



    with left:

        fig = px.line(

            monthly, x="year_month", y="Revenue",

            markers=True, title="Revenue over time",

        )

        fig.update_traces(line=dict(color=BLUE, width=3))

        fig.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="Revenue")

        st.plotly_chart(fig, use_container_width=True)



    with right:

        if "Profit" in monthly:

            fig = px.bar(

                monthly, x="year_month", y="Profit",

                title="Profit over time",

            )

            fig.update_traces(

                marker_color=[

                    "#16A34A" if val >= 0 else "#DC2626"

                    for val in monthly["Profit"]

                ]

            )

            fig.add_hline(y=0, line_dash="dash", line_color=MUTED)

            fig.update_layout(**CHART_LAYOUT, xaxis_title="", yaxis_title="Profit")

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.info(

                "No profit field was mapped. Map a profit column to enable "

                "profitability charts and metrics."

            )



    st.markdown('<div class="section-title">Category and regional performance</div>',

                unsafe_allow_html=True)



    left, right = st.columns(2)



    category_summary = (

        filtered.groupby("category", as_index=False)

        .agg(Revenue=("sales", "sum"))

        .sort_values("Revenue", ascending=True)

    )



    region_summary = (

        filtered.groupby("region", as_index=False)

        .agg(Revenue=("sales", "sum"))

        .sort_values("Revenue", ascending=True)

    )



    with left:

        fig = px.bar(

            category_summary, x="Revenue", y="category",

            orientation="h", title="Revenue by category",

            color_discrete_sequence=[BLUE],

        )

        fig.update_layout(**CHART_LAYOUT, xaxis_title="Revenue", yaxis_title="")

        st.plotly_chart(fig, use_container_width=True)



    with right:

        fig = px.bar(

            region_summary, x="Revenue", y="region",

            orientation="h", title="Revenue by region",

            color_discrete_sequence=[NAVY],

        )

        fig.update_layout(**CHART_LAYOUT, xaxis_title="Revenue", yaxis_title="")

        st.plotly_chart(fig, use_container_width=True)



    st.markdown('<div class="section-title">Product performance</div>',

                unsafe_allow_html=True)



    product_summary = (

        filtered.groupby("product", as_index=False)

        .agg(Revenue=("sales", "sum"))

        .sort_values("Revenue", ascending=False)

        .head(10)

        .sort_values("Revenue")

    )



    fig = px.bar(

        product_summary, x="Revenue", y="product",

        orientation="h", title="Top 10 products by revenue",

        color_discrete_sequence=[BLUE],

    )

    fig.update_layout(**CHART_LAYOUT, xaxis_title="Revenue", yaxis_title="")

    st.plotly_chart(fig, use_container_width=True)



    if "profit" in filtered:

        st.markdown('<div class="section-title">Profitability watchlist</div>',

                    unsafe_allow_html=True)



        loss_products = (

            filtered.groupby(["product", "category"], as_index=False)

            .agg(Revenue=("sales", "sum"), Profit=("profit", "sum"))

            .query("Profit < 0")

            .sort_values("Profit")

            .head(10)

        )



        if loss_products.empty:

            st.success("No loss-making products in the current selection.")

        else:

            st.dataframe(loss_products, use_container_width=True, hide_index=True)





# ============================================================

# 9. DATA EXPLORER

# ============================================================

elif page == "Data Explorer":

    page_header(

        "Data Explorer",

        "Search, inspect, and export records from the selected dataset.",

    )



    search = st.text_input(

        "Search across product, category, region, and order ID",

        placeholder="Enter a keyword",

    )



    explorer = filtered.copy()



    if search.strip():

        searchable = ["product", "category", "region", "order_id"]

        mask = pd.Series(False, index=explorer.index)



        for col in searchable:

            if col in explorer.columns:

                mask |= explorer[col].astype(str).str.contains(

                    search, case=False, na=False

                )



        explorer = explorer[mask]



    st.caption(f"{len(explorer):,} matching records")

    st.dataframe(explorer, use_container_width=True, hide_index=True)



    st.download_button(

        "Download filtered records",

        data=explorer.to_csv(index=False).encode("utf-8"),

        file_name="filtered_business_data.csv",

        mime="text/csv",

    )





# ============================================================

# 10. UPLOAD & ANALYZE

# ============================================================

elif page == "Upload & Analyze":

    page_header(

        "Upload & Analyze",

        "Review the imported dataset and the business fields used for analysis.",

    )



    st.metric("Records analyzed", f"{len(valid_df):,}")

    st.metric("Records in current filters", f"{len(filtered):,}")



    st.markdown("#### Active column mapping")



    mapping_rows = [

        {"Business field": field.replace("_", " ").title(),

         "Source column": col or "Not mapped"}

        for field, col in mapping.items()

    ]



    st.dataframe(

        pd.DataFrame(mapping_rows),

        use_container_width=True,

        hide_index=True,

    )



    st.markdown("#### Imported data preview")

    st.dataframe(df.head(100), use_container_width=True, hide_index=True)



    st.download_button(

        "Download analyzed data",

        data=filtered.to_csv(index=False).encode("utf-8"),

        file_name="analyzed_business_data.csv",

        mime="text/csv",

    )





# ============================================================

# 11. DATA QUALITY

# ============================================================

elif page == "Data Quality":

    page_header(

        "Data Quality",

        "Review missing values, duplicates, invalid records, and field completeness.",

    )



    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Source rows", f"{len(df):,}")

    q2.metric("Exact duplicate rows", f"{duplicate_rows:,}")

    q3.metric("Rows with missing values", f"{int(df.isna().any(axis=1).sum()):,}")

    q4.metric("Rows excluded from analysis", f"{rows_excluded:,}")



    st.markdown('<div class="section-title">Missing values by field</div>',

                unsafe_allow_html=True)



    missing_df = (

        df.isna().sum()

        .rename("Missing values")

        .reset_index()

        .rename(columns={"index": "Column"})

        .sort_values("Missing values", ascending=False)

    )



    missing_df["Missing %"] = (

        missing_df["Missing values"] / len(df) * 100

    ).round(2) if len(df) else 0



    left, right = st.columns([3, 2])



    with left:

        fig = px.bar(

            missing_df.head(20).sort_values("Missing values"),

            x="Missing values",

            y="Column",

            orientation="h",

            title="Missing values by column",

            color_discrete_sequence=[BLUE],

        )

        fig.update_layout(**CHART_LAYOUT, xaxis_title="Missing cells", yaxis_title="")

        st.plotly_chart(fig, use_container_width=True)



    with right:

        st.markdown("#### Field completeness")

        st.dataframe(

            missing_df,

            use_container_width=True,

            hide_index=True,

        )



    st.markdown('<div class="section-title">Validation summary</div>',

                unsafe_allow_html=True)



    validation = pd.DataFrame([

        {"Validation check": "Rows in source file",

         "Result": len(df),

         "Status": "Information"},

        {"Validation check": "Exact duplicate rows",

         "Result": duplicate_rows,

         "Status": "Review" if duplicate_rows else "Pass"},

        {"Validation check": "Missing dates",

         "Result": missing_dates,

         "Status": "Review" if missing_dates else "Pass"},

        {"Validation check": "Missing or invalid sales values",

         "Result": invalid_sales,

         "Status": "Review" if invalid_sales else "Pass"},

        {"Validation check": "Rows excluded from analysis",

         "Result": rows_excluded,

         "Status": "Review" if rows_excluded else "Pass"},

    ])



    st.dataframe(validation, use_container_width=True, hide_index=True)



    st.caption(

        "Quality checks flag potential issues; they do not automatically "

        "prove that every flagged record is incorrect. Review source context "

        "before deleting records."

    )



    st.download_button(

        "Download quality report",

        data=validation.to_csv(index=False).encode("utf-8"),

        file_name="data_quality_summary.csv",

        mime="text/csv",

    )





# ============================================================

# 12. FOOTER

# ============================================================

st.divider()

st.markdown(

    '<div class="source-note">Business Sales Analytics | '

    'Data-driven reporting and quality validation</div>',

    unsafe_allow_html=True,

)
