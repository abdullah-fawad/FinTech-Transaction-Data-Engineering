import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = 5432
DB_NAME = "fintech_transactions"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# FILE CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CANONICAL_FILE = os.path.join(
    BASE_DIR,
    "data",
    "hk_transactions_with_canonical_v2.csv"
)


# ============================================================
# HELPER
# ============================================================

def print_section(title):

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def get_optional_column(df, candidates):

    for column in candidates:

        if column in df.columns:

            return column

    return None


# ============================================================
# START
# ============================================================

print_section("CREATING POSTGRESQL ANALYTICS VIEWS")

print("""
Production table:
    production_transactions

Categorization layer:
    transaction_categorizations
    categories

Analytics layer:
    ↓
    Monthly transaction summary
    Transaction type summary
    Consumer summary
    Data quality summary
    Financial KPI summary
    Category summary
    Transaction-category detail
""")


# ============================================================
# 1. CONNECT
# ============================================================

print_section("1. POSTGRESQL CONNECTION")

try:

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    connection.autocommit = False

    cursor = connection.cursor()

    print("PostgreSQL connection: PASS")

except Exception as e:

    print("\nERROR connecting to PostgreSQL:")
    print(e)

    raise SystemExit


# ============================================================
# 2. VERIFY PRODUCTION TABLE
# ============================================================

print_section("2. VERIFYING PRODUCTION TABLE")

cursor.execute("""
SELECT COUNT(*)
FROM production_transactions;
""")

production_count = cursor.fetchone()[0]

print(
    f"Production transactions: "
    f"{production_count:,}"
)

if production_count == 0:

    print("\nERROR: production_transactions is empty.")

    cursor.close()
    connection.close()

    raise SystemExit

print("Production table validation: PASS")


# ============================================================
# 3. LOAD CANONICAL CLASSIFICATIONS
# ============================================================

print_section("3. LOADING CANONICAL CLASSIFICATIONS")

print(
    f"Canonical file:\n"
    f"{CANONICAL_FILE}"
)

if not os.path.exists(CANONICAL_FILE):

    print("\nERROR: Canonical classification file not found.")

    cursor.close()
    connection.close()

    raise SystemExit


try:

    df = pd.read_csv(
        CANONICAL_FILE,
        low_memory=False
    )

    print(
        f"Canonical records loaded: "
        f"{len(df):,}"
    )

except Exception as e:

    print("\nERROR reading canonical CSV:")
    print(e)

    cursor.close()
    connection.close()

    raise SystemExit


# ------------------------------------------------------------
# Required columns
# ------------------------------------------------------------

required_columns = [

    "voucher_id",
    "canonical_id",
    "canonical_family",
    "canonical_mapping_status"

]


missing_columns = [

    column
    for column in required_columns
    if column not in df.columns

]


if missing_columns:

    print("\nERROR: Required columns missing:")

    for column in missing_columns:

        print(
            f"    {column}"
        )

    cursor.close()
    connection.close()

    raise SystemExit


# ------------------------------------------------------------
# Optional columns
# ------------------------------------------------------------

description_column = get_optional_column(
    df,
    [
        "vch_desc",
        "description",
        "transaction_description"
    ]
)

method_column = get_optional_column(
    df,
    [
        "canonical_mapping_method",
        "mapping_method",
        "classification_method"
    ]
)

matched_term_column = get_optional_column(
    df,
    [
        "canonical_matched_term",
        "matched_term"
    ]
)

confidence_column = get_optional_column(
    df,
    [
        "canonical_confidence",
        "ml_confidence",
        "confidence"
    ]
)

label_source_column = get_optional_column(
    df,
    [
        "canonical_label_source",
        "label_source"
    ]
)


print("\nDetected columns:")

print(
    f"Description:        "
    f"{description_column}"
)

print(
    f"Mapping method:     "
    f"{method_column}"
)

print(
    f"Matched term:       "
    f"{matched_term_column}"
)

print(
    f"Confidence:         "
    f"{confidence_column}"
)

print(
    f"Label source:       "
    f"{label_source_column}"
)


# ============================================================
# 4. PREPARE CATEGORIZATION DATA
# ============================================================

print_section("4. PREPARING TRANSACTION CATEGORIZATIONS")


def clean_value(value):

    if pd.isna(value):

        return None

    return value


categorization_rows = []


for _, row in df.iterrows():

    voucher_id = clean_value(
        row["voucher_id"]
    )

    canonical_id = clean_value(
        row["canonical_id"]
    )

    category_name = clean_value(
        row["canonical_family"]
    )

    mapping_status = clean_value(
        row["canonical_mapping_status"]
    )


    # --------------------------------------------------------
    # UNMAPPED TRANSACTIONS
    # --------------------------------------------------------

    if pd.isna(canonical_id):

        canonical_id = None

        category_name = "UNMAPPED"


    # --------------------------------------------------------
    # METHOD
    # --------------------------------------------------------

    if method_column:

        classification_method = clean_value(
            row[method_column]
        )

    else:

        classification_method = None


    # --------------------------------------------------------
    # DESCRIPTION
    # --------------------------------------------------------

    if description_column:

        description = clean_value(
            row[description_column]
        )

    else:

        description = None


    # --------------------------------------------------------
    # MATCHED TERM
    # --------------------------------------------------------

    if matched_term_column:

        matched_term = clean_value(
            row[matched_term_column]
        )

    else:

        matched_term = None


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    if confidence_column:

        confidence = clean_value(
            row[confidence_column]
        )

    else:

        confidence = None


    # --------------------------------------------------------
    # LABEL SOURCE
    # --------------------------------------------------------

    if label_source_column:

        label_source = clean_value(
            row[label_source_column]
        )

    else:

        label_source = None


    categorization_rows.append(
        (
            int(voucher_id),
            canonical_id,
            category_name,
            mapping_status,
            classification_method,
            matched_term,
            confidence,
            label_source
        )
    )


print(
    f"Prepared categorization records: "
    f"{len(categorization_rows):,}"
)


# ============================================================
# 5. REFRESH TRANSACTION CATEGORIZATIONS
# ============================================================

print_section(
    "5. LOADING TRANSACTION CATEGORIZATIONS"
)

try:

    # Remove previous categorization snapshot.
    # The canonical CSV is our current source of truth.

    cursor.execute("""
    DELETE FROM transaction_categorizations;
    """)

    deleted_count = cursor.rowcount

    print(
        f"Previous categorization records removed: "
        f"{deleted_count:,}"
    )


    insert_sql = """
        INSERT INTO transaction_categorizations
        (
            voucher_id,
            canonical_id,
            category_name,
            mapping_status,
            classification_method,
            matched_term,
            confidence,
            label_source
        )
        VALUES %s
    """


    execute_values(
        cursor,
        insert_sql,
        categorization_rows,
        page_size=5000
    )


    print(
        f"New categorization records inserted: "
        f"{len(categorization_rows):,}"
    )


    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    cursor.execute("""
    SELECT COUNT(*)
    FROM transaction_categorizations;
    """)

    categorization_count = cursor.fetchone()[0]

    print(
        f"transaction_categorizations rows: "
        f"{categorization_count:,}"
    )


    if categorization_count != production_count:

        print(
            "\nWARNING:"
        )

        print(
            "Categorization count does not match "
            "production transaction count."
        )

    else:

        print(
            "Categorization count validation: PASS"
        )


except Exception as e:

    connection.rollback()

    print(
        "\nERROR loading transaction categorizations:"
    )

    print(e)

    cursor.close()
    connection.close()

    raise SystemExit


# ============================================================
# 6. MONTHLY TRANSACTION SUMMARY
# ============================================================

print_section(
    "6. CREATING MONTHLY TRANSACTION SUMMARY"
)

cursor.execute("""
CREATE OR REPLACE VIEW vw_monthly_transaction_summary AS

SELECT

    transaction_year,

    transaction_month,

    transaction_type,

    COUNT(*) AS transaction_count,

    SUM(amount) AS total_amount,

    AVG(amount) AS average_amount,

    MIN(amount) AS minimum_amount,

    MAX(amount) AS maximum_amount

FROM production_transactions

GROUP BY

    transaction_year,
    transaction_month,
    transaction_type

ORDER BY

    transaction_year,
    transaction_month,
    transaction_type;
""")

print(
    "Created: "
    "vw_monthly_transaction_summary"
)


# ============================================================
# 7. TRANSACTION TYPE SUMMARY
# ============================================================

print_section(
    "7. CREATING TRANSACTION TYPE SUMMARY"
)

cursor.execute("""
CREATE OR REPLACE VIEW vw_transaction_type_summary AS

SELECT

    transaction_type,

    COUNT(*) AS transaction_count,

    SUM(amount) AS total_amount,

    AVG(amount) AS average_amount,

    MIN(amount) AS minimum_amount,

    MAX(amount) AS maximum_amount

FROM production_transactions

GROUP BY

    transaction_type

ORDER BY

    total_amount DESC;
""")

print(
    "Created: "
    "vw_transaction_type_summary"
)


# ============================================================
# 8. CONSUMER SUMMARY
# ============================================================

print_section(
    "8. CREATING CONSUMER SUMMARY"
)

cursor.execute("""
CREATE OR REPLACE VIEW vw_consumer_summary AS

SELECT

    consumer_id,

    COUNT(*) AS transaction_count,

    SUM(
        CASE
            WHEN transaction_type = 'Income'
            THEN amount
            ELSE 0
        END
    ) AS total_income,

    SUM(
        CASE
            WHEN transaction_type = 'Expense'
            THEN amount
            ELSE 0
        END
    ) AS total_expense,

    SUM(
        CASE
            WHEN transaction_type = 'Transfer'
            THEN amount
            ELSE 0
        END
    ) AS total_transfer,

    AVG(amount) AS average_transaction_amount,

    MIN(transaction_date) AS first_transaction_date,

    MAX(transaction_date) AS last_transaction_date

FROM production_transactions

WHERE consumer_id IS NOT NULL

GROUP BY

    consumer_id;
""")

print(
    "Created: "
    "vw_consumer_summary"
)


# ============================================================
# 9. DATA QUALITY SUMMARY
# ============================================================

print_section(
    "9. CREATING DATA QUALITY SUMMARY"
)

cursor.execute("""
CREATE OR REPLACE VIEW vw_data_quality_summary AS

SELECT

    COUNT(*) AS total_records,

    SUM(
        CASE
            WHEN missing_description = TRUE
            THEN 1
            ELSE 0
        END
    ) AS missing_descriptions,

    SUM(
        CASE
            WHEN missing_transaction_date = TRUE
            THEN 1
            ELSE 0
        END
    ) AS missing_transaction_dates,

    SUM(
        CASE
            WHEN negative_amount = TRUE
            THEN 1
            ELSE 0
        END
    ) AS negative_amounts,

    SUM(
        CASE
            WHEN zero_amount = TRUE
            THEN 1
            ELSE 0
        END
    ) AS zero_amounts,

    SUM(
        CASE
            WHEN invalid_transaction_type = TRUE
            THEN 1
            ELSE 0
        END
    ) AS invalid_transaction_types,

    SUM(
        CASE
            WHEN future_transaction_date = TRUE
            THEN 1
            ELSE 0
        END
    ) AS future_transaction_dates

FROM production_transactions;
""")

print(
    "Created: "
    "vw_data_quality_summary"
)


# ============================================================
# 10. FINANCIAL KPI SUMMARY
# ============================================================

print_section(
    "10. CREATING FINANCIAL KPI SUMMARY"
)

cursor.execute("""
CREATE OR REPLACE VIEW vw_financial_kpis AS

SELECT

    COUNT(*) AS total_transactions,

    COUNT(
        DISTINCT consumer_id
    ) AS total_consumers,

    SUM(
        CASE
            WHEN transaction_type = 'Income'
            THEN amount
            ELSE 0
        END
    ) AS total_income,

    SUM(
        CASE
            WHEN transaction_type = 'Expense'
            THEN amount
            ELSE 0
        END
    ) AS total_expense,

    SUM(
        CASE
            WHEN transaction_type = 'Transfer'
            THEN amount
            ELSE 0
        END
    ) AS total_transfers,

    AVG(amount) AS average_transaction_amount,

    MIN(transaction_date) AS earliest_transaction_date,

    MAX(transaction_date) AS latest_transaction_date

FROM production_transactions;
""")

print(
    "Created: "
    "vw_financial_kpis"
)


# ============================================================
# 11. TRANSACTION-CATEGORY DETAIL VIEW
# ============================================================

print_section(
    "11. CREATING TRANSACTION-CATEGORY DETAIL VIEW"
)

cursor.execute("""
CREATE OR REPLACE VIEW vw_transactions_with_category AS

SELECT

    p.voucher_id,

    p.consumer_id,

    p.transaction_date,

    p.transaction_year,

    p.transaction_month,

    p.transaction_type,

    p.amount,

    p.description,

    COALESCE(
    tc.canonical_id,
    NULL
    ) AS canonical_id,

    COALESCE(
        tc.category_name,
        'UNMAPPED'
    ) AS category_name,

    tc.mapping_status,

    tc.classification_method,

    tc.matched_term,

    tc.confidence,

    tc.label_source

FROM production_transactions p

LEFT JOIN transaction_categorizations tc

    ON p.voucher_id = tc.voucher_id;
""")

print(
    "Created: "
    "vw_transactions_with_category"
)


# ============================================================
# 12. CATEGORY SUMMARY
# ============================================================

print_section(
    "12. CREATING CATEGORY SUMMARY"
)

cursor.execute("""
CREATE OR REPLACE VIEW vw_category_summary AS

SELECT

    COALESCE(
        tc.category_name,
        'UNMAPPED'
    ) AS category_name,

    COUNT(*) AS transaction_count,

    SUM(
        CASE
            WHEN p.transaction_type = 'Income'
            THEN p.amount
            ELSE 0
        END
    ) AS total_income,

    SUM(
        CASE
            WHEN p.transaction_type = 'Expense'
            THEN ABS(p.amount)
            ELSE 0
        END
    ) AS total_expense,

    SUM(
        CASE
            WHEN p.transaction_type = 'Transfer'
            THEN ABS(p.amount)
            ELSE 0
        END
    ) AS total_transfer,

    SUM(
        ABS(p.amount)
    ) AS total_amount,

    AVG(
        ABS(p.amount)
    ) AS average_amount

FROM production_transactions p

LEFT JOIN transaction_categorizations tc

    ON p.voucher_id = tc.voucher_id

GROUP BY

    COALESCE(
        tc.category_name,
        'UNMAPPED'
    )

ORDER BY

    total_expense DESC;
""")

print(
    "Created: "
    "vw_category_summary"
)


# ============================================================
# 13. CATEGORY + TRANSACTION TYPE SUMMARY
# ============================================================

print_section(
    "13. CREATING CATEGORY TRANSACTION-TYPE SUMMARY"
)

cursor.execute("""
CREATE OR REPLACE VIEW vw_category_transaction_type_summary AS

SELECT

    COALESCE(
        tc.category_name,
        'UNMAPPED'
    ) AS category_name,

    p.transaction_type,

    COUNT(*) AS transaction_count,

    SUM(
        p.amount
    ) AS total_amount,

    SUM(
        CASE
            WHEN p.transaction_type = 'Expense'
            THEN ABS(p.amount)
            ELSE 0
        END
    ) AS total_expense

FROM production_transactions p

LEFT JOIN transaction_categorizations tc

    ON p.voucher_id = tc.voucher_id

GROUP BY

    COALESCE(
        tc.category_name,
        'UNMAPPED'
    ),

    p.transaction_type

ORDER BY

    category_name,

    transaction_type;
""")

print(
    "Created: "
    "vw_category_transaction_type_summary"
)


# ============================================================
# 14. COMMIT
# ============================================================

print_section(
    "14. SAVING ANALYTICS LAYER"
)

try:

    connection.commit()

    print(
        "Analytics layer committed successfully."
    )

except Exception as e:

    connection.rollback()

    print(
        "\nERROR committing analytics layer:"
    )

    print(e)

    cursor.close()
    connection.close()

    raise SystemExit


# ============================================================
# 15. VERIFY VIEWS
# ============================================================

print_section(
    "15. VERIFYING ANALYTICS VIEWS"
)

views = [

    "vw_monthly_transaction_summary",

    "vw_transaction_type_summary",

    "vw_consumer_summary",

    "vw_data_quality_summary",

    "vw_financial_kpis",

    "vw_transactions_with_category",

    "vw_category_summary",

    "vw_category_transaction_type_summary"

]


for view in views:

    try:

        cursor.execute(
            f"SELECT COUNT(*) FROM {view};"
        )

        count = cursor.fetchone()[0]

        print(
            f"{view:<45}"
            f"{count:,} rows"
        )

    except Exception as e:

        print(
            f"{view:<45}"
            f"ERROR"
        )

        print(e)


# ============================================================
# 16. CATEGORY SUMMARY PREVIEW
# ============================================================

print_section(
    "16. CATEGORY SUMMARY"
)

cursor.execute("""
SELECT

    category_name,

    transaction_count,

    total_income,

    total_expense,

    total_transfer

FROM vw_category_summary

ORDER BY

    total_expense DESC;
""")


category_rows = cursor.fetchall()


print(
    f"{'CATEGORY':<30}"
    f"{'COUNT':>15}"
    f"{'INCOME':>20}"
    f"{'EXPENSE':>20}"
)

print("-" * 90)


for row in category_rows:

    category = row[0]

    count = row[1]

    income = row[2]

    expense = row[3]


    print(
        f"{str(category):<30}"
        f"{count:>15,}"
        f"{str(income):>20}"
        f"{str(expense):>20}"
    )


# ============================================================
# 17. CATEGORIZATION SUMMARY
# ============================================================

print_section(
    "17. CATEGORIZATION SUMMARY"
)

cursor.execute("""
SELECT

    mapping_status,

    COUNT(*) AS transaction_count

FROM transaction_categorizations

GROUP BY

    mapping_status

ORDER BY

    transaction_count DESC;
""")


mapping_rows = cursor.fetchall()


print(
    f"{'MAPPING STATUS':<30}"
    f"{'COUNT':>20}"
)

print("-" * 55)


for row in mapping_rows:

    print(
        f"{str(row[0]):<30}"
        f"{row[1]:>20,}"
    )


# ============================================================
# 18. FINANCIAL KPI SUMMARY
# ============================================================

print_section(
    "18. FINANCIAL KPI SUMMARY"
)

cursor.execute("""
SELECT

    total_transactions,

    total_consumers,

    total_income,

    total_expense,

    total_transfers,

    average_transaction_amount,

    earliest_transaction_date,

    latest_transaction_date

FROM vw_financial_kpis;
""")

kpi = cursor.fetchone()


print(
    f"Total transactions:          "
    f"{kpi[0]:,}"
)

print(
    f"Total consumers:             "
    f"{kpi[1]:,}"
)

print(
    f"Total income:                "
    f"{kpi[2]}"
)

print(
    f"Total expense:               "
    f"{kpi[3]}"
)

print(
    f"Total transfers:             "
    f"{kpi[4]}"
)

print(
    f"Average transaction amount:  "
    f"{kpi[5]}"
)

print(
    f"Earliest transaction date:   "
    f"{kpi[6]}"
)

print(
    f"Latest transaction date:     "
    f"{kpi[7]}"
)


# ============================================================
# 19. CLOSE CONNECTION
# ============================================================

cursor.close()
connection.close()
