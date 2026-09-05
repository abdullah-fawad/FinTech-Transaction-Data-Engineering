import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "hk_transactions_with_canonical_v2.csv"
)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = "5432"
DB_NAME = "fintech_transactions"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")

BATCH_SIZE = 1000


# ============================================================
# HELPER FUNCTION
# ============================================================

def clean_value(value):

    if pd.isna(value):
        return None

    return value


# ============================================================
# START
# ============================================================

print("=" * 90)
print("LOADING TRANSACTIONS INTO POSTGRESQL")
print("=" * 90)

print(f"\nInput file:")
print(INPUT_FILE)


if not os.path.exists(INPUT_FILE):

    print("\nERROR: Input file does not exist.")
    print(INPUT_FILE)
    raise SystemExit


# ============================================================
# LOAD CSV
# ============================================================

print("\n" + "=" * 90)
print("LOADING CSV")
print("=" * 90)

df = pd.read_csv(INPUT_FILE)

print(f"Transactions loaded from CSV: {len(df):,}")


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\n" + "=" * 90)
print("BASIC VALIDATION")
print("=" * 90)


required_columns = [
    "id",
    "voucher_id",
    "consumer_id",
    "vch_type",
    "vch_amount",
    "vch_date",
    "vch_desc",
    "fc_amount",
    "fc_rate",
    "account_id",
    "record_created_on",
    "category_id",
    "use_case_title",
    "source_category_id",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    print("\nERROR: Required columns are missing:")

    for column in missing_columns:
        print(" -", column)

    raise SystemExit


print("Required columns: PASS")


# ============================================================
# CHECK VOUCHER ID
# ============================================================

print("\nChecking voucher_id...")

duplicate_voucher_ids = df["voucher_id"].duplicated().sum()

null_voucher_ids = df["voucher_id"].isna().sum()


print(f"Null voucher_id values:      {null_voucher_ids}")
print(f"Duplicate voucher_id values: {duplicate_voucher_ids}")


if null_voucher_ids > 0:

    print("\nERROR: voucher_id contains NULL values.")
    raise SystemExit


if duplicate_voucher_ids > 0:

    print("\nERROR: voucher_id contains duplicates.")
    raise SystemExit


print("voucher_id uniqueness: PASS")


# ============================================================
# CHECK TRANSACTION TYPES
# ============================================================

print("\nChecking transaction types...")

print(df["vch_type"].value_counts(dropna=False))


valid_types = {
    "Expense",
    "Income",
    "Transfer"
}


invalid_types = set(
    df["vch_type"].dropna().unique()
) - valid_types


if invalid_types:

    print("\nWARNING: Unexpected transaction types found:")

    for value in invalid_types:
        print(" -", value)

else:

    print("Transaction type validation: PASS")


# ============================================================
# DATA TYPE CONVERSION
# ============================================================

print("\n" + "=" * 90)
print("CONVERTING DATA TYPES")
print("=" * 90)


df["id"] = pd.to_numeric(
    df["id"],
    errors="coerce"
)

df["voucher_id"] = pd.to_numeric(
    df["voucher_id"],
    errors="coerce"
)

df["consumer_id"] = pd.to_numeric(
    df["consumer_id"],
    errors="coerce"
)

df["vch_amount"] = pd.to_numeric(
    df["vch_amount"],
    errors="coerce"
)

df["fc_amount"] = pd.to_numeric(
    df["fc_amount"],
    errors="coerce"
)

df["fc_rate"] = pd.to_numeric(
    df["fc_rate"],
    errors="coerce"
)

df["account_id"] = pd.to_numeric(
    df["account_id"],
    errors="coerce"
)

df["category_id"] = pd.to_numeric(
    df["category_id"],
    errors="coerce"
)

df["source_category_id"] = pd.to_numeric(
    df["source_category_id"],
    errors="coerce"
)


df["vch_date"] = pd.to_datetime(
    df["vch_date"],
    errors="coerce"
)


df["vch_created_on"] = pd.to_datetime(
    df["vch_created_on"],
    errors="coerce"
)


df["record_created_on"] = pd.to_datetime(
    df["record_created_on"],
    errors="coerce"
)


print("Numeric conversions completed.")
print("Date conversions completed.")


# ============================================================
# CHECK CONVERSION PROBLEMS
# ============================================================

print("\n" + "=" * 90)
print("POST-CONVERSION VALIDATION")
print("=" * 90)


print(
    "Invalid voucher_id:",
    df["voucher_id"].isna().sum()
)

print(
    "Invalid amount:",
    df["vch_amount"].isna().sum()
)

print(
    "Invalid transaction dates:",
    df["vch_date"].isna().sum()
)


# ============================================================
# CONNECT TO POSTGRESQL
# ============================================================

print("\n" + "=" * 90)
print("CONNECTING TO POSTGRESQL")
print("=" * 90)


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

    print("PostgreSQL connection: SUCCESS")


except Exception as e:

    print("\nERROR connecting to PostgreSQL:")
    print(e)

    raise SystemExit


# ============================================================
# CHECK DATABASE TABLE
# ============================================================

print("\nChecking transactions table...")

cursor.execute("""
SELECT COUNT(*)
FROM transactions;
""")

existing_count = cursor.fetchone()[0]

print(
    f"Existing transactions in database: {existing_count:,}"
)


# ============================================================
# PREPARE INSERT DATA
# ============================================================

print("\n" + "=" * 90)
print("PREPARING TRANSACTION RECORDS")
print("=" * 90)


insert_columns = [
    "voucher_id",
    "source_id",
    "consumer_id",
    "transaction_type",
    "amount",
    "transaction_date",
    "transaction_year",
    "transaction_month",
    "transaction_day",
    "created_on",
    "description",
    "foreign_currency_amount",
    "foreign_currency_rate",
    "account_id",
    "record_created_on",
    "original_category_id",
    "use_case_title",
    "source_category_id",
]


records = []


for _, row in df.iterrows():

    record = (

        clean_value(row["voucher_id"]),

        clean_value(row["id"]),

        clean_value(row["consumer_id"]),

        clean_value(row["vch_type"]),

        clean_value(row["vch_amount"]),

        clean_value(row["vch_date"]),

        clean_value(row["vch_year"]),

        clean_value(row["vch_month"]),

        clean_value(row["vch_day"]),

        clean_value(row["vch_created_on"]),

        clean_value(row["vch_desc"]),

        clean_value(row["fc_amount"]),

        clean_value(row["fc_rate"]),

        clean_value(row["account_id"]),

        clean_value(row["record_created_on"]),

        clean_value(row["category_id"]),

        clean_value(row["use_case_title"]),

        clean_value(row["source_category_id"]),
    )

    records.append(record)


print(f"Records prepared: {len(records):,}")


# ============================================================
# INSERT TRANSACTIONS
# ============================================================

print("\n" + "=" * 90)
print("INSERTING TRANSACTIONS")
print("=" * 90)


insert_sql = """
INSERT INTO transactions (
    voucher_id,
    source_id,
    consumer_id,
    transaction_type,
    amount,
    transaction_date,
    transaction_year,
    transaction_month,
    transaction_day,
    created_on,
    description,
    foreign_currency_amount,
    foreign_currency_rate,
    account_id,
    record_created_on,
    original_category_id,
    use_case_title,
    source_category_id
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s
)
ON CONFLICT (voucher_id)
DO UPDATE SET

    source_id = EXCLUDED.source_id,

    consumer_id = EXCLUDED.consumer_id,

    transaction_type = EXCLUDED.transaction_type,

    amount = EXCLUDED.amount,

    transaction_date = EXCLUDED.transaction_date,

    transaction_year = EXCLUDED.transaction_year,

    transaction_month = EXCLUDED.transaction_month,

    transaction_day = EXCLUDED.transaction_day,

    created_on = EXCLUDED.created_on,

    description = EXCLUDED.description,

    foreign_currency_amount =
        EXCLUDED.foreign_currency_amount,

    foreign_currency_rate =
        EXCLUDED.foreign_currency_rate,

    account_id = EXCLUDED.account_id,

    record_created_on =
        EXCLUDED.record_created_on,

    original_category_id =
        EXCLUDED.original_category_id,

    use_case_title =
        EXCLUDED.use_case_title,

    source_category_id =
        EXCLUDED.source_category_id;
"""


try:

    execute_batch(
        cursor,
        insert_sql,
        records,
        page_size=BATCH_SIZE
    )

    connection.commit()

    print(
        f"Successfully inserted/updated "
        f"{len(records):,} transactions."
    )


except Exception as e:

    connection.rollback()

    print("\nERROR during transaction loading:")
    print(e)

    cursor.close()
    connection.close()

    raise SystemExit


# ============================================================
# VERIFY LOADED DATA
# ============================================================

print("\n" + "=" * 90)
print("VERIFYING DATABASE LOAD")
print("=" * 90)


cursor.execute("""
SELECT COUNT(*)
FROM transactions;
""")

final_count = cursor.fetchone()[0]


print(
    f"Transactions in PostgreSQL: {final_count:,}"
)


cursor.execute("""
SELECT
    transaction_type,
    COUNT(*)
FROM transactions
GROUP BY transaction_type
ORDER BY transaction_type;
""")


print("\nTransaction type distribution:")

for transaction_type, count in cursor.fetchall():

    print(
        f"{transaction_type:<12} {count:,}"
    )


# ============================================================
# DATA QUALITY LOG
# ============================================================

cursor.execute("""
INSERT INTO data_quality_log (
    check_name,
    check_status,
    records_checked,
    records_failed,
    details
)
VALUES (
    %s,
    %s,
    %s,
    %s,
    %s
);
""", (
    "TRANSACTION_LOAD",
    "PASS" if final_count >= len(df) else "WARNING",
    len(df),
    0,
    f"Loaded {len(df):,} transaction records."
))


connection.commit()


# ============================================================
# CLOSE CONNECTION
# ============================================================

cursor.close()
connection.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 90)
print("POSTGRESQL DATA LOAD COMPLETE")
print("=" * 90)

print(
    f"\nSource records:    {len(df):,}"
)

print(
    f"Database records:  {final_count:,}"
)

print(
    f"Difference:        {len(df) - final_count:,}"
)


