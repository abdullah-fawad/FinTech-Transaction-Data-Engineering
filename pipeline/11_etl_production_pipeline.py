import os
from io import StringIO

import boto3
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

BATCH_SIZE = 1000


# ============================================================
# AWS S3 CONFIGURATION
# ============================================================

S3_BUCKET = "fintech-transactions-storage"

S3_RAW_KEY = "raw/hk_transactions_table.csv"

S3_PROCESSED_KEY = "processed/transactions_processed.csv"


# ============================================================
# POSTGRESQL CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = 5432
DB_NAME = "fintech_transactions"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):

    if pd.isna(value):
        return None

    return value


def print_section(title):

    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


# ============================================================
# START
# ============================================================

print_section("PRODUCTION ETL PIPELINE")

print("\nData flow:")
print("AWS S3 RAW")
print("  ↓")
print("EXTRACT")
print("  ↓")
print("TRANSFORM")
print("  ↓")
print("DATA QUALITY FLAGS")
print("  ↓")
print("AWS S3 PROCESSED")
print("  ↓")
print("POSTGRESQL")
print("  ↓")
print("VALIDATE")


# ============================================================
# 1. EXTRACT FROM AWS S3
# ============================================================

print_section("1. EXTRACT FROM AWS S3")

print("\nS3 bucket:")
print(S3_BUCKET)

print("\nS3 raw object:")
print(S3_RAW_KEY)


try:

    s3 = boto3.client("s3")

    response = s3.get_object(
        Bucket=S3_BUCKET,
        Key=S3_RAW_KEY
    )

    raw_data = response["Body"].read()

    df = pd.read_csv(
        StringIO(
            raw_data.decode("utf-8")
        ),
        low_memory=False
    )

except Exception as e:

    print("\nERROR reading raw file from S3:")
    print(e)

    raise SystemExit


print(
    f"\nSource records: {len(df):,}"
)

print("S3 extraction: PASS")


# ============================================================
# 2. SOURCE SCHEMA VALIDATION
# ============================================================

print_section("2. SOURCE SCHEMA VALIDATION")


required_columns = [

    "id",
    "voucher_id",
    "consumer_id",

    "vch_type",
    "vch_amount",
    "vch_date",

    "vch_year",
    "vch_month",
    "vch_day",

    "vch_created_on",
    "vch_desc",

    "fc_amount",
    "fc_rate",

    "account_id",
    "record_created_on",

    "category_id",
    "use_case_title"

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
# 3. SOURCE KEY VALIDATION
# ============================================================

print_section("3. SOURCE KEY VALIDATION")


duplicate_vouchers = (
    df["voucher_id"].duplicated().sum()
)

null_vouchers = (
    df["voucher_id"].isna().sum()
)


print(
    f"Duplicate voucher IDs: {duplicate_vouchers:,}"
)

print(
    f"NULL voucher IDs:      {null_vouchers:,}"
)


if duplicate_vouchers > 0:

    print(
        "\nERROR: Duplicate voucher IDs detected."
    )

    raise SystemExit


if null_vouchers > 0:

    print(
        "\nERROR: NULL voucher IDs detected."
    )

    raise SystemExit


print("\nVoucher ID validation: PASS")


# ============================================================
# 4. TRANSFORM
# ============================================================

print_section("4. TRANSFORM")


# ------------------------------------------------------------
# NUMERIC FIELDS
# ------------------------------------------------------------

print("\nConverting numeric fields...")


numeric_columns = [

    "id",
    "voucher_id",
    "consumer_id",

    "vch_amount",

    "vch_year",
    "vch_month",
    "vch_day",

    "fc_amount",
    "fc_rate",

    "account_id",
    "category_id"

]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


print("Numeric conversion: COMPLETE")


# ------------------------------------------------------------
# DATE FIELDS
# ------------------------------------------------------------

print("\nConverting date fields...")


date_columns = [

    "vch_date",
    "vch_created_on",
    "record_created_on"

]


for column in date_columns:

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )


print("Date conversion: COMPLETE")


# ------------------------------------------------------------
# TEXT FIELDS
# ------------------------------------------------------------

print("\nCleaning text fields...")


text_columns = [

    "vch_type",
    "vch_desc",
    "use_case_title"

]


for column in text_columns:

    df[column] = df[column].astype("string")

    df[column] = (
        df[column]
        .str.strip()
    )


print("Text cleaning: COMPLETE")


# ============================================================
# 5. DATA QUALITY FLAGS
# ============================================================

print_section("5. DATA QUALITY FLAGS")


# ------------------------------------------------------------
# MISSING DESCRIPTION
# ------------------------------------------------------------

df["missing_description"] = (

    df["vch_desc"].isna()

    |

    (
        df["vch_desc"]
        .fillna("")
        .str.strip()
        == ""
    )

)


# ------------------------------------------------------------
# MISSING DATE
# ------------------------------------------------------------

df["missing_transaction_date"] = (

    df["vch_date"].isna()

)


# ------------------------------------------------------------
# NEGATIVE AMOUNT
# ------------------------------------------------------------

df["negative_amount"] = (

    df["vch_amount"] < 0

)


# ------------------------------------------------------------
# ZERO AMOUNT
# ------------------------------------------------------------

df["zero_amount"] = (

    df["vch_amount"] == 0

)


# ------------------------------------------------------------
# INVALID TRANSACTION TYPE
# ------------------------------------------------------------

valid_transaction_types = {

    "Expense",
    "Income",
    "Transfer"

}


df["invalid_transaction_type"] = (

    ~df["vch_type"].isin(
        valid_transaction_types
    )

)


# ------------------------------------------------------------
# FUTURE DATE
# ------------------------------------------------------------

today = pd.Timestamp.today().normalize()


df["future_transaction_date"] = (

    df["vch_date"] > today

)


print("Data quality flags: COMPLETE")


# ============================================================
# 6. TRANSFORMATION SUMMARY
# ============================================================

print_section("6. TRANSFORMATION SUMMARY")


print(
    f"Total records:             "
    f"{len(df):,}"
)

print(
    f"Missing descriptions:      "
    f"{df['missing_description'].sum():,}"
)

print(
    f"Missing transaction dates: "
    f"{df['missing_transaction_date'].sum():,}"
)

print(
    f"Negative amounts:          "
    f"{df['negative_amount'].sum():,}"
)

print(
    f"Zero amounts:              "
    f"{df['zero_amount'].sum():,}"
)

print(
    f"Invalid transaction types: "
    f"{df['invalid_transaction_type'].sum():,}"
)

print(
    f"Future transaction dates:  "
    f"{df['future_transaction_date'].sum():,}"
)


# ============================================================
# 7. PREPARE PRODUCTION DATASET
# ============================================================

print_section("7. PREPARING PRODUCTION DATASET")


production_columns = [

    "voucher_id",
    "id",
    "consumer_id",

    "vch_type",
    "vch_amount",
    "vch_date",

    "vch_year",
    "vch_month",
    "vch_day",

    "vch_created_on",
    "vch_desc",

    "fc_amount",
    "fc_rate",

    "account_id",
    "record_created_on",

    "category_id",
    "use_case_title",

    "missing_description",
    "missing_transaction_date",
    "negative_amount",
    "zero_amount",
    "invalid_transaction_type",
    "future_transaction_date"

]


production_df = df[
    production_columns
].copy()


print(
    f"Production records prepared: "
    f"{len(production_df):,}"
)


# ============================================================
# 8. SAVE PROCESSED DATA TO AWS S3
# ============================================================

print_section("8. SAVING PROCESSED DATA TO AWS S3")


try:

    processed_csv = production_df.to_csv(
        index=False
    )

    s3.put_object(

        Bucket=S3_BUCKET,

        Key=S3_PROCESSED_KEY,

        Body=processed_csv.encode("utf-8"),

        ContentType="text/csv"

    )

    print(
        "\nProcessed dataset uploaded successfully."
    )

    print(
        f"Bucket: {S3_BUCKET}"
    )

    print(
        f"Key:    {S3_PROCESSED_KEY}"
    )

    print("S3 processed upload: PASS")


except Exception as e:

    print(
        "\nERROR uploading processed data to S3:"
    )

    print(e)

    raise SystemExit


# ============================================================
# 9. POSTGRESQL CONNECTION
# ============================================================

print_section("9. POSTGRESQL CONNECTION")


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

    print(
        "\nERROR connecting to PostgreSQL:"
    )

    print(e)

    raise SystemExit


# ============================================================
# 10. CREATE PRODUCTION TABLE
# ============================================================

print_section("10. CREATING PRODUCTION TABLE")


cursor.execute("""

CREATE TABLE IF NOT EXISTS production_transactions (

    voucher_id BIGINT PRIMARY KEY,

    source_id BIGINT,

    consumer_id BIGINT,

    transaction_type VARCHAR(50),

    amount NUMERIC(18,2),

    transaction_date DATE,

    transaction_year INTEGER,

    transaction_month INTEGER,

    transaction_day INTEGER,

    created_on TIMESTAMP,

    description TEXT,

    foreign_currency_amount NUMERIC(18,2),

    foreign_currency_rate NUMERIC(18,6),

    account_id BIGINT,

    record_created_on TIMESTAMP,

    original_category_id NUMERIC,

    use_case_title VARCHAR(255),

    missing_description BOOLEAN,

    missing_transaction_date BOOLEAN,

    negative_amount BOOLEAN,

    zero_amount BOOLEAN,

    invalid_transaction_type BOOLEAN,

    future_transaction_date BOOLEAN,

    etl_loaded_at TIMESTAMP
        DEFAULT CURRENT_TIMESTAMP

);

""")


connection.commit()


print(
    "Production table: READY"
)


# ============================================================
# 11. INSERT DATA
# ============================================================

print_section("11. LOADING PRODUCTION DATA")


insert_sql = """

INSERT INTO production_transactions (

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

    missing_description,
    missing_transaction_date,
    negative_amount,
    zero_amount,
    invalid_transaction_type,
    future_transaction_date

)

VALUES (

    %s, %s, %s,

    %s, %s, %s,

    %s, %s, %s,

    %s, %s,

    %s, %s,

    %s, %s,

    %s, %s,

    %s, %s, %s, %s, %s, %s

)

ON CONFLICT (voucher_id)

DO UPDATE SET

    source_id =
        EXCLUDED.source_id,

    consumer_id =
        EXCLUDED.consumer_id,

    transaction_type =
        EXCLUDED.transaction_type,

    amount =
        EXCLUDED.amount,

    transaction_date =
        EXCLUDED.transaction_date,

    transaction_year =
        EXCLUDED.transaction_year,

    transaction_month =
        EXCLUDED.transaction_month,

    transaction_day =
        EXCLUDED.transaction_day,

    created_on =
        EXCLUDED.created_on,

    description =
        EXCLUDED.description,

    foreign_currency_amount =
        EXCLUDED.foreign_currency_amount,

    foreign_currency_rate =
        EXCLUDED.foreign_currency_rate,

    account_id =
        EXCLUDED.account_id,

    record_created_on =
        EXCLUDED.record_created_on,

    original_category_id =
        EXCLUDED.original_category_id,

    use_case_title =
        EXCLUDED.use_case_title,

    missing_description =
        EXCLUDED.missing_description,

    missing_transaction_date =
        EXCLUDED.missing_transaction_date,

    negative_amount =
        EXCLUDED.negative_amount,

    zero_amount =
        EXCLUDED.zero_amount,

    invalid_transaction_type =
        EXCLUDED.invalid_transaction_type,

    future_transaction_date =
        EXCLUDED.future_transaction_date,

    etl_loaded_at =
        CURRENT_TIMESTAMP;

"""


records = []


for _, row in production_df.iterrows():

    records.append((

        clean_value(
            row["voucher_id"]
        ),

        clean_value(
            row["id"]
        ),

        clean_value(
            row["consumer_id"]
        ),

        clean_value(
            row["vch_type"]
        ),

        clean_value(
            row["vch_amount"]
        ),

        clean_value(
            row["vch_date"]
        ),

        clean_value(
            row["vch_year"]
        ),

        clean_value(
            row["vch_month"]
        ),

        clean_value(
            row["vch_day"]
        ),

        clean_value(
            row["vch_created_on"]
        ),

        clean_value(
            row["vch_desc"]
        ),

        clean_value(
            row["fc_amount"]
        ),

        clean_value(
            row["fc_rate"]
        ),

        clean_value(
            row["account_id"]
        ),

        clean_value(
            row["record_created_on"]
        ),

        clean_value(
            row["category_id"]
        ),

        clean_value(
            row["use_case_title"]
        ),

        clean_value(
            row["missing_description"]
        ),

        clean_value(
            row["missing_transaction_date"]
        ),

        clean_value(
            row["negative_amount"]
        ),

        clean_value(
            row["zero_amount"]
        ),

        clean_value(
            row["invalid_transaction_type"]
        ),

        clean_value(
            row["future_transaction_date"]
        )

    ))


print(
    f"Records ready for insertion: "
    f"{len(records):,}"
)


try:

    execute_batch(

        cursor,
        insert_sql,
        records,
        page_size=BATCH_SIZE

    )

    connection.commit()

    print(
        f"\nSuccessfully loaded "
        f"{len(records):,} records."
    )


except Exception as e:

    connection.rollback()

    print(
        "\nERROR loading production data:"
    )

    print(e)

    cursor.close()
    connection.close()

    raise SystemExit


# ============================================================
# 12. POST-LOAD VALIDATION
# ============================================================

print_section("12. POST-LOAD VALIDATION")


# ------------------------------------------------------------
# RECORD COUNT
# ------------------------------------------------------------

cursor.execute("""

SELECT COUNT(*)

FROM production_transactions;

""")


database_count = cursor.fetchone()[0]


print(
    f"Production table records: "
    f"{database_count:,}"
)


# ------------------------------------------------------------
# TRANSACTION TYPES
# ------------------------------------------------------------

cursor.execute("""

SELECT

    transaction_type,
    COUNT(*)

FROM production_transactions

GROUP BY transaction_type

ORDER BY transaction_type;

""")


print("\nTransaction type distribution:")


for transaction_type, count in cursor.fetchall():

    print(
        f"{str(transaction_type):<15}"
        f"{count:,}"
    )


# ------------------------------------------------------------
# DATA QUALITY SUMMARY
# ------------------------------------------------------------

print("\nData quality summary:")


cursor.execute("""

SELECT

    SUM(
        CASE
            WHEN missing_description
            THEN 1
            ELSE 0
        END
    ),

    SUM(
        CASE
            WHEN missing_transaction_date
            THEN 1
            ELSE 0
        END
    ),

    SUM(
        CASE
            WHEN negative_amount
            THEN 1
            ELSE 0
        END
    ),

    SUM(
        CASE
            WHEN zero_amount
            THEN 1
            ELSE 0
        END
    )

FROM production_transactions;

""")


quality_result = cursor.fetchone()


print(
    f"Missing descriptions:       "
    f"{quality_result[0]:,}"
)

print(
    f"Missing transaction dates:  "
    f"{quality_result[1]:,}"
)

print(
    f"Negative amounts:           "
    f"{quality_result[2]:,}"
)

print(
    f"Zero amounts:               "
    f"{quality_result[3]:,}"
)


# ============================================================
# 13. ETL STATUS
# ============================================================

print_section("13. ETL STATUS")


etl_status = (

    "PASS"

    if database_count == len(
        production_df
    )

    else "REVIEW"

)


print(
    f"Source records:       "
    f"{len(df):,}"
)

print(
    f"Database records:     "
    f"{database_count:,}"
)

print(
    f"Record difference:    "
    f"{len(df) - database_count:,}"
)

print(
    f"\nETL STATUS: {etl_status}"
)


# ============================================================
# 14. DATA QUALITY LOG
# ============================================================

print_section("14. DATA QUALITY LOG")


cursor.execute("""

SELECT EXISTS (

    SELECT 1

    FROM information_schema.tables

    WHERE table_name = 'data_quality_log'

);

""")


quality_log_exists = cursor.fetchone()[0]


if quality_log_exists:

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

        "PRODUCTION_ETL_LOAD",

        etl_status,

        len(production_df),

        abs(
            len(production_df)
            - database_count
        ),

        "Raw transaction CSV extracted from AWS S3, "
        "transformed, quality flags generated, "
        "processed dataset stored in AWS S3, and "
        "production data loaded into PostgreSQL."

    ))

    connection.commit()

    print(
        "ETL quality log: RECORDED"
    )

else:

    print(
        "data_quality_log table not found."
    )

    print(
        "ETL quality log: SKIPPED"
    )


# ============================================================
# 15. CLOSE DATABASE CONNECTION
# ============================================================

cursor.close()
connection.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print_section("PRODUCTION ETL COMPLETE")


print(
    f"\nS3 raw source:       "
    f"s3://{S3_BUCKET}/{S3_RAW_KEY}"
)

print(
    f"S3 processed output: "
    f"s3://{S3_BUCKET}/{S3_PROCESSED_KEY}"
)

print(
    f"\nSource records:       "
    f"{len(df):,}"
)

print(
    f"Transformed records:  "
    f"{len(production_df):,}"
)

print(
    f"Database records:     "
    f"{database_count:,}"
)

print(
    f"Record difference:    "
    f"{len(production_df) - database_count:,}"
)

print(
    f"\nETL STATUS: {etl_status}"
)