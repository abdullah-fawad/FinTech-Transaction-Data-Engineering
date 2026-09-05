import os
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = ROOT / "data" / "raw" / "hk_transactions_table.csv"

DB_HOST = os.getenv("DB_HOST", "host.docker.internal")
DB_NAME = "fintech_transactions"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = 5432


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INCREMENTAL PRODUCTION ETL")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Read source
    # --------------------------------------------------------

    print("\n[1] Reading source data...")

    df = pd.read_csv(SOURCE_FILE)

    print(f"Source records: {len(df):,}")

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
        "use_case_title",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing source columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 2. Connect to PostgreSQL
    # --------------------------------------------------------

    print("\n[2] Connecting to PostgreSQL...")

    conn = get_connection()

    print("PostgreSQL connection: PASS")

    # --------------------------------------------------------
    # 3. Get existing voucher IDs
    # --------------------------------------------------------

    print("\n[3] Checking existing transactions...")

    cur = conn.cursor()

    cur.execute("""
        SELECT voucher_id
        FROM production_transactions;
    """)

    existing_ids = {
        row[0]
        for row in cur.fetchall()
    }

    print(f"Existing database records: {len(existing_ids):,}")

    # --------------------------------------------------------
    # 4. Detect new transactions
    # --------------------------------------------------------

    new_df = df[
        ~df["voucher_id"].isin(existing_ids)
    ].copy()

    print(f"New transactions detected: {len(new_df):,}")

    # --------------------------------------------------------
    # 5. No new data
    # --------------------------------------------------------

    if new_df.empty:

        print("\nNo new transactions found.")
        print("Incremental ETL STATUS: NO NEW DATA")

        cur.close()
        conn.close()

        return

    # --------------------------------------------------------
    # 6. Transform
    # --------------------------------------------------------

    print("\n[4] Transforming new transactions...")

    transformed = pd.DataFrame()

    transformed["voucher_id"] = new_df["voucher_id"]
    transformed["source_id"] = new_df["id"]
    transformed["consumer_id"] = new_df["consumer_id"]

    transformed["transaction_type"] = new_df["vch_type"]
    transformed["amount"] = new_df["vch_amount"]

    transformed["transaction_date"] = pd.to_datetime(
        new_df["vch_date"],
        errors="coerce"
    ).dt.date

    transformed["transaction_year"] = pd.to_numeric(
        new_df["vch_year"],
        errors="coerce"
    )

    transformed["transaction_month"] = pd.to_numeric(
        new_df["vch_month"],
        errors="coerce"
    )

    transformed["transaction_day"] = pd.to_numeric(
        new_df["vch_day"],
        errors="coerce"
    )

    transformed["created_on"] = pd.to_datetime(
        new_df["vch_created_on"],
        errors="coerce"
    )

    transformed["description"] = new_df["vch_desc"]

    transformed["foreign_currency_amount"] = new_df["fc_amount"]
    transformed["foreign_currency_rate"] = new_df["fc_rate"]

    transformed["account_id"] = new_df["account_id"]

    transformed["record_created_on"] = pd.to_datetime(
        new_df["record_created_on"],
        errors="coerce"
    )

    transformed["original_category_id"] = new_df["category_id"]

    transformed["use_case_title"] = new_df["use_case_title"]

    # --------------------------------------------------------
    # 7. Data-quality flags
    # --------------------------------------------------------

    transformed["missing_description"] = (
        transformed["description"].isna()
        | transformed["description"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    transformed["missing_transaction_date"] = (
        transformed["transaction_date"].isna()
    )

    transformed["negative_amount"] = (
        pd.to_numeric(
            transformed["amount"],
            errors="coerce"
        ) < 0
    )

    transformed["zero_amount"] = (
        pd.to_numeric(
            transformed["amount"],
            errors="coerce"
        ) == 0
    )

    transformed["invalid_transaction_type"] = (
        ~transformed["transaction_type"].isin(
            ["Income", "Expense"]
        )
    )

    today = pd.Timestamp.today().date()

    transformed["future_transaction_date"] = (
        transformed["transaction_date"].notna()
        & (
            transformed["transaction_date"] > today
        )
    )

    # --------------------------------------------------------
    # 8. Insert using execute_batch
    # --------------------------------------------------------

    print("\n[5] Loading new transactions into PostgreSQL...")

    columns = [
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
        "missing_description",
        "missing_transaction_date",
        "negative_amount",
        "zero_amount",
        "invalid_transaction_type",
        "future_transaction_date",
    ]

    insert_sql = f"""
        INSERT INTO production_transactions (
            {", ".join(columns)}
        )
        VALUES (
            {", ".join(["%s"] * len(columns))}
        )
        ON CONFLICT (voucher_id) DO NOTHING;
    """

    values = [
        tuple(row[column] for column in columns)
        for _, row in transformed.iterrows()
    ]

    execute_batch(
        cur,
        insert_sql,
        values,
        page_size=1000
    )

    conn.commit()

    # --------------------------------------------------------
    # 9. Determine inserted records
    # --------------------------------------------------------

    cur.execute("""
        SELECT COUNT(*)
        FROM production_transactions
        WHERE voucher_id = ANY(%s);
    """, (
        list(new_df["voucher_id"]),
    ))

    matching_records = cur.fetchone()[0]

    # --------------------------------------------------------
    # 10. Final count
    # --------------------------------------------------------

    cur.execute("""
        SELECT COUNT(*)
        FROM production_transactions;
    """)

    database_count = cur.fetchone()[0]

    # --------------------------------------------------------
    # 11. Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INCREMENTAL ETL RESULTS")
    print("=" * 70)

    print(f"Source records:              {len(df):,}")
    print(f"Existing database records:   {len(existing_ids):,}")
    print(f"New transactions detected:   {len(new_df):,}")
    print(f"New transactions inserted:   {matching_records:,}")
    print(f"Final database records:      {database_count:,}")

    if matching_records == len(new_df):

        print("\nINCREMENTAL ETL STATUS: PASS")

    else:

        print("\nINCREMENTAL ETL STATUS: CHECK")

    print("=" * 70)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()