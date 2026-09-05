import os
import psycopg2


# ============================================================
# CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = "5432"
DB_NAME = "fintech_transactions"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# CONNECT
# ============================================================

print("=" * 90)
print("POSTGRESQL DATA QUALITY VALIDATION")
print("=" * 90)

print("\nConnecting to PostgreSQL...")

try:

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    cursor = connection.cursor()

    print("Connection: PASS")

except Exception as e:

    print("\nERROR connecting to PostgreSQL:")
    print(e)

    raise SystemExit


# ============================================================
# HELPER
# ============================================================

def run_check(name, query):

    cursor.execute(query)

    result = cursor.fetchone()[0]

    print(f"{name:<50} {result:,}")

    return result


# ============================================================
# 1. TOTAL ROW COUNT
# ============================================================

print("\n" + "=" * 90)
print("1. RECORD COUNT")
print("=" * 90)

total_records = run_check(
    "Total transactions",
    """
    SELECT COUNT(*)
    FROM transactions;
    """
)


# ============================================================
# 2. DUPLICATE VOUCHER IDS
# ============================================================

print("\n" + "=" * 90)
print("2. DUPLICATE CHECK")
print("=" * 90)

duplicate_vouchers = run_check(
    "Duplicate voucher IDs",
    """
    SELECT COUNT(*)
    FROM (
        SELECT voucher_id
        FROM transactions
        GROUP BY voucher_id
        HAVING COUNT(*) > 1
    ) duplicates;
    """
)


# ============================================================
# 3. NULL PRIMARY KEY
# ============================================================

print("\n" + "=" * 90)
print("3. PRIMARY KEY CHECK")
print("=" * 90)

null_vouchers = run_check(
    "NULL voucher IDs",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE voucher_id IS NULL;
    """
)


# ============================================================
# 4. TRANSACTION TYPES
# ============================================================

print("\n" + "=" * 90)
print("4. TRANSACTION TYPE CHECK")
print("=" * 90)

cursor.execute("""
SELECT
    transaction_type,
    COUNT(*)
FROM transactions
GROUP BY transaction_type
ORDER BY transaction_type;
""")

transaction_types = cursor.fetchall()

for transaction_type, count in transaction_types:

    print(
        f"{str(transaction_type):<20} {count:,}"
    )


# ============================================================
# INVALID TRANSACTION TYPES
# ============================================================

invalid_types = run_check(
    "Invalid transaction types",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE transaction_type NOT IN
        ('Expense', 'Income', 'Transfer');
    """
)


# ============================================================
# 5. NULL DESCRIPTION
# ============================================================

print("\n" + "=" * 90)
print("5. DESCRIPTION QUALITY")
print("=" * 90)

null_descriptions = run_check(
    "NULL descriptions",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE description IS NULL;
    """
)


empty_descriptions = run_check(
    "Empty descriptions",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE TRIM(description) = '';
    """
)


# ============================================================
# 6. NULL AMOUNTS
# ============================================================

print("\n" + "=" * 90)
print("6. AMOUNT QUALITY")
print("=" * 90)

null_amounts = run_check(
    "NULL transaction amounts",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE amount IS NULL;
    """
)


negative_amounts = run_check(
    "Negative transaction amounts",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE amount < 0;
    """
)


zero_amounts = run_check(
    "Zero transaction amounts",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE amount = 0;
    """
)


# ============================================================
# 7. DATE QUALITY
# ============================================================

print("\n" + "=" * 90)
print("7. DATE QUALITY")
print("=" * 90)

null_dates = run_check(
    "NULL transaction dates",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE transaction_date IS NULL;
    """
)


future_dates = run_check(
    "Future transaction dates",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE transaction_date > CURRENT_DATE;
    """
)


# ============================================================
# 8. YEAR / MONTH CONSISTENCY
# ============================================================

print("\n" + "=" * 90)
print("8. DATE COMPONENT CONSISTENCY")
print("=" * 90)

invalid_year = run_check(
    "Incorrect transaction year",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE transaction_date IS NOT NULL
      AND transaction_year <>
          EXTRACT(YEAR FROM transaction_date);
    """
)


invalid_month = run_check(
    "Incorrect transaction month",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE transaction_date IS NOT NULL
      AND transaction_month <>
          EXTRACT(MONTH FROM transaction_date);
    """
)


invalid_day = run_check(
    "Incorrect transaction day",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE transaction_date IS NOT NULL
      AND transaction_day <>
          EXTRACT(DAY FROM transaction_date);
    """
)


# ============================================================
# 9. CATEGORY QUALITY
# ============================================================

print("\n" + "=" * 90)
print("9. CATEGORY QUALITY")
print("=" * 90)

null_categories = run_check(
    "NULL original category IDs",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE original_category_id IS NULL;
    """
)


# ============================================================
# 10. CONSUMER QUALITY
# ============================================================

print("\n" + "=" * 90)
print("10. CONSUMER QUALITY")
print("=" * 90)

null_consumers = run_check(
    "NULL consumer IDs",
    """
    SELECT COUNT(*)
    FROM transactions
    WHERE consumer_id IS NULL;
    """
)


# ============================================================
# 11. CATEGORY TABLE
# ============================================================

print("\n" + "=" * 90)
print("11. CATEGORY TABLE")
print("=" * 90)

category_count = run_check(
    "Canonical categories",
    """
    SELECT COUNT(*)
    FROM categories;
    """
)


cursor.execute("""
SELECT
    canonical_id,
    category_name
FROM categories
ORDER BY canonical_id;
""")

print("\nCanonical categories:")

for canonical_id, category_name in cursor.fetchall():

    print(
        f"{canonical_id}  {category_name}"
    )


# ============================================================
# 12. DATA QUALITY LOG
# ============================================================

print("\n" + "=" * 90)
print("12. DATA QUALITY LOG")
print("=" * 90)

cursor.execute("""
SELECT COUNT(*)
FROM data_quality_log;
""")

quality_log_count = cursor.fetchone()[0]

print(
    f"Existing quality log records: {quality_log_count:,}"
)


# ============================================================
# OVERALL RESULT
# ============================================================

print("\n" + "=" * 90)
print("OVERALL DATA QUALITY RESULT")
print("=" * 90)


checks = {

    "Duplicate voucher IDs": duplicate_vouchers,

    "NULL voucher IDs": null_vouchers,

    "Invalid transaction types": invalid_types,

    "Empty descriptions": empty_descriptions,

    "NULL amounts": null_amounts,

    "Negative amounts": negative_amounts,

    "NULL transaction dates": null_dates,

    "Future transaction dates": future_dates,

    "Invalid years": invalid_year,

    "Invalid months": invalid_month,

    "Invalid days": invalid_day,

}


failed_checks = []

for check_name, value in checks.items():

    if value > 0:

        failed_checks.append(
            (check_name, value)
        )


if len(failed_checks) == 0:

    print("\nOVERALL STATUS: PASS")

    print(
        "\nAll critical database quality checks passed."
    )

else:

    print("\nOVERALL STATUS: REVIEW")

    print("\nChecks requiring attention:")

    for check_name, value in failed_checks:

        print(
            f" - {check_name}: {value:,}"
        )


# ============================================================
# WRITE RESULT TO LOG
# ============================================================

status = (
    "PASS"
    if len(failed_checks) == 0
    else "REVIEW"
)


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

    "POSTGRESQL_DATA_QUALITY",

    status,

    total_records,

    sum(
        value
        for _, value in failed_checks
    ),

    "PostgreSQL transaction data quality validation."

))


connection.commit()


# ============================================================
# CLOSE
# ============================================================

cursor.close()
connection.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 90)
print("POSTGRESQL VALIDATION COMPLETE")
print("=" * 90)

print("\nRecords validated:")
print(f"{total_records:,}")

print("\nQuality checks completed:")
print(len(checks))
