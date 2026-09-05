import os
import psycopg2


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = "5432"
DB_NAME = "fintech_transactions"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# CONNECT TO POSTGRESQL
# ============================================================

print("=" * 90)
print("CREATING POSTGRESQL DATABASE SCHEMA")
print("=" * 90)

print("\nConnecting to PostgreSQL...")

try:
    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname="postgres"
    )

    connection.autocommit = True

    cursor = connection.cursor()

    print("PostgreSQL connection successful.")

except Exception as e:
    print("\nERROR: Could not connect to PostgreSQL.")
    print(e)
    raise SystemExit


# ============================================================
# CREATE DATABASE
# ============================================================

print("\n" + "=" * 90)
print("CREATING DATABASE")
print("=" * 90)

cursor.execute(
    "SELECT 1 FROM pg_database WHERE datname = %s",
    (DB_NAME,)
)

database_exists = cursor.fetchone()

if database_exists:
    print(f"Database '{DB_NAME}' already exists.")
else:
    cursor.execute(
        f'CREATE DATABASE "{DB_NAME}"'
    )
    print(f"Database '{DB_NAME}' created successfully.")


cursor.close()
connection.close()


# ============================================================
# CONNECT TO NEW DATABASE
# ============================================================

print("\nConnecting to fintech_transactions...")

connection = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    dbname=DB_NAME
)

connection.autocommit = True

cursor = connection.cursor()

print("Connected successfully.")


# ============================================================
# CREATE SCHEMA
# ============================================================

print("\n" + "=" * 90)
print("CREATING TABLES")
print("=" * 90)


# ------------------------------------------------------------
# TRANSACTIONS
# ------------------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS transactions (

    voucher_id BIGINT PRIMARY KEY,

    source_id BIGINT,

    consumer_id BIGINT,

    transaction_type VARCHAR(20),

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

    original_category_id BIGINT,

    use_case_title TEXT,

    source_category_id BIGINT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

);
""")

print("transactions table created.")


# ------------------------------------------------------------
# CATEGORIES
# ------------------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS categories (

    canonical_id INTEGER PRIMARY KEY,

    category_name VARCHAR(50) NOT NULL,

    description TEXT

);
""")

print("categories table created.")


# ------------------------------------------------------------
# TRANSACTION CATEGORIZATION
# ------------------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS transaction_categorizations (

    voucher_id BIGINT PRIMARY KEY,

    canonical_id INTEGER,

    category_name VARCHAR(50),

    mapping_status VARCHAR(50),

    classification_method VARCHAR(50),

    matched_term TEXT,

    confidence NUMERIC(5,4),

    label_source VARCHAR(50),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_transaction
        FOREIGN KEY (voucher_id)
        REFERENCES transactions(voucher_id),

    CONSTRAINT fk_category
        FOREIGN KEY (canonical_id)
        REFERENCES categories(canonical_id)

);
""")

print("transaction_categorizations table created.")


# ------------------------------------------------------------
# DATA QUALITY LOG
# ------------------------------------------------------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS data_quality_log (

    quality_check_id SERIAL PRIMARY KEY,

    check_name VARCHAR(100),

    check_status VARCHAR(20),

    records_checked BIGINT,

    records_failed BIGINT,

    check_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    details TEXT

);
""")

print("data_quality_log table created.")


# ============================================================
# INSERT CANONICAL CATEGORIES
# ============================================================

print("\n" + "=" * 90)
print("INSERTING CANONICAL CATEGORIES")
print("=" * 90)


categories = [
    (2001, "Dining", "Food, restaurants and dining expenses"),
    (2002, "Fuel", "Fuel and petrol related transactions"),
    (2003, "Transport", "Transportation and travel expenses"),
    (2004, "Utilities", "Utility bills and household services"),
    (2005, "Healthcare", "Medical and healthcare expenses"),
    (2006, "Shopping", "Shopping and retail purchases"),
    (2007, "Transfers", "Money transfers and cash movements"),
    (2008, "Income", "Salary and other income"),
    (2009, "Groceries", "Grocery and household food purchases"),
]


cursor.executemany("""
INSERT INTO categories
    (canonical_id, category_name, description)
VALUES
    (%s, %s, %s)
ON CONFLICT (canonical_id)
DO UPDATE SET
    category_name = EXCLUDED.category_name,
    description = EXCLUDED.description;
""", categories)

print(f"Canonical categories inserted: {len(categories)}")


# ============================================================
# INDEXES
# ============================================================

print("\n" + "=" * 90)
print("CREATING INDEXES")
print("=" * 90)


indexes = [

    """
    CREATE INDEX IF NOT EXISTS idx_transactions_consumer
    ON transactions(consumer_id);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_transactions_date
    ON transactions(transaction_date);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_transactions_type
    ON transactions(transaction_type);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_transactions_category
    ON transactions(original_category_id);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_categorization_category
    ON transaction_categorizations(canonical_id);
    """,

    """
    CREATE INDEX IF NOT EXISTS idx_categorization_method
    ON transaction_categorizations(classification_method);
    """

]


for index_sql in indexes:
    cursor.execute(index_sql)

print("Indexes created successfully.")


# ============================================================
# VERIFY TABLES
# ============================================================

print("\n" + "=" * 90)
print("VERIFYING DATABASE")
print("=" * 90)

cursor.execute("""
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;
""")

tables = cursor.fetchall()

print("\nTables currently in database:")

for table in tables:
    print(" -", table[0])


# ============================================================
# CLOSE
# ============================================================

cursor.close()
connection.close()

print("\n" + "=" * 90)
print("POSTGRESQL SCHEMA CREATION COMPLETE")
print("=" * 90)

print("\nDatabase:")
print(DB_NAME)

print("\nTables created:")
print("1. transactions")
print("2. categories")
print("3. transaction_categorizations")
print("4. data_quality_log")
