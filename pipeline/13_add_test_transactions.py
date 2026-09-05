from pathlib import Path
import pandas as pd


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hk_transactions_table.csv"
)


# =============================================================================
# LOAD EXISTING DATA
# =============================================================================

print("=" * 90)
print("ADDING TEST TRANSACTIONS")
print("=" * 90)

print()
print(f"Loading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Existing transactions: {len(df):,}")


# =============================================================================
# DETERMINE NEW IDs
# =============================================================================

max_id = pd.to_numeric(df["id"], errors="coerce").max()
max_voucher_id = pd.to_numeric(
    df["voucher_id"],
    errors="coerce"
).max()

new_id_start = int(max_id) + 1
new_voucher_start = int(max_voucher_id) + 1


# =============================================================================
# CREATE NEW TEST TRANSACTIONS
# =============================================================================

test_transactions = [
    {
        "vch_type": "Expense",
        "vch_amount": 850,
        "vch_date": "2026-09-03",
        "vch_desc": "kfc dinner",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
    {
        "vch_type": "Expense",
        "vch_amount": 5000,
        "vch_date": "2026-09-03",
        "vch_desc": "shell petrol",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
    {
        "vch_type": "Expense",
        "vch_amount": 2500,
        "vch_date": "2026-09-03",
        "vch_desc": "grocery shopping at imtiaz",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
    {
        "vch_type": "Expense",
        "vch_amount": 3200,
        "vch_date": "2026-09-03",
        "vch_desc": "electricity bill payment",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
    {
        "vch_type": "Expense",
        "vch_amount": 1200,
        "vch_date": "2026-09-03",
        "vch_desc": "careem ride",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
    {
        "vch_type": "Expense",
        "vch_amount": 1800,
        "vch_date": "2026-09-03",
        "vch_desc": "medical pharmacy purchase",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
    {
        "vch_type": "Expense",
        "vch_amount": 4500,
        "vch_date": "2026-09-03",
        "vch_desc": "shopping at clothing store",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
    {
        "vch_type": "Expense",
        "vch_amount": 900,
        "vch_date": "2026-09-03",
        "vch_desc": "netflix subscription",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
    {
        "vch_type": "Income",
        "vch_amount": 75000,
        "vch_date": "2026-09-03",
        "vch_desc": "monthly salary received",
        "category_id": 4,
        "use_case_title": "Income",
        "consumer_id": 331984,
        "account_id": 4,
    },
    {
        "vch_type": "Expense",
        "vch_amount": 1500,
        "vch_date": "2026-09-03",
        "vch_desc": "restaurant dinner",
        "category_id": 6,
        "use_case_title": "Expense",
        "consumer_id": 331984,
        "account_id": 1,
    },
]


# =============================================================================
# BUILD ROWS USING EXISTING DATA STRUCTURE
# =============================================================================

new_rows = []

for i, transaction in enumerate(test_transactions):

    transaction_date = pd.Timestamp(transaction["vch_date"])

    row = {
        "id": new_id_start + i,
        "voucher_id": new_voucher_start + i,
        "consumer_id": transaction["consumer_id"],
        "vch_type": transaction["vch_type"],
        "vch_amount": transaction["vch_amount"],
        "vch_date": transaction["vch_date"],
        "vch_year": transaction_date.year,
        "vch_month": transaction_date.month,
        "vch_day": transaction_date.day,
        "vch_created_on": transaction["vch_date"],
        "vch_desc": transaction["vch_desc"],
        "fc_amount": 0,
        "fc_rate": 0,
        "account_id": transaction["account_id"],
        "record_created_on": transaction["vch_date"],
        "category_id": transaction["category_id"],
        "use_case_title": transaction["use_case_title"],
    }

    new_rows.append(row)


new_df = pd.DataFrame(new_rows)


# =============================================================================
# MATCH EXISTING COLUMN ORDER
# =============================================================================

new_df = new_df[df.columns]


# =============================================================================
# APPEND
# =============================================================================

updated_df = pd.concat(
    [df, new_df],
    ignore_index=True
)


# =============================================================================
# SAVE
# =============================================================================

updated_df.to_csv(
    INPUT_FILE,
    index=False
)


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

print()
print("=" * 90)
print("NEW TEST TRANSACTIONS")
print("=" * 90)

print(
    new_df[
        [
            "id",
            "voucher_id",
            "consumer_id",
            "vch_type",
            "vch_amount",
            "vch_date",
            "vch_desc",
        ]
    ].to_string(index=False)
)

print()
print("=" * 90)
print("RESULT")
print("=" * 90)

print(f"Previous transaction count: {len(df):,}")
print(f"New transactions added:     {len(new_df):,}")
print(f"New transaction count:      {len(updated_df):,}")

print()
print(f"Updated file:")
print(INPUT_FILE)

print()
print("STATUS: PASS")