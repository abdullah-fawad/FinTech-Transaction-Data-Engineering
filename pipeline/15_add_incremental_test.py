from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = ROOT / "data" / "raw" / "hk_transactions_table.csv"


df = pd.read_csv(SOURCE_FILE)

new_id = int(df["id"].max()) + 1
new_voucher_id = int(df["voucher_id"].max()) + 1

new_row = {
    "id": new_id,
    "voucher_id": new_voucher_id,
    "consumer_id": 331984,
    "vch_type": "Expense",
    "vch_amount": 2200,
    "vch_date": "2026-09-03",
    "vch_year": 2026,
    "vch_month": 9,
    "vch_day": 3,
    "vch_created_on": "2026-09-03 12:00:00",
    "vch_desc": "incremental test restaurant payment",
    "fc_amount": None,
    "fc_rate": None,
    "account_id": 1,
    "record_created_on": "2026-09-03 12:00:00",
    "category_id": 6,
    "use_case_title": "Expense",
}

df = pd.concat(
    [df, pd.DataFrame([new_row])],
    ignore_index=True
)

df.to_csv(SOURCE_FILE, index=False)

print("=" * 60)
print("INCREMENTAL TEST TRANSACTION ADDED")
print("=" * 60)
print(f"New ID:          {new_id}")
print(f"New voucher ID:  {new_voucher_id}")
print(f"Description:     {new_row['vch_desc']}")
print(f"Amount:          {new_row['vch_amount']}")
print(f"Total records:   {len(df):,}")
print("=" * 60)