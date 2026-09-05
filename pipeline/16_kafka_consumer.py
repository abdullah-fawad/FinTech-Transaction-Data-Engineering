import json
import os
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer, KafkaProducer


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)

KAFKA_TOPIC = "transactions"

ROOT = Path(__file__).resolve().parent.parent

SOURCE_FILE = (
    ROOT
    / "data"
    / "raw"
    / "hk_transactions_table.csv"
)


# ============================================================
# KAFKA PRODUCER
# ============================================================

def create_kafka_producer():

    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,

        value_serializer=lambda transaction:
            json.dumps(transaction).encode("utf-8"),
    )


# ============================================================
# GET LATEST TRANSACTION
# ============================================================

def get_latest_transaction():

    if not SOURCE_FILE.exists():

        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}"
        )

    df = pd.read_csv(SOURCE_FILE)

    if df.empty:

        raise ValueError(
            "Source transaction file is empty."
        )

    row = df.iloc[-1]

    transaction = {
        "voucher_id": int(row["voucher_id"]),
        "consumer_id": int(row["consumer_id"]),
        "transaction_type": row["vch_type"],
        "amount": float(row["vch_amount"]),
        "transaction_date": str(row["vch_date"]),
        "description": str(row["vch_desc"]),
    }

    return transaction


# ============================================================
# PUBLISH TRANSACTION TO KAFKA
# ============================================================

def publish_transaction():

    print("=" * 60)
    print("KAFKA TRANSACTION PRODUCER")
    print("=" * 60)

    print(f"Kafka server: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topic:        {KAFKA_TOPIC}")
    print()

    transaction = get_latest_transaction()

    print("Transaction selected from simulated bank:")
    print()

    print(f"Voucher ID:       {transaction['voucher_id']}")
    print(f"Consumer ID:      {transaction['consumer_id']}")
    print(f"Transaction Type: {transaction['transaction_type']}")
    print(f"Amount:           {transaction['amount']}")
    print(f"Description:      {transaction['description']}")
    print(f"Transaction Date: {transaction['transaction_date']}")
    print()

    producer = None

    try:

        producer = create_kafka_producer()

        print("Kafka producer connected successfully.")

        future = producer.send(
            KAFKA_TOPIC,
            value=transaction
        )

        metadata = future.get(timeout=10)

        print()
        print("KAFKA PRODUCER: PASS")
        print(f"Topic:           {metadata.topic}")
        print(f"Partition:       {metadata.partition}")
        print(f"Offset:          {metadata.offset}")

    except Exception as error:

        print()
        print("KAFKA PRODUCER: FAIL")
        print(error)

        raise

    finally:

        if producer is not None:

            producer.flush()
            producer.close()

            print()
            print("Kafka producer closed.")


# ============================================================
# KAFKA CONSUMER
# ============================================================

def create_kafka_consumer():

    return KafkaConsumer(
        KAFKA_TOPIC,

        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,

        auto_offset_reset="earliest",

        enable_auto_commit=True,

        group_id="fintech-transaction-consumer",

        value_deserializer=lambda message:
            json.loads(message.decode("utf-8")),
    )


def consume_transactions():

    print("=" * 60)
    print("KAFKA TRANSACTION CONSUMER")
    print("=" * 60)

    print(f"Kafka server: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Topic: {KAFKA_TOPIC}")
    print()

    consumer = create_kafka_consumer()

    print("Consumer connected successfully.")
    print("Waiting for transactions...")
    print()

    try:

        for message in consumer:

            transaction = message.value

            print("-" * 60)
            print("NEW TRANSACTION RECEIVED")
            print("-" * 60)

            print(
                f"Voucher ID:       "
                f"{transaction.get('voucher_id')}"
            )

            print(
                f"Consumer ID:      "
                f"{transaction.get('consumer_id')}"
            )

            print(
                f"Transaction Type: "
                f"{transaction.get('transaction_type')}"
            )

            print(
                f"Amount:           "
                f"{transaction.get('amount')}"
            )

            print(
                f"Description:      "
                f"{transaction.get('description')}"
            )

            print(
                f"Transaction Date: "
                f"{transaction.get('transaction_date')}"
            )

            print()

    except KeyboardInterrupt:

        print()
        print("Consumer stopped by user.")

    finally:

        consumer.close()

        print("Kafka consumer closed.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    publish_transaction()

    print()
    print("=" * 60)
    print("STARTING KAFKA CONSUMER")
    print("=" * 60)
    print()

    consume_transactions()