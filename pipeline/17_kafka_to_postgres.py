import json
import os
from datetime import datetime
from decimal import Decimal

import psycopg2
from kafka import KafkaConsumer
from kafka.errors import KafkaError


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "localhost:9092"
)

KAFKA_TOPIC = "transactions"
KAFKA_GROUP_ID = "fintech-postgres-consumer"


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = 5432
DB_NAME = "fintech_transactions"
DB_USER = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ============================================================
# TRANSFORM TRANSACTION
# ============================================================

def transform_transaction(transaction):

    voucher_id = transaction.get("voucher_id")
    consumer_id = transaction.get("consumer_id")
    transaction_type = transaction.get("transaction_type")
    amount = transaction.get("amount")
    description = transaction.get("description")
    transaction_date = transaction.get("transaction_date")

    if voucher_id is None:
        raise ValueError("Missing voucher_id")

    if transaction_type not in ("Expense", "Income"):
        raise ValueError(
            f"Invalid transaction_type: {transaction_type}"
        )

    if amount is None:
        raise ValueError("Missing amount")

    if transaction_date:
        transaction_date = datetime.strptime(
            transaction_date,
            "%Y-%m-%d"
        ).date()

    return {
        "voucher_id": voucher_id,
        "consumer_id": consumer_id,
        "transaction_type": transaction_type,
        "amount": Decimal(str(amount)),
        "transaction_date": transaction_date,
        "description": description,
    }


# ============================================================
# INSERT INTO POSTGRESQL
# ============================================================

def insert_transaction(connection, transaction):

    query = """
        INSERT INTO production_transactions (
            voucher_id,
            consumer_id,
            transaction_type,
            amount,
            transaction_date,
            description
        )
        VALUES (
            %(voucher_id)s,
            %(consumer_id)s,
            %(transaction_type)s,
            %(amount)s,
            %(transaction_date)s,
            %(description)s
        )
        ON CONFLICT (voucher_id) DO NOTHING
        RETURNING voucher_id;
    """

    with connection.cursor() as cursor:

        cursor.execute(query, transaction)

        result = cursor.fetchone()

    connection.commit()

    return result


# ============================================================
# CREATE KAFKA CONSUMER
# ============================================================

def create_kafka_consumer():

    return KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,

        auto_offset_reset="latest",

        enable_auto_commit=True,

        group_id=KAFKA_GROUP_ID,

        # Receive raw bytes.
        # We manually decode JSON below.
        value_deserializer=None,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("KAFKA → POSTGRESQL TRANSACTION CONSUMER")
    print("=" * 60)

    print(f"Kafka server: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Kafka topic:  {KAFKA_TOPIC}")
    print(f"Consumer group: {KAFKA_GROUP_ID}")
    print(f"Database:     {DB_NAME}")
    print()

    # --------------------------------------------------------
    # PostgreSQL connection
    # --------------------------------------------------------

    try:

        connection = get_db_connection()

        print("PostgreSQL connection: PASS")

    except Exception as error:

        print("PostgreSQL connection: FAIL")
        print(error)

        return

    # --------------------------------------------------------
    # Kafka connection
    # --------------------------------------------------------

    try:

        consumer = create_kafka_consumer()

        print("Kafka consumer connection: PASS")
        print("Waiting for new transactions...")
        print()

    except KafkaError as error:

        print("Kafka consumer connection: FAIL")
        print(error)

        connection.close()

        return

    # --------------------------------------------------------
    # Consume transactions
    # --------------------------------------------------------

    try:

        for message in consumer:

            print("-" * 60)
            print("NEW KAFKA TRANSACTION")
            print("-" * 60)

            try:

                # ------------------------------------------------
                # Decode Kafka message
                # ------------------------------------------------

                raw_message = message.value.decode("utf-8")

                transaction = json.loads(raw_message)

                print(f"Kafka partition: {message.partition}")
                print(f"Kafka offset:    {message.offset}")

                print()
                print("Transaction:")
                print(transaction)

                # ------------------------------------------------
                # Transform
                # ------------------------------------------------

                transformed = transform_transaction(
                    transaction
                )

                print()
                print("Transformation: PASS")

                # ------------------------------------------------
                # PostgreSQL insert
                # ------------------------------------------------

                result = insert_transaction(
                    connection,
                    transformed
                )

                if result:

                    print(
                        "PostgreSQL INSERT: PASS "
                        f"(voucher_id={result[0]})"
                    )

                else:

                    print(
                        "PostgreSQL INSERT: SKIPPED "
                        "(voucher_id already exists)"
                    )

                print("Transaction processing: PASS")

            except json.JSONDecodeError as error:

                print("JSON validation: FAIL")
                print(f"Invalid Kafka message: {error}")

            except ValueError as error:

                connection.rollback()

                print("Transaction validation: FAIL")
                print(error)

            except Exception as error:

                connection.rollback()

                print("Transaction processing: FAIL")
                print(error)

            print()

    except KeyboardInterrupt:

        print()
        print("Consumer stopped by user.")

    except KafkaError as error:

        print()
        print("Kafka error:")
        print(error)

    finally:

        consumer.close()
        connection.close()

        print("Kafka consumer closed.")
        print("PostgreSQL connection closed.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()