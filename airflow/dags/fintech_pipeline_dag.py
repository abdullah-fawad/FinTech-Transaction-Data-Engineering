from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


PROJECT_ROOT = "/opt/airflow/project"
PYTHON = "python"


default_args = {
    "owner": "fintech-data-engineering",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="fintech_transaction_pipeline",
    description="Production FinTech transaction ETL pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fintech", "data-engineering", "transactions", "etl"],
) as dag:

    create_database_schema = BashOperator(
        task_id="create_postgres_schema",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/07_create_postgres_schema.py"
        ),
    )

    load_transactions = BashOperator(
        task_id="load_transactions_to_postgres",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/08_load_transactions_to_postgres.py"
        ),
    )

    validate_database = BashOperator(
        task_id="validate_postgres_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/09_validate_postgres_data.py"
        ),
    )

    inspect_production = BashOperator(
        task_id="inspect_production_data",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/10_inspect_production_data.py"
        ),
    )

    production_etl = BashOperator(
        task_id="production_etl",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/11_etl_production_pipeline.py"
        ),
    )

    create_analytics_views = BashOperator(
        task_id="create_analytics_views",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/12_create_analytics_views.py"
        ),
    )

    (
        create_database_schema
        >> load_transactions
        >> validate_database
        >> inspect_production
        >> production_etl
        >> create_analytics_views
    )