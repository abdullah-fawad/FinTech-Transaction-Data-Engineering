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
    dag_id="fintech_incremental_pipeline",
    description="Incremental FinTech transaction ETL pipeline",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fintech", "data-engineering", "incremental", "transactions"],
) as dag:

    incremental_etl = BashOperator(
        task_id="incremental_production_etl",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/14_incremental_etl.py"
        ),
    )

    refresh_analytics = BashOperator(
        task_id="refresh_analytics_views",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/12_create_analytics_views.py"
        ),
    )

    incremental_etl >> refresh_analytics