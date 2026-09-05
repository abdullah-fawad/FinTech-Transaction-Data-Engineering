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
    dag_id="fintech_model_training",
    description="Build training data and train the ML transaction classifier",
    default_args=default_args,
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["fintech", "machine-learning", "training"],
) as dag:

    build_taxonomy = BashOperator(
        task_id="build_canonical_taxonomy",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/01_build_taxonomy.py"
        ),
    )

    apply_mappings = BashOperator(
        task_id="apply_approved_mappings",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/02_apply_approved_mappings.py"
        ),
    )

    build_ml_dataset = BashOperator(
        task_id="build_ml_training_dataset",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/03_build_ml_training_dataset.py"
        ),
    )

    train_ml_model = BashOperator(
        task_id="train_ml_classifier",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            f"{PYTHON} pipeline/04_train_ml_classifier.py"
        ),
    )

    (
        build_taxonomy
        >> apply_mappings
        >> build_ml_dataset
        >> train_ml_model
    )