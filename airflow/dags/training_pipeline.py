import os
import subprocess

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator


# ==================================================
# Default arguments
# ==================================================

default_args = {
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=5),
}


# ==================================================
# Training
# ==================================================

def training():
    """Run the training pipeline."""

    from phishing.pipeline.training_pipeline import (
        initiate_training_pipeline,
    )

    initiate_training_pipeline()


# ==================================================
# Sync artifacts and models to S3
# ==================================================

def sync_artifact_to_s3_bucket():
    """Sync artifacts and saved models to S3."""

    bucket_name = os.getenv("BUCKET_NAME")

    if not bucket_name:
        raise ValueError(
            "BUCKET_NAME environment variable is not set."
        )

    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "/app/artifacts",
            f"s3://{bucket_name}/artifacts",
        ],
        check=True,
    )

    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            "/app/saved_models",
            f"s3://{bucket_name}/saved_models",
        ],
        check=True,
    )


# ==================================================
# DAG
# ==================================================

with DAG(
    dag_id="phishing_domain_detection",
    description="Phishing domain training pipeline",
    default_args=default_args,
    schedule="@weekly",
    start_date=pendulum.datetime(
        2025,
        4,
        29,
        tz="UTC",
    ),
    catchup=False,
    max_active_runs=1,
    tags=["phishing", "mlops", "training"],
) as dag:

    training_pipeline_task = PythonOperator(
        task_id="mlops_training_pipeline",
        python_callable=training,
    )

    sync_data_to_s3_task = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifact_to_s3_bucket,
    )

    training_pipeline_task >> sync_data_to_s3_task

