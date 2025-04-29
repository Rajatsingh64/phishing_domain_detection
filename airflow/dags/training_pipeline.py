import os
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

# Default args for retries etc.
default_args = {
    'retries': 2,
    'retry_delay': pendulum.duration(minutes=5),
}

def training(**kwargs):
    """Run the training pipeline."""
    from phishing.pipeline.training_pipeline import initiate_training_pipeline
    initiate_training_pipeline()

def sync_artifact_to_s3_bucket(**kwargs):
    """Sync artifacts and models to S3 bucket."""
    bucket_name = os.getenv("BUCKET_NAME")
    if bucket_name:
        os.system(f"aws s3 sync /app/artifacts s3://{bucket_name}/artifacts")
        os.system(f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models")
    else:
        raise ValueError("BUCKET_NAME environment variable not set.")

with DAG(
    dag_id="phishing_domain_detection",
    description="phishing domain training pipeline",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2025, 4, 29, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    concurrency=4,
    tags=["phishing", "mlops", "training"],
) as dag:

    training_pipeline_task = PythonOperator(
        task_id="mlops_training_pipeline",
        python_callable=training,
        provide_context=True,
    )

    sync_data_to_s3_task = PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifact_to_s3_bucket,
        provide_context=True,
    )

    # Define task dependencies
    training_pipeline_task >> sync_data_to_s3_task
