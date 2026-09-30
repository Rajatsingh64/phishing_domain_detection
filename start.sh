#!/bin/sh
set -e

echo "========================================"
echo "Starting container"
echo "========================================"

echo "Python version:"
python --version

echo "Airflow version:"
airflow version

echo "Checking psycopg2..."

python - <<'PY'
import importlib.util

spec = importlib.util.find_spec("psycopg2")

if spec is None:
    raise SystemExit(
        "ERROR: psycopg2 is not installed. "
        "Rebuild the Docker image."
    )

import psycopg2

print(f"psycopg2-ok: {psycopg2.__version__}")
PY


start_s3_sync() {

    echo "Starting S3 sync..."

    if [ -n "${BUCKET_NAME:-}" ]; then

        mkdir -p /app/saved_models

        aws s3 sync \
            "s3://${BUCKET_NAME}/saved_models" \
            /app/saved_models

        echo "Saved models sync complete."

    else

        echo "BUCKET_NAME is not set. Skipping S3 sync."

    fi
}


# ==========================================================
# AIRFLOW
# ==========================================================

if [ "$1" = "airflow" ]; then

    : "${AIRFLOW_EMAIL:?AIRFLOW_EMAIL is required}"
    : "${AIRFLOW_USERNAME:?AIRFLOW_USERNAME is required}"
    : "${AIRFLOW_PASSWORD:?AIRFLOW_PASSWORD is required}"

    start_s3_sync

    echo "Migrating Airflow database..."

    airflow db migrate

    echo "Airflow database migration completed."


    echo "Checking Airflow admin user..."

    if airflow users list 2>/dev/null | grep -q "$AIRFLOW_USERNAME"; then

        echo "Airflow admin user already exists."

    else

        echo "Creating Airflow admin user..."

        airflow users create \
            --username "$AIRFLOW_USERNAME" \
            --password "$AIRFLOW_PASSWORD" \
            --firstname "Airflow" \
            --lastname "Admin" \
            --role Admin \
            --email "$AIRFLOW_EMAIL"

        echo "Airflow admin user created."

    fi


    echo "========================================"
    echo "Starting Airflow"
    echo "========================================"


    echo "Starting Airflow scheduler..."

    airflow scheduler &
    SCHEDULER_PID=$!


    echo "Starting Airflow API server..."

    airflow api-server \
        --host 0.0.0.0 \
        --port 8080 &

    API_PID=$!


    echo "========================================"
    echo "Airflow started"
    echo "API server: http://0.0.0.0:8080"
    echo "Scheduler PID: $SCHEDULER_PID"
    echo "API PID: $API_PID"
    echo "========================================"


    while kill -0 "$SCHEDULER_PID" 2>/dev/null &&
          kill -0 "$API_PID" 2>/dev/null
    do
        sleep 5
    done


    echo "Airflow process stopped."

    exit 1


# ==========================================================
# STREAMLIT
# ==========================================================

elif [ "$1" = "streamlit" ]; then

    echo "========================================"
    echo "Starting Streamlit"
    echo "========================================"


    start_s3_sync


    echo "Starting Streamlit app..."


    exec streamlit run app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.enableCORS false


# ==========================================================
# UNKNOWN COMMAND
# ==========================================================

else

    echo "Unknown service: $1"

    exec "$@"

fi

