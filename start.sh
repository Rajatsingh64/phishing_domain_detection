#!/bin/sh

set -e

# ==================================================
# Required environment variables
# ==================================================

: "${AIRFLOW_EMAIL:?AIRFLOW_EMAIL is required}"
: "${AIRFLOW_USERNAME:?AIRFLOW_USERNAME is required}"
: "${AIRFLOW_PASSWORD:?AIRFLOW_PASSWORD is required}"
: "${AWS_REGION:?AWS_REGION is required}"
: "${BUCKET_NAME:?BUCKET_NAME is required}"

# ==================================================
# AIRFLOW
# ==================================================

if [ "$1" = "airflow" ]; then

    echo "========================================"
    echo "Starting Airflow"
    echo "========================================"

    echo "Python version:"
    python --version

    echo "Airflow version:"
    airflow version

    # --------------------------------------------------
    # Check psycopg2
    # --------------------------------------------------

    echo "Checking psycopg2..."

    python - <<'PY'
import importlib.util

spec = importlib.util.find_spec("psycopg2")

if spec is None:
    raise SystemExit(
        "ERROR: psycopg2 is not installed."
    )

import psycopg2

print(f"psycopg2-ok: {psycopg2.__version__}")
PY

    # --------------------------------------------------
    # S3 sync
    # --------------------------------------------------

    echo "Starting S3 sync..."

    mkdir -p /app/saved_models

    aws s3 sync \
        "s3://${BUCKET_NAME}/saved_models" \
        /app/saved_models

    echo "Saved models sync complete."

    # --------------------------------------------------
    # Airflow database migration
    # --------------------------------------------------

    echo "Migrating Airflow database..."

    airflow db migrate

    echo "Airflow database migration completed."

    # --------------------------------------------------
    # Create Airflow admin user
    # --------------------------------------------------

    echo "Checking Airflow admin user..."

    if airflow users list 2>/dev/null | grep -w "$AIRFLOW_USERNAME" > /dev/null 2>&1; then

        echo "Airflow user already exists."

    else

        echo "Creating Airflow admin user..."

        airflow users create \
            --username "$AIRFLOW_USERNAME" \
            --firstname "Admin" \
            --lastname "User" \
            --role Admin \
            --email "$AIRFLOW_EMAIL" \
            --password "$AIRFLOW_PASSWORD"

        echo "Airflow admin user created."

    fi

    # --------------------------------------------------
    # Start scheduler
    # --------------------------------------------------

    echo "Starting Airflow scheduler..."

    airflow scheduler &
    SCHEDULER_PID=$!

    # --------------------------------------------------
    # Start API server
    # --------------------------------------------------

    echo "Starting Airflow API server..."

    airflow api-server \
        --host 0.0.0.0 \
        --port 8080 &

    API_PID=$!

    # --------------------------------------------------
    # Status
    # --------------------------------------------------

    echo "========================================"
    echo "Airflow started"
    echo "========================================"
    echo "API server: 0.0.0.0:8080"
    echo "Scheduler PID: $SCHEDULER_PID"
    echo "API PID: $API_PID"
    echo "========================================"

    # --------------------------------------------------
    # Keep container alive
    # --------------------------------------------------

    while kill -0 "$SCHEDULER_PID" 2>/dev/null &&
          kill -0 "$API_PID" 2>/dev/null
    do
        sleep 5
    done

    echo "Airflow process stopped."

    exit 1


# ==================================================
# STREAMLIT
# ==================================================

elif [ "$1" = "streamlit" ]; then

    echo "========================================"
    echo "Starting Streamlit"
    echo "========================================"

    echo "Python version:"
    python --version

    # --------------------------------------------------
    # S3 sync
    # --------------------------------------------------

    echo "Starting S3 sync..."

    mkdir -p /app/saved_models

    aws s3 sync \
        "s3://${BUCKET_NAME}/saved_models" \
        /app/saved_models

    echo "Saved models sync complete."

    # --------------------------------------------------
    # Start Streamlit
    # --------------------------------------------------

    echo "Starting Streamlit application..."

    exec streamlit run app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.enableCORS false


# ==================================================
# UNKNOWN COMMAND
# ==================================================

else

    echo "Unknown service: $1"

    exec "$@"

fi

