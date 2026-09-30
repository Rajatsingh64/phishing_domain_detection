#!/bin/sh
set -e

: "${AIRFLOW_EMAIL:=admin@example.com}"
: "${AIRFLOW_USERNAME:=admin}"
: "${AIRFLOW_PASSWORD:=admin}"
: "${AWS_REGION:=us-east-1}"

python - <<'PY'
import importlib.util
import sys
spec = importlib.util.find_spec("psycopg2")
if spec is None:
    raise SystemExit("psycopg2 is not installed. Rebuild the Docker image after installing PostgreSQL dependencies.")
print("psycopg2-ok")
PY

# Common S3 sync functionality for both Airflow and Streamlit
start_s3_sync() {
  echo "Starting S3 sync (if BUCKET_NAME is set)..."
  if [ -n "$BUCKET_NAME" ]; then
    mkdir -p /app/saved_models
    aws s3 sync s3://"$BUCKET_NAME"/saved_models /app/saved_models
    echo "Saved models sync complete."
  else
    echo "BUCKET_NAME is not set. Skipping S3 sync."
  fi
}

# Airflow section
if [ "$1" = "airflow" ]; then
  start_s3_sync  # Perform the S3 sync

  echo "Migrating Airflow DB..."
  if airflow db migrate >/dev/null 2>&1; then
    echo "Airflow DB migration completed."
  else
    echo "Airflow db migrate not available; trying legacy upgrade path..."
    airflow db upgrade
    echo "Legacy Airflow DB upgrade completed."
  fi

  echo "Checking if Admin user exists..."
  if ! airflow users list | grep -w "$AIRFLOW_USERNAME" > /dev/null 2>&1; then
    echo "Creating Admin user..."
    airflow users create \
      --email "$AIRFLOW_EMAIL" \
      --firstname "Admin" \
      --lastname "User" \
      --password "$AIRFLOW_PASSWORD" \
      --role "Admin" \
      --username "$AIRFLOW_USERNAME"
  else
    echo "Admin user exists."
  fi

  # Start Airflow scheduler in the background
  nohup airflow scheduler &

  # Start Airflow webserver
  airflow webserver

# Streamlit section
elif [ "$1" = "streamlit" ]; then
  start_s3_sync  # Perform the S3 sync

  echo "Starting Streamlit app..."
  exec streamlit run app.py --server.port 8501 --server.address=0.0.0.0 --server.enableCORS false

else
  echo "Unknown service: $1"
  exec "$@"
fi
