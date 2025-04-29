#!/bin/sh
set -e

# Log file location
LOG_FILE="/app/logs/startup.log"

# Function to log messages
log_message() {
  echo "$(date) - $1" >> $LOG_FILE
}

# Common S3 sync functionality for both Airflow and Streamlit
start_s3_sync() {
  log_message "Starting S3 sync (if BUCKET_NAME is set)..."
  if [ -n "$BUCKET_NAME" ]; then
    log_message "Syncing saved models from S3..."
    mkdir -p /app/saved_models
    aws s3 sync s3://"$BUCKET_NAME"/saved_models /app/saved_models
    log_message "Saved models sync complete."
  else
    log_message "BUCKET_NAME is not set. Skipping S3 sync."
  fi
}

# Airflow section
if [ "$1" = "airflow" ]; then
  start_s3_sync  # Perform the S3 sync
  
  log_message "Migrating Airflow DB..."
  airflow db upgrade
  log_message "Airflow DB migration completed."

  log_message "Checking if Admin user exists..."
  if ! airflow users list | grep -w "$AIRFLOW_USERNAME" > /dev/null 2>&1; then
    log_message "Creating Admin user..."
    airflow users create \
      --email "$AIRFLOW_EMAIL" \
      --firstname "Admin" \
      --lastname "User" \
      --password "$AIRFLOW_PASSWORD" \
      --role "Admin" \
      --username "$AIRFLOW_USERNAME"
    log_message "Admin user created."
  else
    log_message "Admin user exists."
  fi

  # Start Airflow scheduler in the background
  log_message "Starting Airflow scheduler..."
  nohup airflow scheduler &

  # Start Airflow webserver
  log_message "Starting Airflow webserver..."
  exec airflow webserver

# Streamlit section
elif [ "$1" = "streamlit" ]; then
  start_s3_sync  # Perform the S3 sync
  
  log_message "Starting Streamlit app..."
  exec streamlit run app.py --server.port 8501 --server.address=0.0.0.0 --server.enableCORS false

else
  log_message "Unknown service: $1"
  exec "$@"
fi
