# Use Python 3.8 base image
FROM python:3.8-slim

# Set environment variables early
ENV AIRFLOW_HOME="/app/airflow"
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
ENV AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_POOL_SIZE=50
ENV AIRFLOW__DATABASE__SQL_ALCHEMY_MAX_OVERFLOW=50

# Set working directory
WORKDIR /app

# Copy everything into /app
COPY . /app/

# Install system dependencies & PostgreSQL driver, AWS CLI, Supervisor
RUN apt update -y && \
    apt install -y gcc libpq-dev awscli supervisor && \
    # Clean up after installing dependencies to reduce image size
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    # Upgrade pip and install psycopg2
    pip3 install --upgrade pip && \
    # Install Python dependencies
    pip3 install --no-cache-dir -r requirements.txt

# Create logs directory for Airflow
RUN mkdir -p /app/airflow/logs

# Make start script executable
RUN chmod +x start.sh

# Expose ports for Airflow and Streamlit
EXPOSE 8080 8501

# Set the entrypoint to the startup script
ENTRYPOINT ["/app/start.sh"]
