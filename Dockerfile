FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRFLOW_HOME=/app/airflow \
    AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True \
    AIRFLOW__CORE__DAGBAG_IMPORT_TIMEOUT=1000 \
    AIRFLOW__DATABASE__SQL_ALCHEMY_POOL_SIZE=50 \
    AIRFLOW__DATABASE__SQL_ALCHEMY_MAX_OVERFLOW=50

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libpq-dev \
        curl \
        awscli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . /app/

RUN mkdir -p /app/airflow/logs /app/saved_models && \
    chmod +x /app/start.sh

EXPOSE 8080 8501

ENTRYPOINT ["/app/start.sh"]
