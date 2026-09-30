"""Project configuration for the phishing detector.

This module loads the environment variables, decodes the Google Cloud service
account credentials, and creates the BigQuery client used by the project.
"""

import base64
import json
import os

from dataclasses import dataclass

from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv()


@dataclass
class EnvironmentVariables:
    """Store the environment values used by the pipeline."""

    google_cloud_credentials_json: str = os.getenv("GOOGLE_CREDENTIALS_B64", "")
    table_id: str = os.getenv("Table_ID", "")


env = EnvironmentVariables()

if not env.google_cloud_credentials_json:
    raise ValueError("GOOGLE_CREDENTIALS_B64 is not set.")
if not env.table_id:
    raise ValueError("Table_ID is not set.")

credentials_payload = base64.b64decode(env.google_cloud_credentials_json).decode("utf-8")
credentials_info = json.loads(credentials_payload)
credentials = service_account.Credentials.from_service_account_info(credentials_info)

project_id = credentials.project_id
table_id = env.table_id

google_client = bigquery.Client(credentials=credentials, project=project_id)

TARGET_COLUMN = "phishing"