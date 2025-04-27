"""
This script handles the loading of environment variables,
decoding Base64-encoded Google Cloud service account credentials,
and establishes a BigQuery client connection using explicit credentials.

"""

from dotenv import load_dotenv
from dataclasses import dataclass
from google.cloud import bigquery
from google.oauth2 import service_account
import pickle
import os
import sys
import base64
import json

# Load environment variables from the .env file
print('\nLoading .env file...')
load_dotenv()

@dataclass
class EnvironmentVariables:
    """
    Class to manage and access environment variables.
    This structure allows easy retrieval and future enhancement
    for sensitive information like passwords and URLs.
    """
    google_clouds_credentials_json: str = os.getenv("GOOGLE_CREDENTIALS_B64")
    table_id: str = os.getenv("Table_ID")
  

# Initialize environment variables
env = EnvironmentVariables()

# Decode the Base64-encoded credentials string
credentials_str = base64.b64decode(env.google_clouds_credentials_json).decode('utf-8')

# Load the decoded string into a dictionary
credentials_info = json.loads(credentials_str)

# Create service account credentials from the dictionary
credentials = service_account.Credentials.from_service_account_info(credentials_info)

project_id=credentials.project_id
# Extract table_id for usage
table_id = env.table_id

# Initialize the BigQuery client using the provided credentials and project ID
google_client = bigquery.Client(credentials=credentials, project=project_id)

print(" Successfully connected to BigQuery using explicit credentials. ")

TARGET_COLUMN="phishing"