from dotenv import load_dotenv
from dataclasses import dataclass
from google.cloud import bigquery
from google.oauth2 import service_account
import pickle
import os,sys
import json 

print(f'\nloading .env file ......')
load_dotenv()  #loading sensitive information from .env

#Creating class for environments Variables, in future i can hide my sensitive information like (eg.passwords ,url)
@dataclass
class EnvironmentVariables:
    google_clouds_credentials = os.getenv("GOOGLE_CREDENTIALS_PATH") 
    table_id:str=os.getenv("Table_ID")


env=EnvironmentVariables()

credentials_path=env.google_clouds_credentials
# Create a Credentials object from the service account file
credentials = service_account.Credentials.from_service_account_file(credentials_path)
# Extract the project_id from the credentials (or set it manually)
project_id = credentials.project_id
table_id=env.table_id

# Initialize the BigQuery client with the credentials and project ID
google_client = bigquery.Client(credentials=credentials, project=project_id)
print("Connected to BigQuery using explicit credentials file.")

# open("important_features_names.pkl" , "rb") as file:
    #important_features=pickle.load(file)


