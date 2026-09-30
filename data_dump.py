import os
import json
import pandas as pd
from google.cloud import bigquery
from phishing.config import google_client
from phishing.config import table_id
import warnings
warnings.filterwarnings("ignore")

# Load your data into a Pandas DataFrame
# Replace 'your_file.csv' with the path to your CSV file
df = pd.read_csv("dataset/dataset_full.csv")

# Specify your BigQuery table ID (format: "project_id.dataset_name.table_name")
# Configure the load job; here WRITE_TRUNCATE will replace the table if it exists.
job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    # Optionally, define the schema explicitly:
    # schema=[
    #     bigquery.SchemaField("column1", "STRING"),
    #     bigquery.SchemaField("column2", "INTEGER"),
    #     # ... add more fields as needed
    # ],
)

# Start the load job to upload the DataFrame to BigQuery
load_job = google_client.load_table_from_dataframe(df, table_id, job_config=job_config)

# Wait for the load job to complete
load_job.result()

# Retrieve the destination table to verify the number of rows loaded
destination_table = google_client.get_table(table_id)
# Number of rows loaded
num_rows = destination_table.num_rows
# Total bytes used by the table
num_bytes = destination_table.num_bytes /(1024*1024) # Convert num_bytes (bytes) to MB
# Number of columns in the table (schema is a list of SchemaField objects)
num_columns = len(destination_table.schema)
print(f"{'>'*20} Loaded {num_rows} rows and {num_columns} columns (Table Size: {num_bytes:.2f} mb) into Google Cloud BigQuery {'<'*20}")