import pandas as pd
import numpy as np
from datetime import datetime
import os ,sys
import warnings
warnings.filterwarnings("ignore")

def get_table_as_dataframe(client, table_id):
    """
    =========================================================================================
    This function extracts data from a Google Cloud (BigQuery) database and returns it as a DataFrame.
    
    client   : Initialized BigQuery client with the proper credentials and project ID.
    table_id : Fully qualified BigQuery table ID in the format "project_id.dataset_name.table_name".

    return   : Table data as a DataFrame.
    =========================================================================================
    """
    try:
        # Use backticks to enclose the table_id in the query string
        query = f"SELECT * FROM `{table_id}`"
        data = client.query(query).to_dataframe()
        return data
    except Exception as e:
        raise Exception(e)
    

