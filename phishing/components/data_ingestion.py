#from phishing.config import google_client, table_id
from phishing.utils import get_table_as_dataframe
from phishing.logger import logging
from phishing.exception import PhishingException
from phishing.entity.config_entity import DataIngestionConfig
from sklearn.model_selection import train_test_split  
from phishing.entity.artifact_entity import DataIngestionArtifact
import pandas as pd
import os
import sys

class DataIngestion:
    """
    Class for handling the data ingestion process. This includes loading data from
    external sources (e.g., BigQuery or local files), cleaning it, and splitting it 
    into training and testing datasets.

    Attributes:
        data_ingestion_config (DataIngestionConfig): Configuration parameters for the ingestion process.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with the provided configuration.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for data ingestion process.

        Raises:
            PhishingException: If there is an issue during initialization.
        """
        try:
            logging.info(f"{'>'*20}  Data Ingestion {'<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise PhishingException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process: extracts data, cleans it, splits it,
        and stores the datasets in appropriate locations.

        Returns:
            DataIngestionArtifact: Artifact containing paths to the feature store, training, and testing data.

        Raises:
            PhishingException: If there is an issue during the ingestion process.
        """
        try:
            logging.info(f"Extracting Data from Google BigQuery")
            # If you want to load from BigQuery, uncomment below line
            #df = get_table_as_dataframe(client=google_client, table_id=table_id)
            # For now, loading from a local CSV file for demo purposes
            df = pd.read_csv("dataset/dataset_full.csv")

            # Clean the dataset
            logging.info(f"Dropping Null values from the dataset if available")
            df = df.dropna()

            logging.info(f"Dropping Duplicated Values from dataset if available")
            df = df.drop_duplicates()

            # Store the original dataset in the feature store folder
            logging.info(f"Storing Original dataset in a Data Ingestion feature store folder")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            # Create feature store directory if it doesn't exist
            os.makedirs(feature_store_dir, exist_ok=True)
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            logging.info(f"Original Dataset Stored inside feature store folder")

            # Split the dataset into training and testing sets
            logging.info(f"Splitting training and testing dataset")
            train_df, test_df = train_test_split(df, test_size=self.data_ingestion_config.test_threshold, random_state=42)

            # Store the train and test datasets in the data ingestion datasets folder
            logging.info(f"Creating dataset Directory if not Available")
            logging.info(f"Storing train and test dataset inside a data ingestion dataset folder")
            dataset_dir=os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir , exist_ok=True)
            train_df.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info(f"Train and test datasets successfully stored inside the datasets folder")

            # Create and return the DataIngestionArtifact object
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )

            logging.info(f"Data Ingestion artifact {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise PhishingException(e, sys)
