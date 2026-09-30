import os
import sys

from sklearn.model_selection import train_test_split

from phishing.config import google_client, table_id
from phishing.entity.artifact_entity import DataIngestionArtifact
from phishing.entity.config_entity import DataIngestionConfig
from phishing.exception import PhishingException
from phishing.logger import logging
from phishing.utils import get_table_as_dataframe


class DataIngestion:
    """Load raw data, clean it, and split it into train and test sets."""

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """Store the configured paths for the ingestion step."""
        try:
            logging.info(f"{'>' * 20} Data ingestion started {'<' * 20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as exc:
            raise PhishingException(exc, sys) from exc

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Extract the dataset, clean it, and save the train/test files."""
        try:
            logging.info("Loading the phishing dataset from Google BigQuery.")
            dataset = get_table_as_dataframe(client=google_client, table_id=table_id)

            logging.info("Removing null values and duplicate records.")
            dataset = dataset.dropna().drop_duplicates()

            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            dataset.to_csv(self.data_ingestion_config.feature_store_file_path, index=False, header=True)
            logging.info("Original dataset saved in the feature store.")

            logging.info("Splitting the dataset into train and test sets.")
            train_dataset, test_dataset = train_test_split(
                dataset,
                test_size=self.data_ingestion_config.test_threshold,
                random_state=42,
            )

            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir, exist_ok=True)
            train_dataset.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_dataset.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)
            logging.info("Training and test datasets saved successfully.")

            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path,
            )

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as exc:
            raise PhishingException(exc, sys) from exc
