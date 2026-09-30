from phishing.exception import PhishingException
from phishing.logger import logging
from phishing.entity import artifact_entity, config_entity
from scipy.stats import ks_2samp
from phishing.config import TARGET_COLUMN
from phishing.utils import convert_columns_to_float, writing_yml_file
import pandas as pd
import numpy as np
import os, sys


class DataValidation:
    """
    Class responsible for validating data by checking for missing columns and data drift.
    """

    def __init__(self, 
                 data_validation_config: config_entity.DataValidationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact
                ):
        """
        Initializes DataValidation class with configuration and ingestion artifacts.
        """
        try:
            logging.info(f"{'>'*20}  Data Validation Started {'<'*20}")
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.validation_error = dict()
        except Exception as e:
            raise PhishingException(e, sys)

    def is_required_columns_exists(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> bool:
        """
        Checks if all required columns from base_df exist in current_df.
        
        Args:
            base_df (pd.DataFrame): The base reference DataFrame.
            current_df (pd.DataFrame): The current DataFrame to validate.
            report_key_name (str): Key name to save missing columns report.

        Returns:
            bool: True if all required columns exist, False otherwise.
        """
        try:
            logging.info(f"Checking if required columns exist in the current dataset.")

            base_columns = base_df.columns
            current_columns = current_df.columns
           
            missing_columns = []

            for column in base_columns:
                if column not in current_columns:
                    missing_columns.append(column)
                    logging.info(f"Required column '{column}' is missing in current DataFrame.")

            if missing_columns:
                self.validation_error[report_key_name] = missing_columns
                logging.warning(f"Missing columns detected: {missing_columns}")
                return False

            logging.info(f"All required columns are available in the DataFrame.")
            return True
            
        except Exception as e:
            raise PhishingException(e, sys)

    def data_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, report_key_name: str) -> None:
        """
        Performs a Kolmogorov-Smirnov test to detect data drift between base and current datasets.
        
        Args:
            base_df (pd.DataFrame): The base reference DataFrame.
            current_df (pd.DataFrame): The current DataFrame to check for drift.
            report_key_name (str): Key name to save data drift report.
        """
        try:
            data_report = dict()
            
            for column in base_df.columns:
                base_data = base_df[column]
                current_data = current_df[column]
                logging.info(f"Performing KS test for column: {column}. Data types: {base_data.dtype} vs {current_data.dtype}")

                same_distribution = ks_2samp(base_data, current_data)

                if same_distribution.pvalue > 0.05:
                    data_report[column] = {
                        "p-values": float(same_distribution.pvalue),
                        "same_distribution": True
                    }
                    logging.info(f"Column {column} - No significant drift detected (p-values: {same_distribution.pvalue}).")
                else:
                    data_report[column] = {
                        "p-values": float(same_distribution.pvalue),
                        "same_distribution": False
                    }
                    logging.info(f"Column {column} - Data drift detected (p-values: {same_distribution.pvalue}).")

            self.validation_error[report_key_name] = data_report
            logging.info(f"Data drift report generated successfully.")

        except Exception as e:
            raise PhishingException(e, sys)

    def initiate_data_validation(self) -> artifact_entity.DataValidationArtifact:
        """
        Initiates the data validation process including missing column checks and data drift detection.
        
        Returns:
            DataValidationArtifact: Contains the path to the validation report YAML file.
        """
        try:
            # Reading base, train, and test datasets
            logging.info("Reading base data as DataFrame.")
            base_df = pd.read_csv(self.data_validation_config.base_data_file_path)

            logging.info("Reading train and test datasets as DataFrames.")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

             # Convert columns to float type for base, train, and test dataframes (excluding target column)
            logging.info("Converting columns to float type for base DataFrame.")
            exclude_columns = [TARGET_COLUMN]
            base_df = convert_columns_to_float(base_df, exclude_columns=exclude_columns)
            
            logging.info("Converting columns to float type for train DataFrame.")
            train_df = convert_columns_to_float(train_df, exclude_columns=exclude_columns)

            logging.info("Converting columns to float type for test DataFrame.")
            test_df = convert_columns_to_float(test_df, exclude_columns=exclude_columns)

            # Check if required columns exist
            logging.info("Validating required columns in train DataFrame.")
            train_df_columns_status = self.is_required_columns_exists(
                base_df=base_df, 
                current_df=train_df,
                report_key_name="missing_columns_within_train_dataset"
            )

            logging.info("Validating required columns in test DataFrame.")
            test_df_columns_status = self.is_required_columns_exists(
                base_df=base_df,
                current_df=test_df,
                report_key_name="missing_columns_within_test_dataset"
            )

            # Perform data drift detection if required columns are present
            if train_df_columns_status:
                logging.info(f"All required columns present in train DataFrame. Detecting data drift for train dataset.")
                self.data_drift(base_df=base_df, current_df=train_df, report_key_name="data_drift_within_train_dataset")
            else:
                logging.warning(f"Missing required columns in train DataFrame. Skipping data drift detection.")

            if test_df_columns_status:
                logging.info(f"All required columns present in test DataFrame. Detecting data drift for test dataset.")
                self.data_drift(base_df=base_df, current_df=test_df, report_key_name="data_drift_within_test_dataset")
            else:
                logging.warning(f"Missing required columns in test DataFrame. Skipping data drift detection.")

            # Write the validation error report to a YAML file
            logging.info(f"Writing validation error report to YAML file.")
            writing_yml_file(
                file_path=self.data_validation_config.report_yaml_file_path,
                data=self.validation_error
            )
            logging.info(f"Validation error report written successfully.")

            # Create and return DataValidationArtifact
            data_validation_artifact = artifact_entity.DataValidationArtifact(
                report_yml_file_path=self.data_validation_config.report_yaml_file_path
            )

            logging.info(f"Data validation artifact created successfully: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise PhishingException(e, sys)
