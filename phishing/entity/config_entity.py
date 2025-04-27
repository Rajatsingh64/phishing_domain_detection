"""
Configuration classes for MLOps Training Pipeline.
Handles paths and parameters required for data ingestion and artifact management.
"""

from datetime import datetime
from phishing.logger import logging
from phishing.exception import PhishingException
import os
import sys

class TrainingPipelineConfig:
    """
    Configuration class for setting up the MLOps training pipeline.
    It creates an artifact directory to store all outputs generated during the pipeline.
    """

    def __init__(self):
        """
        Initialize the artifact directory with a timestamp to avoid overwriting previous runs.
        """
        try:
            logging.info(f"{'>'*20} MLOps Training Pipeline Initialization {'<'*20}")
            self.artifact_dir = os.path.join(
                os.getcwd(), 
                "artifact", 
                f"{datetime.now().strftime('%y%m%d__%H%M%S')}"
            )
        except Exception as e:
            raise PhishingException(e, sys)

class DataIngestionConfig:
    """
    Configuration class for managing paths and parameters related to data ingestion.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize the data ingestion configuration using the artifact directory from TrainingPipelineConfig.

        Args:
            training_pipeline_config (TrainingPipelineConfig): Instance containing artifact directory path.
        """
        try:
            # Create base directory for data ingestion artifacts
            self.data_ingestion_dir = os.path.join(
                training_pipeline_config.artifact_dir, 
                "data_ingestion"
            )

            # Path to store the complete feature store dataset
            self.feature_store_file_path = os.path.join(
                self.data_ingestion_dir, 
                "feature store", 
                "main_dataset.csv"
            )

            # Paths to store the train and test datasets after split
            self.train_file_path = os.path.join(
                self.data_ingestion_dir, 
                "datasets", 
                "train.csv"
            )

            self.test_file_path = os.path.join(
                self.data_ingestion_dir, 
                "datasets", 
                "test.csv"
            )

            # Proportion of the dataset to be used as test data
            self.test_threshold = 0.2

        except Exception as e:
            raise PhishingException(e, sys)


class DataValidationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        try:
            
            self.data_validaion_dir=os.path.join(
                training_pipeline_config.artifact_dir ,
                "data_validation"
            )
        
            self.report_yaml_file_path=os.path.join(
                self.data_validaion_dir , "report.yml"

            )

            self.missing_columns_threshold=0.2

            self.base_data_file_path=os.path.join(
                'dataset/dataset_full.csv'
            )

        except Exception as e:
            raise PhishingException(e, sys)
        

class DataTransformationConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        try:
            self.data_transformation_dir=os.path.join(training_pipeline_config.artifact_dir , "data_transformation")
            self.transformed_data_dir = os.path.join(self.data_transformation_dir, "transformed")
            self.transformed_original_data_file_path=os.path.join(self.transformed_data_dir , "main.csv")
            self.correlation_threshold=0.8

        except Exception as e:
            raise PhishingException(e, sys)
        

class ModelTrainingConfig:

    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        try:
            self.model_training_dir=os.path.join(training_pipeline_config.artifact_dir , "model training")
            self.model_file_path=os.path.join(self.model_training_dir , "model" , "model.pkl")
            self.accuarcy_threshold=0.9
            self.overfitting_threshold=0.05
            self.roc_auc_plot_image_path=os.path.join(self.model_training_dir , "plots" , "roc_auc_plot.jpg")
            self.top_features_model_trained_file_path=os.path.join(self.model_training_dir , "top_features.pkl")
            self.top_features_plot_file_path=os.path.join(self.model_training_dir , "plots" , "Top_Feature_plot.jpg")
        
        except Exception as e:
            raise PhishingException(e, sys)