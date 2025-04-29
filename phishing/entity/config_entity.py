"""
Configuration classes for the MLOps Training Pipeline.
Handles all paths and parameters needed for data ingestion, transformation, training, and deployment.
"""

from datetime import datetime
from phishing.logger import logging
from phishing.exception import PhishingException
import os
import sys

class TrainingPipelineConfig:
    """
    Sets up the main artifact directory for the MLOps training pipeline.
    """

    def __init__(self):
        """
        Initialize artifact directory with a timestamp to separate each run.
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
    Configuration for managing paths and parameters related to data ingestion.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize data ingestion config with paths for feature store, train and test datasets.
        """
        try:
            self.data_ingestion_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                "data_ingestion"
            )
            self.feature_store_file_path = os.path.join(
                self.data_ingestion_dir,
                "feature_store",
                "main_dataset.csv"
            )
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
            self.test_threshold = 0.2  # 20% test size
        except Exception as e:
            raise PhishingException(e, sys)

class DataValidationConfig:
    """
    Configuration for managing data validation steps and settings.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize data validation config with paths for validation reports.
        """
        try:
            self.data_validation_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                "data_validation"
            )
            self.report_yaml_file_path = os.path.join(
                self.data_validation_dir,
                "report.yml"
            )
            self.missing_columns_threshold = 0.2  # Allowable missing column threshold
            self.base_data_file_path = os.path.join(
                "base_dataset.csv"
            )
        except Exception as e:
            raise PhishingException(e, sys)

class DataTransformationConfig:
    """
    Configuration for managing data transformation settings.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize data transformation config with paths for transformed datasets.
        """
        try:
            self.data_transformation_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                "data_transformation"
            )
            self.transformed_data_dir = os.path.join(
                self.data_transformation_dir,
                "transformed"
            )
            self.transformed_original_data_file_path = os.path.join(
                self.transformed_data_dir,
                "main.csv"
            )
            self.correlation_threshold = 0.9  # Threshold for feature correlation filtering
        except Exception as e:
            raise PhishingException(e, sys)

class ModelTrainingConfig:
    """
    Configuration for model training parameters and output paths.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize model training config with paths for model files and plots.
        """
        try:
            self.model_training_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                "model_training"
            )
            self.model_file_path = os.path.join(
                self.model_training_dir,
                "model",
                "model.pkl"
            )
            self.accuracy_threshold = 0.9  # Minimum acceptable model accuracy
            self.overfitting_threshold = 0.05  # Acceptable overfitting gap
            self.roc_auc_plot_image_path = os.path.join(
                self.model_training_dir,
                "plots",
                "roc_auc_plot.jpg"
            )
            self.model_feature_names_file_path = os.path.join(
                self.model_training_dir,
                "model_feature_names.pkl"
            )
            self.top_features_plot_file_path = os.path.join(
                self.model_training_dir,
                "plots",
                "top_feature_plot.jpg"
            )
        except Exception as e:
            raise PhishingException(e, sys)

class ModelEvaluationConfig:
    """
    Configuration for evaluating model performance after training.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize model evaluation config with thresholds for performance change.
        """
        try:
            self.model_accuracy_change_threshold = 0.02  # Minimum accuracy change needed to accept a new model
        except Exception as e:
            raise PhishingException(e, sys)

class ModelPusherConfig:
    """
    Configuration for saving and pushing the final model for deployment.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize model pusher config with paths for saving models.
        """
        try:
            self.model_pusher_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                "model_pusher"
            )
            
            self.saved_model_dir = os.path.join(
                "saved_models"
            )
            
            self.pusher_model_dir = os.path.join(
                self.model_pusher_dir,
                "saved_models"
            )
            
            self.pusher_model_path = os.path.join(
                self.pusher_model_dir,
                "model.pkl"
            )
            self.pusher_model_features_names_file_path = os.path.join(
                self.pusher_model_dir,
                "model_feature_names.pkl"
            )
        except Exception as e:
            raise PhishingException(e, sys)
