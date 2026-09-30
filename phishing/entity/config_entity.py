"""Configuration classes for the training pipeline.

These classes define the artifact paths and thresholds used throughout the ML workflow.
"""

from datetime import datetime
from phishing.logger import logging
from phishing.exception import PhishingException
import os
import sys


class TrainingPipelineConfig:
    """Create the artifact directory for each pipeline run."""

    def __init__(self):
        """Initialize the artifact directory with a timestamp for this run."""
        try:
            logging.info(f"{'>' * 20} MLOps training pipeline initialization {'<' * 20}")
            self.artifact_dir = os.path.join(
                os.getcwd(),
                "artifacts",
                f"{datetime.now().strftime('%y%m%d__%H%M%S')}"
            )
        except Exception as e:
            raise PhishingException(e, sys)


class DataIngestionConfig:
    """Store the file paths and split settings for the ingestion step."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """Set up the feature-store and train/test dataset paths."""
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
            self.test_threshold = 0.2
        except Exception as e:
            raise PhishingException(e, sys)


class DataValidationConfig:
    """Store the validation settings and files used by the validation step."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """Set the validation report path and threshold values."""
        try:
            self.data_validation_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                "data_validation"
            )
            self.report_yaml_file_path = os.path.join(
                self.data_validation_dir,
                "report.yml"
            )
            self.missing_columns_threshold = 0.2
            self.base_data_file_path = os.path.join("dataset/dataset_full.csv")
        except Exception as e:
            raise PhishingException(e, sys)


class DataTransformationConfig:
    """Store the transformation configuration and output paths."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """Set the output directory and correlation threshold for feature filtering."""
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
            self.correlation_threshold = 0.9
        except Exception as e:
            raise PhishingException(e, sys)


class ModelTrainingConfig:
    """Store the model training parameters and output file paths."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """Set the directories and thresholds for model training artifacts."""
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
            self.accuracy_threshold = 0.9
            self.overfitting_threshold = 0.05
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
    """Store the evaluation threshold used to accept a new model."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """Set the minimum improvement needed to accept a new model."""
        try:
            self.model_accuracy_change_threshold = 0.02
        except Exception as e:
            raise PhishingException(e, sys)


class ModelPusherConfig:
    """Store the model deployment target paths."""

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """Set the saved-model directories used during deployment."""
        try:
            self.model_pusher_dir = os.path.join(
                training_pipeline_config.artifact_dir,
                "model_pusher"
            )

            self.saved_model_dir = os.path.join("saved_models")

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
