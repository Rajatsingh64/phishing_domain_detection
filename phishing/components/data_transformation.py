import os
import sys

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from phishing import utils
from phishing.config import TARGET_COLUMN
from phishing.entity import artifact_entity, config_entity
from phishing.exception import PhishingException
from phishing.logger import logging


class DataTransformation:
    """Prepare the training dataset by balancing and removing redundant features."""

    def __init__(
        self,
        data_transformation_config: config_entity.DataTransformationConfig,
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
    ):
        """Store the transformation settings and the ingestion artifact."""
        try:
            logging.info(f"{'>' * 20} Data transformation started {'<' * 20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as exc:
            raise PhishingException(exc, sys) from exc

    def remove_high_correlation_features(self, dataframe: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """Remove features whose pairwise correlation is above the threshold."""
        try:
            logging.info("Checking the feature set for highly correlated columns.")

            correlation_matrix = dataframe.corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )

            columns_to_drop = [
                column_name for column_name in upper_triangle.columns
                if any(upper_triangle[column_name] > threshold)
            ]
            logging.info(f"Removing highly correlated columns: {columns_to_drop}")

            return dataframe.drop(columns=columns_to_drop)
        except Exception as exc:
            raise PhishingException(exc, sys) from exc

    def apply_smote_to_training_data(self, train_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Apply SMOTE on the training split only, to avoid data leakage."""
        try:
            logging.info("Balancing the training set with SMOTE.")

            if TARGET_COLUMN not in train_dataframe.columns:
                raise ValueError(f"Target column '{TARGET_COLUMN}' not found in the training data.")

            feature_matrix = train_dataframe.drop(TARGET_COLUMN, axis=1)
            target_series = train_dataframe[TARGET_COLUMN]

            if target_series.nunique() < 2:
                logging.info("Skipping SMOTE because the target has fewer than two classes.")
                return train_dataframe.copy()

            smote = SMOTE(random_state=42)
            balanced_features, balanced_target = smote.fit_resample(feature_matrix, target_series)
            balanced_training_data = pd.concat([balanced_features, balanced_target], axis=1)

            logging.info(
                "Training data after SMOTE: "
                f"feature shape {balanced_features.shape}, target shape {balanced_target.shape}"
            )
            return balanced_training_data
        except Exception as exc:
            raise PhishingException(exc, sys) from exc

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        """Balance the training data, drop redundant features, and save the transformed set."""
        try:
            logging.info("Reading the train and test datasets for transformation.")
            train_dataframe = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_dataframe = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Applying SMOTE to the training split.")
            transformed_train_data = self.apply_smote_to_training_data(train_dataframe)

            logging.info("Removing highly correlated features from the transformed training set.")
            training_feature_matrix = transformed_train_data.drop(TARGET_COLUMN, axis=1)
            filtered_feature_matrix = self.remove_high_correlation_features(
                dataframe=training_feature_matrix,
                threshold=self.data_transformation_config.correlation_threshold,
            )
            transformed_train_data = pd.concat(
                [filtered_feature_matrix, transformed_train_data[TARGET_COLUMN]],
                axis=1,
            )

            logging.info("Saving the transformed training dataset.")
            transformed_dir = os.path.dirname(self.data_transformation_config.transformed_original_data_file_path)
            os.makedirs(transformed_dir, exist_ok=True)
            transformed_train_data.to_csv(
                self.data_transformation_config.transformed_original_data_file_path,
                index=False,
                header=True,
            )

            _ = test_dataframe
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformed_original_data_file_path=self.data_transformation_config.transformed_original_data_file_path,
            )

            logging.info(f"Data transformation artifact created: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as exc:
            raise PhishingException(exc, sys) from exc
