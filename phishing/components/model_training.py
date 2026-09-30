import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from phishing import utils
from phishing.config import TARGET_COLUMN
from phishing.entity import artifact_entity, config_entity
from phishing.exception import PhishingException
from phishing.logger import logging


class ModelTraining:
    """Train the phishing model and store the training artifacts."""

    def __init__(
        self,
        model_training_config: config_entity.ModelTrainingConfig,
        data_transformation_artifact: artifact_entity.DataTransformationArtifact,
        data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
    ):
        """Store the training configuration and related pipeline artifacts."""
        try:
            logging.info(f"{'>' * 20} Model training started {'<' * 20}")
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as exc:
            raise PhishingException(exc, sys) from exc

    def train_rf_model_with_top_features(
        self,
        feature_matrix: pd.DataFrame,
        target_series: pd.Series,
        top_n: int = 25,
    ):
        """Train the model, rank the features, and retrain on the strongest subset."""
        try:
            random_forest_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=30,
                random_state=42,
                min_samples_split=2,
                max_features="log2",
                min_samples_leaf=1,
            )

            logging.info("Training the Random Forest model with the full feature set.")
            random_forest_model.fit(feature_matrix, target_series)

            importances = random_forest_model.feature_importances_
            top_feature_indexes = importances.argsort()[::-1][:top_n]
            top_feature_names = feature_matrix.columns[top_feature_indexes].tolist()

            logging.info(f"Selected the top {top_n} features: {top_feature_names}")

            selected_feature_matrix = feature_matrix[top_feature_names]
            logging.info(f"Retraining the model on the top {top_n} features.")
            random_forest_model.fit(selected_feature_matrix, target_series)

            return random_forest_model, top_feature_names
        except Exception as exc:
            raise PhishingException(exc, sys) from exc

    def initiate_model_training(self) -> artifact_entity.ModelTrainingArtifact:
        """Load the prepared data, train the model, and save the outputs."""
        try:
            logging.info("Loading the transformed training set.")
            if os.path.exists(self.data_transformation_artifact.transformed_original_data_file_path):
                train_dataframe = pd.read_csv(self.data_transformation_artifact.transformed_original_data_file_path)
                logging.info("Using the transformed training data for model fitting.")
            else:
                train_dataframe = pd.read_csv(self.data_ingestion_artifact.train_file_path)
                logging.info("Using the raw training data because the transformed file is not available.")

            logging.info("Loading the test dataset.")
            test_dataframe = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            logging.info("Separating features from the target for training and evaluation.")
            train_feature_matrix = train_dataframe.drop(TARGET_COLUMN, axis=1)
            train_target_series = train_dataframe[TARGET_COLUMN]
            test_feature_matrix = test_dataframe.drop(TARGET_COLUMN, axis=1)
            test_target_series = test_dataframe[TARGET_COLUMN]

            trained_model, top_feature_names = self.train_rf_model_with_top_features(
                feature_matrix=train_feature_matrix,
                target_series=train_target_series,
                top_n=40,
            )

            logging.info(f"Top features selected for training: {top_feature_names}")

            logging.info("Evaluating the model on the training split.")
            train_predictions = trained_model.predict(train_feature_matrix[top_feature_names])
            train_accuracy = accuracy_score(train_target_series, train_predictions)
            logging.info(f"Training accuracy: {train_accuracy}")

            logging.info("Evaluating the model on the test split.")
            test_predictions = trained_model.predict(test_feature_matrix[top_feature_names])
            test_accuracy = accuracy_score(test_target_series, test_predictions)
            logging.info(f"Testing accuracy: {test_accuracy}")

            logging.info("Checking the model against the configured thresholds.")
            if test_accuracy < self.model_training_config.accuracy_threshold:
                raise Exception(
                    f"Model accuracy {test_accuracy} is below the expected threshold "
                    f"{self.model_training_config.accuracy_threshold}"
                )

            accuracy_gap = abs(train_accuracy - test_accuracy)
            logging.info(f"Train-test accuracy gap: {accuracy_gap}")
            if accuracy_gap > self.model_training_config.overfitting_threshold:
                raise Exception(
                    f"The model is overfitting. Accuracy difference {accuracy_gap} exceeds the permitted threshold "
                    f"{self.model_training_config.overfitting_threshold}"
                )

            logging.info("Saving the trained model to disk.")
            utils.save_object(file_path=self.model_training_config.model_file_path, obj=trained_model)
            logging.info(f"Model saved successfully at {self.model_training_config.model_file_path}")

            logging.info("Saving the selected feature names.")
            utils.save_features_names(
                top_features=top_feature_names,
                save_path=self.model_training_config.model_feature_names_file_path,
            )

            plot_directory = os.path.dirname(self.model_training_config.top_features_plot_file_path)
            os.makedirs(plot_directory, exist_ok=True)

            logging.info("Saving the ROC curve for the test split.")
            test_probabilities = trained_model.predict_proba(test_feature_matrix[top_feature_names])[:, 1]
            utils.plot_and_save_model_evaluation(
                y_true=test_target_series,
                y_pred_proba=test_probabilities,
                title="ROC-AUC Curve: Testing Data",
                save_path=self.model_training_config.roc_auc_plot_image_path,
            )

            logging.info("Saving the top feature importance plot.")
            utils.plot_and_save_feature_importances(
                model=trained_model,
                feature_names=train_feature_matrix.columns.tolist(),
                top_n=40,
                save_path=self.model_training_config.top_features_plot_file_path,
            )

            model_training_artifact = artifact_entity.ModelTrainingArtifact(
                model_file_path=self.model_training_config.model_file_path,
                train_accuracy_score=train_accuracy,
                test_accuracy_score=test_accuracy,
                model_feature_names_file_path=self.model_training_config.model_feature_names_file_path,
            )
            logging.info(f"Model training artifact created: {model_training_artifact}")

            return model_training_artifact
        except Exception as exc:
            raise PhishingException(exc, sys) from exc
