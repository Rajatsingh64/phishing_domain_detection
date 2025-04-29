from phishing.exception import PhishingException
from phishing.logger import logging
import numpy as np
import pandas as pd
from phishing.entity import artifact_entity, config_entity
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from phishing.config import TARGET_COLUMN
from phishing import utils
import os
import sys


class ModelTraining:
    """
    Class responsible for training a Random Forest classifier
    and evaluating its performance using the top N important features.
    """

    def __init__(self, model_training_config: config_entity.ModelTrainingConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        """
        Initializes the ModelTraining class with configuration and artifacts.

        Parameters:
            model_training_config (ModelTrainingConfig): Configuration for model training.
            data_transformation_artifact (DataTransformationArtifact): Data transformation details.
            data_ingestion_artifact (DataIngestionArtifact): Data ingestion details.
        """
        try:
            logging.info(f"{'>' * 20} Started Model Training {'<' * 20}")
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise PhishingException(e, sys)

    def train_rf_model_with_top_features(self, input_df: pd.DataFrame, target_df: pd.Series, top_n: int = 25):
        """
        Trains a Random Forest model on all features, then retrains using the top N important features.

        Parameters:
            input_df (pd.DataFrame): Input feature DataFrame.
            target_df (pd.Series): Target variable Series.
            top_n (int): Number of top features to retain for final training.

        Returns:
            rf_model (RandomForestClassifier): Trained Random Forest model.
            top_feature_names (List[str]): List of top N selected feature names.
        """
        try:
            # Initialize the Random Forest Classifier
            rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=30,
                random_state=42,
                min_samples_split=2,
                max_features='log2',
                min_samples_leaf=1
            )

            logging.info("Training Random Forest model with all features...")
            rf_model.fit(input_df, target_df)

            # Get feature importances and select top N features
            importances = rf_model.feature_importances_
            indices = importances.argsort()[::-1][:top_n]
            top_feature_names = input_df.columns[indices].tolist()

            logging.info(f"Top {top_n} selected features: {top_feature_names}")

            # Retrain the model using only the top N features
            top_features_df = input_df[top_feature_names]
            logging.info(f"Retraining model using top {top_n} features...")
            rf_model.fit(top_features_df, target_df)

            return rf_model, top_feature_names

        except Exception as e:
            raise PhishingException(e, sys)

    def initiate_model_training(self) -> artifact_entity.ModelTrainingArtifact:
        """
        Initiates model training, evaluates performance, saves artifacts, and returns the training artifact.

        Returns:
            ModelTrainingArtifact: Artifact containing model file path, training/testing accuracy, and top features path.
        """
        try:
            # Load datasets
            logging.info("Loading transformed original dataset...")
            main_df = pd.read_csv(self.data_transformation_artifact.transformed_original_data_file_path)

            logging.info("Loading training and testing datasets...")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Split inputs and targets
            logging.info("Splitting input features and target variable for original dataset...")
            main_input_df = main_df.drop(TARGET_COLUMN, axis=1)
            main_target_df = main_df[TARGET_COLUMN]

            logging.info("Splitting input features and target variable for training and testing datasets...")
            train_input_df = train_df.drop(TARGET_COLUMN, axis=1)
            train_target_df = train_df[TARGET_COLUMN]
            test_input_df = test_df.drop(TARGET_COLUMN, axis=1)
            test_target_df = test_df[TARGET_COLUMN]

            # Train model and select top features
            rf_model, top_feature_names = self.train_rf_model_with_top_features(
                input_df=main_input_df,
                target_df=main_target_df,
                top_n=40
            )

            logging.info(f"Top features used for model training: {top_feature_names}")

            # Evaluate model on training data
            logging.info("Evaluating model on training data...")
            yhat_train = rf_model.predict(train_input_df[top_feature_names])
            acc_train_score = accuracy_score(train_target_df, yhat_train)
            logging.info(f"Training Accuracy Score: {acc_train_score}")

            # Evaluate model on testing data
            logging.info("Evaluating model on testing data...")
            yhat_test = rf_model.predict(test_input_df[top_feature_names])
            acc_test_score = accuracy_score(test_target_df, yhat_test)
            logging.info(f"Testing Accuracy Score: {acc_test_score}")

            # Performance validation
            logging.info("Validating model performance thresholds...")
            if acc_test_score < self.model_training_config.accuracy_threshold:
                raise Exception(
                    f"Model accuracy {acc_test_score} is below the expected threshold "
                    f"{self.model_training_config.accuracy_threshold}"
                )

            diff = abs(acc_train_score - acc_test_score)
            logging.info(f"Train-Test Accuracy Difference: {diff}")
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(
                    f"Model is overfitting. Accuracy difference {diff} exceeds allowed threshold "
                    f"{self.model_training_config.overfitting_threshold}"
                )

            # Save model
            logging.info("Saving trained model...")
            utils.save_object(file_path=self.model_training_config.model_file_path, obj=rf_model)
            logging.info(f"Model saved successfully at {self.model_training_config.model_file_path}")

            # Save top features
            logging.info("Saving selected top features...")
            utils.save_features_names(
                top_features=top_feature_names,
                save_path=self.model_training_config.model_feature_names_file_path
            )

            # Ensure plot directory exists
            plot_dir = os.path.dirname(self.model_training_config.top_features_plot_file_path)
            os.makedirs(plot_dir, exist_ok=True)

            # Save ROC-AUC plot
            logging.info("Saving ROC AUC curve for testing data...")
            y_proba = rf_model.predict_proba(test_input_df[top_feature_names])[:, 1]
            utils.plot_and_save_model_evaluation(
                y_true=test_target_df,
                y_pred_proba=y_proba,
                title="ROC-AUC Curve: Testing Data",
                save_path=self.model_training_config.roc_auc_plot_image_path
            )

            # Save feature importance plot
            logging.info("Saving Top 40 Feature Importance Plot...")
            utils.plot_and_save_feature_importances(
                model=rf_model,
                feature_names=main_input_df.columns.tolist(),
                top_n=40,
                save_path=self.model_training_config.top_features_plot_file_path
            )

            # Create and return model training artifact
            model_training_artifact = artifact_entity.ModelTrainingArtifact(
                model_file_path=self.model_training_config.model_file_path,
                train_accuracy_score=acc_train_score,
                test_accuracy_score=acc_test_score,
                model_feature_names_file_path=self.model_training_config.model_feature_names_file_path
            )
            logging.info(f"Model Training Artifact Created: {model_training_artifact}")

            return model_training_artifact

        except Exception as e:
            raise PhishingException(e, sys)
