from phishing.exception import PhishingException
from phishing.logger import logging
from phishing.entity import artifact_entity, config_entity
from phishing.predictor import ModelResolver
from phishing.config import TARGET_COLUMN
from phishing.utils import load_object
from phishing import utils

import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score


class ModelEvaluation:
    """Compare the current model against the latest saved model before accepting it."""

    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainingArtifact):
        """Store the evaluation settings and linked artifacts."""
        try:
            logging.info(f"{'>>' * 20} Model Evaluation {'<<' * 20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
            logging.info("Model evaluation initialization successful.")
        except Exception as e:
            raise PhishingException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        """Check whether the current model is better than the previously saved one."""
        try:
            logging.info("Starting the model comparison.")

            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=None
                )
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            logging.info("Loading the previous model and feature list.")
            previous_model = load_object(self.model_resolver.get_latest_model_path())
            previous_top_features = utils.load_features_names(
                self.model_resolver.get_latest_model_feature_names_file_path()
            )

            logging.info("Loading the current trained model and feature list.")
            current_model = load_object(self.model_trainer_artifact.model_file_path)
            current_top_features = utils.load_features_names(
                self.model_trainer_artifact.model_feature_names_file_path
            )

            logging.info("Preparing the test data for comparison.")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            input_df = test_df.drop(TARGET_COLUMN, axis=1)
            target_df = test_df[TARGET_COLUMN]

            logging.info("Evaluating the previous model on the test set.")
            y_pred_prev = previous_model.predict(input_df[previous_top_features])
            prev_model_score = accuracy_score(target_df, y_pred_prev)
            logging.info(f"Previous model accuracy: {prev_model_score}")

            logging.info("Evaluating the current model on the test set.")
            y_pred_current = current_model.predict(input_df[current_top_features])
            curr_model_score = accuracy_score(target_df, y_pred_current)
            logging.info(f"Current model accuracy: {curr_model_score}")

            if curr_model_score <= prev_model_score:
                logging.info("The current model is not better than the previous model.")
                raise Exception("Current trained model is not better than the previous model.")

            improved_accuracy = curr_model_score - prev_model_score
            logging.info(f"Improved accuracy: {improved_accuracy}")

            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=improved_accuracy
            )
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise PhishingException(e, sys)
