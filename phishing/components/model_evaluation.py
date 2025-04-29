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
    """
    Handles evaluation of the newly trained model against the previously saved model.
    """

    def __init__(self,
                 model_eval_config: config_entity.ModelEvaluationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 model_trainer_artifact: artifact_entity.ModelTrainingArtifact):
        """
        Initializes ModelEvaluation with config and artifacts.
        """
        try:
            logging.info(f"{'>>' * 20} Model Evaluation {'<<' * 20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
            logging.info("ModelEvaluation initialization successful.")
        except Exception as e:
            raise PhishingException(e, sys)

    def initiate_model_evaluation(self) -> artifact_entity.ModelEvaluationArtifact:
        """
        Compares current model with previously saved model and returns evaluation artifact.
        """
        try:
            logging.info("Starting model comparison.")

            # Step 1: Check if a previously saved model exists
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path is None:
                # No previous model found, accept the current model
                model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                    is_model_accepted=True,
                    improved_accuracy=None
                )
                logging.info(f"Model evaluation artifact: {model_eval_artifact}")
                return model_eval_artifact

            # Step 2: Load previous model and top features
            logging.info("Loading previous model and top features.")
            previous_model = load_object(self.model_resolver.get_latest_model_path())
            previous_top_features = utils.load_features_names(
                self.model_resolver.get_latest_model_feature_names_file_path()
            )

            # Step 3: Load current model and top features
            logging.info("Loading current trained model and top features.")
            current_model = load_object(self.model_trainer_artifact.model_file_path)
            current_top_features = utils.load_features_names(
                self.model_trainer_artifact.model_feature_names_file_path
            )

            # Step 4: Prepare test data
            logging.info("Preparing test data.")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            input_df = test_df.drop(TARGET_COLUMN, axis=1)
            target_df = test_df[TARGET_COLUMN]

            # Step 5: Evaluate previous model
            logging.info("Evaluating previous model on test data.")
            y_pred_prev = previous_model.predict(input_df[previous_top_features])
            prev_model_score = accuracy_score(target_df, y_pred_prev)
            logging.info(f"Previous model accuracy: {prev_model_score}")

            # Step 6: Evaluate current model
            logging.info("Evaluating current model on test data.")
            y_pred_current = current_model.predict(input_df[current_top_features])
            curr_model_score = accuracy_score(target_df, y_pred_current)
            logging.info(f"Current model accuracy: {curr_model_score}")

            # Step 7: Compare model scores
            if curr_model_score <= prev_model_score:
                logging.info("Current model is worse than previous model.")
                raise Exception("Current trained model is not better than the previous model.")

            improved_accuracy = curr_model_score - prev_model_score
            logging.info(f"Improved accuracy: {improved_accuracy}")

            # Step 8: Prepare and return artifact
            model_eval_artifact = artifact_entity.ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=improved_accuracy
            )
            logging.info(f"Model evaluation artifact: {model_eval_artifact}")
            return model_eval_artifact

        except Exception as e:
            raise PhishingException(e, sys)
