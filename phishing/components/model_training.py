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
    Class to handle model training tasks, including training a Random Forest classifier and evaluating its performance.
    """

    def __init__(self, model_training_config: config_entity.ModelTrainingConfig,
                 data_transformation_artifact: artifact_entity.DataTransformationArtifact,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        """
        Initializes the ModelTraining class with configuration, data transformation artifact, and data ingestion artifact.

        Parameters:
            model_training_config (config_entity.ModelTrainingConfig): Configuration for model training.
            data_transformation_artifact (artifact_entity.DataTransformationArtifact): Data transformation details.
            data_ingestion_artifact (artifact_entity.DataIngestionArtifact): Data ingestion details.
        """
        try:
            logging.info(f"{'>' * 20} Model Training Started {'<' * 20}")
            self.model_training_config = model_training_config
            self.data_transformation_artifact = data_transformation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise PhishingException(e, sys)
    
    def train_rf_model_with_top_features(self, input_df: pd.DataFrame, target_df: pd.Series, top_n: int = 25):
        """
        Train a Random Forest model with all features, then refit using the top N important features.

        Parameters:
            input_df (pd.DataFrame): The feature DataFrame for model training.
            target_df (pd.Series): The target variable (Series) for model training.
            top_n (int): The number of top features to use for the final model training.

        Returns:
            rf_model (RandomForestClassifier): The trained Random Forest model using the top N features.
            top_feature_names (List[str]): The list of top N feature names.
        """
        try:
            # Initialize the RandomForestClassifier
            rf_model = RandomForestClassifier(
                n_estimators=150,
                max_depth=30,
                random_state=42,
                min_samples_split=2,
                max_features='log2',
                min_samples_leaf=1
            )

            # Step 1: Train the model with all features
            logging.info(f"Training Model with all features...")
            rf_model.fit(input_df, target_df)

            # Step 2: Extract feature importances and select the top N features
            importances = rf_model.feature_importances_
            indices = importances.argsort()[::-1][:top_n]
            top_feature_names = input_df.columns[indices].tolist()

            # Log the selected features
            logging.info(f"Selected Top {top_n} Features: {top_feature_names}")

            # Select only top N features for retraining
            top_features_df = input_df[top_feature_names]

            # Step 3: Re-train the model using only the top N features
            logging.info(f"Training Model with Top {top_n} Features...")
            rf_model.fit(top_features_df, target_df)

            # Return the trained model and the list of top feature names
            return rf_model, top_feature_names
        
        except Exception as e:
            raise PhishingException(e, sys)


    def initiate_model_training(self) -> artifact_entity.ModelTrainingArtifact:
        """
        Initiates the model training process, evaluating the model on both training and testing data.

        Returns:
            model_trainer_artifact (artifact_entity.ModelTrainingArtifact): The artifact containing the trained model's
            file path, training and testing accuracy scores, and the top features model file path.
        """
        try:
            # Loading original data and transformed data
            logging.info("Loading original main data array from data transformation artifact")
            main_df = pd.read_csv(self.data_transformation_artifact.transformed_original_data_file_path)

            logging.info("Loading training and testing data from data ingestion artifact")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

            # Splitting input and target features from original data
            logging.info("Splitting input and target features from original data")
            main_input_df = main_df.drop(TARGET_COLUMN, axis=1)
            main_target_df = main_df[TARGET_COLUMN]

            # Splitting input and target features for train and test data
            logging.info("Splitting input and target features for train and test data")
            train_input_df = train_df.drop(TARGET_COLUMN, axis=1)
            train_target_df = train_df[TARGET_COLUMN]
            test_input_df = test_df.drop(TARGET_COLUMN, axis=1)
            test_target_df = test_df[TARGET_COLUMN]

            # Train Random Forest model and select top features
            rf_model, top_features_names = self.train_rf_model_with_top_features(input_df=main_input_df,
                                                                          target_df=main_target_df, 
                                                                          top_n=25)
            
            logging.info(f"{top_features_names} are top  features model trained on")
            
            # Calculate Accuracy score of training data
            logging.info("Calculating accuracy score of training data.")
            # Instead of passing the DataFrame directly, use the column names
            yhat_train = rf_model.predict(train_input_df[top_features_names])
            acc_train_score = accuracy_score(y_true=train_target_df, y_pred=yhat_train)
            logging.info(f"Training accuracy score: {acc_train_score}")

            # Calculate accuracy score of testing data
            logging.info("Calculating accuracy score of testing data.")
            yhat_test = rf_model.predict(test_input_df[top_features_names])
            acc_test_score = accuracy_score(y_true=test_target_df, y_pred=yhat_test)
            logging.info(f"Testing accuracy score: {acc_test_score}")

            # Check if the model meets the expected performance criteria
            logging.info("Checking if the model meets the expected performance criteria.")
            if acc_test_score < self.model_training_config.accuarcy_threshold:
                raise Exception(
                    f"Model did not meet the expected accuracy: {self.model_training_config.accuarcy_threshold}. "
                    f"Actual test score: {acc_test_score}"
                )

            # Check for overfitting: difference between train and test accuracy scores
            diff = abs(acc_train_score - acc_test_score)
            logging.info(f"Difference between training and testing accuracy scores: {diff}")
            if diff > self.model_training_config.overfitting_threshold:
                raise Exception(
                    f"Difference between training and testing scores ({diff}) exceeds the overfitting threshold "
                    f"{self.model_training_config.overfitting_threshold}"
                )

            # Save the trained model
            logging.info("Saving the trained model.")
            utils.save_object(file_path=self.model_training_config.model_file_path, obj=rf_model)
            logging.info("Model saved successfully.")

            # Save the top features model as a pickle file
            utils.save_top_features(top_features=top_features_names,
                                    save_path=self.model_training_config.top_features_model_trained_file_path)

            # Create plot folder if it doesn't exist
            logging.info("Creating plot folder inside model training artifact folder if not available.")
            plot_dir = os.path.dirname(self.model_training_config.top_features_plot_file_path)
            os.makedirs(plot_dir, exist_ok=True)

            
            logging.info("Saving ROC AUC plot for testing data inside model training plot folder")
            # Save ROC AUC score plot for testing data
            y_proba = rf_model.predict_proba(test_input_df[top_features_names])[:, 1]
            utils.plot_and_save_model_evaluation(y_true=test_target_df,
                                                  y_pred_proba=y_proba,
                                                  title="Testing Data", 
                                                  save_path=self.model_training_config.roc_auc_plot_image_path)
            logging.info("Saving Top 25 Features Visualization graph to model training plot folder")
            # Save top features plot image
            utils.plot_and_save_feature_importances(model=rf_model,
                                                    feature_names=main_input_df.columns.tolist(),
                                                    top_n=25,
                                                    save_path=self.model_training_config.top_features_plot_file_path)

            # Prepare and return the model training artifact
            model_trainer_artifact = artifact_entity.ModelTrainingArtifact(
                model_file_path=self.model_training_config.model_file_path,
                train_accuracy_score=acc_train_score,
                test_accuracy_score=acc_test_score,
                top_features_model_trained_file_path=self.model_training_config.top_features_model_trained_file_path
            )
            logging.info(f"Model training artifact created successfully: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise PhishingException(e, sys)
