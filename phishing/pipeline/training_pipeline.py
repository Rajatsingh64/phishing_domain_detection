from phishing.logger import logging
from phishing.exception import PhishingException
from phishing.components.data_ingestion import DataIngestion
from phishing.components import data_validation, data_transformation, model_training, model_evaluation, model_pusher
from phishing.entity.config_entity import (DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, 
                                            DataTransformationConfig, ModelTrainingConfig, ModelEvaluationConfig, 
                                            ModelPusherConfig)
import os, sys
import pandas as pd

def initiate_training_pipeline():
    """
    Initiates the entire training pipeline including data ingestion, validation, transformation, model training,
    evaluation, and model pushing (saving).
    """
    try:
        # Initialize Training Pipeline Configuration
        training_pipeline_config = TrainingPipelineConfig()
        
        # -----------------------------------
        # Step 1: Data Ingestion
        # -----------------------------------
        logging.info(f"{'>'*20} Starting Data Ingestion {'<'*20}")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"{'>'*20} Data Ingestion Completed Successfully {'<'*20}")
        
        # -----------------------------------
        # Step 2: Data Validation
        # -----------------------------------
        logging.info(f"{'>'*20} Starting Data Validation {'<'*20}")
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation_instance = data_validation.DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_validation_artifact = data_validation_instance.initiate_data_validation()
        logging.info(f"{'>'*20} Data Validation Completed Successfully {'<'*20}")

        # -----------------------------------
        # Step 3: Data Transformation
        # -----------------------------------
        logging.info(f"{'>'*20} Starting Data Transformation {'<'*20}")
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation_instance = data_transformation.DataTransformation(
            data_transformation_config=data_transformation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_transformation_artifact = data_transformation_instance.initiate_data_transformation()
        logging.info(f"{'>'*20} Data Transformation Completed Successfully {'<'*20}")

        # -----------------------------------
        # Step 4: Model Training
        # -----------------------------------
        logging.info(f"{'>'*20} Starting  Model Training {'<'*20}")
        model_training_config = ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_training_instance = model_training.ModelTraining(
            model_training_config=model_training_config,
            data_transformation_artifact=data_transformation_artifact,
            data_ingestion_artifact=data_ingestion_artifact
        )
        model_training_artifact = model_training_instance.initiate_model_training()
        logging.info(f"{'>'*20} Model Training Completed Successfully {'<'*20}")

        # -----------------------------------
        # Step 5: Model Evaluation
        # -----------------------------------
        logging.info(f"{'>'*20} Starting Model Evaluation {'<'*20}")
        model_evaluation_config = ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation_instance = model_evaluation.ModelEvaluation(
            model_eval_config=model_evaluation_config,
            data_ingestion_artifact=data_ingestion_artifact,
            data_transformation_artifact=data_transformation_artifact,
            model_trainer_artifact=model_training_artifact
        )
        model_evaluation_artifact = model_evaluation_instance.initiate_model_evaluation()
        logging.info(f"{'>'*20} Model Evaluation Completed Successfully {'<'*20}")

        # -----------------------------------
        # Step 6: Model Pushing (Saving Model)
        # -----------------------------------
        logging.info(f"{'>'*20} Starting Model Pushing {'<'*20}")
        model_pusher_config = ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher_instance = model_pusher.ModelPusher(
            data_transformation_artifact=data_transformation_artifact,
            model_pusher_config=model_pusher_config,
            model_trainer_artifact=model_training_artifact
        )
        model_pusher_artifact = model_pusher_instance.initiate_model_pusher()
        logging.info(f"{'>'*10} Model Pusher Completed {'<'*10}")

        # -----------------------------------
        # Final: Training Pipeline Completed
        # -----------------------------------
        logging.info(f"{'>'*10} Training Pipeline Completed Successfully! {'<'*10}")

    except Exception as e:
        raise PhishingException(e, sys)
