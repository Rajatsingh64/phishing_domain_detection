
from phishing.logger import logging 
from phishing.exception import PhishingException 
from phishing.components.data_ingestion import DataIngestion
from phishing.components import data_validation , data_transformation , model_training
from phishing.entity.config_entity import DataIngestionConfig , TrainingPipelineConfig , DataValidationConfig , DataTransformationConfig , ModelTrainingConfig
import os ,sys
import pandas as pd

if __name__ == "__main__":
    
    try:
        
        training_pipeline_config=TrainingPipelineConfig()
        
        # Data Ingestion
        data_ingestion_config=DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
        print(f"{'>'*20} Data Ingestion Completed Sucessfully {'<'*20}")
        logging.info(f"{'>'*20} Data Ingestion Completed Sucessfully {'<'*20}")

        # DataValidation
        data_validation_config=DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation=data_validation.DataValidation(
                                                data_validation_config=data_validation_config  ,
                                                data_ingestion_artifact=data_ingestion_artifact
                                            )
        data_validation_artifact=data_validation.initiate_data_validation()
        print(f"{'>'*20} Data Validation Completed Sucessfully {'<'*20}")
        logging.info(f"{'>'*20} Data Validation Completed Sucessfully {'<'*20}")
    
        # Data Transformation
        data_transformation_config=DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation=data_transformation.DataTransformation(data_transformation_config=data_transformation_config , 
                                                                   data_ingestion_artifact=data_ingestion_artifact
                                                                 )
        data_transformation_artifact=data_transformation.initiate_data_transformation()
        print(f"{'>'*20} Data Transformation Completed Sucessfully {'<'*20}")
        logging.info(f"{'>'*20} Data Transformation Completed Sucessfully {'<'*20}")

        #model training 
        model_training_config=ModelTrainingConfig(training_pipeline_config=training_pipeline_config)
        model_training_=model_training.ModelTraining(model_training_config=model_training_config , 
                                                    data_transformation_artifact=data_transformation_artifact,
                                                    data_ingestion_artifact=data_ingestion_artifact)
        model_training_artifact=model_training_.initiate_model_training()
        print(f"{'>'*20} Model Training Completed Sucessfully {'<'*20}")
        logging.info(f"{'>'*20} Model Training Completed Sucessfully {'<'*20}")


    except Exception as e:
       raise PhishingException(e,sys)