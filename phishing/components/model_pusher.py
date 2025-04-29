from phishing.predictor import ModelResolver
from phishing.entity.config_entity import ModelPusherConfig
from phishing.exception import PhishingException
import os, sys
from phishing.utils import load_object, save_object, load_features_names
from phishing.logger import logging
from phishing.entity.artifact_entity import DataTransformationArtifact, ModelTrainingArtifact, ModelPusherArtifact

class ModelPusher:
    def __init__(self,
                 model_pusher_config: ModelPusherConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainingArtifact):
        """
        Initializes the ModelPusher with configuration and artifact information.
        Also initializes a ModelResolver instance to manage the saved model directory.
        """
        try:
            logging.info(f"{'>>'*20} Model Pusher Initialization {'<<'*20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            # Initialize ModelResolver to manage the directory for saved models
            self.model_resolver = ModelResolver(model_registry=self.model_pusher_config.saved_model_dir)
            logging.info("ModelPusher initialization successful.")
        except Exception as e:
            raise PhishingException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        """
        Initiates the model pushing process:
          1. Loads the transformer, trained model, and top features.
          2. Saves these objects to the model pusher directory.
          3. Saves the same objects to the saved model directory for versioning.
          4. Returns a ModelPusherArtifact containing the relevant directory paths.
        """
        try:
            # ------------------------------------------------------------------------
            # Step 1: Load the latest trained model with latest trained features names
            # ------------------------------------------------------------------------
            logging.info("Loading the model with latest trained features names from artifacts.")
            
            model = load_object(file_path=self.model_trainer_artifact.model_file_path)
            top_feature_file = load_features_names(file_path=self.model_trainer_artifact.model_feature_names_file_path)
           
            logging.info("Successfully loaded the latest trained model and top features.")

            # ---------------------------------------------------------------------------------------
            # Step 2: Save the model and latest trained feature names to the model pusher directory
            # ---------------------------------------------------------------------------------------
            logging.info("Saving the model and latest features names to the model pusher directory.")
           
            save_object(file_path=self.model_pusher_config.pusher_model_path, obj=model)
            save_object(file_path=self.model_pusher_config.pusher_model_features_names_file_path, obj=top_feature_file)

            logging.info("Model and latest features  names saved successfully in the model pusher directory.")

            # --------------------------------------------------------------------
            # Step 3: Save the model and top features to the saved model directory
            # --------------------------------------------------------------------
            logging.info("Saving the model and latest trained features names to the saved model directory.")
            
            model_path = self.model_resolver.get_latest_save_model_path()
            top_feature_path = self.model_resolver.get_latest_save_model_feature_names_file_path()
        
            save_object(file_path=model_path, obj=model)
            save_object(file_path=top_feature_path, obj=top_feature_file)
           
            logging.info("Model and latest features names saved successfully in the saved model directory.")

            # --------------------------------------------------------------------
            # Step 4: Create and return the ModelPusherArtifact
            # --------------------------------------------------------------------
            model_pusher_artifact = ModelPusherArtifact(
                pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                saved_model_dir=self.model_pusher_config.saved_model_dir
            )
            logging.info(f"Model PusherArtifact created successfully: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise PhishingException(e, sys)
