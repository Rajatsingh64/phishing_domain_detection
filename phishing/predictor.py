import os
from glob import glob
from typing import Optional

class ModelResolver:
    """
    Class to resolve and manage model file paths based on versioning in the model registry.
    """
    def __init__(self,
                 model_registry: str = "saved_models",
                 model_dir_name: str = "model",
                 top_feature_dir_name: str = "features_names"):
        """
        Initializes the ModelResolver with the specified registry and directory names.
        """
        self.model_registry = model_registry
        self.model_dir_name = model_dir_name
        self.top_feature_dir_name = top_feature_dir_name

    def get_latest_dir_path(self) -> Optional[str]:
        """
        Gets the latest version directory (highest numeric folder) from the model registry.
        """
        try:
            dir_names = os.listdir(self.model_registry)
            if len(dir_names) == 0:
                return None
            dir_names = list(map(int, dir_names))
            latest_dir_name = max(dir_names)
            return os.path.join(self.model_registry, str(latest_dir_name))
        except Exception as e:
            raise Exception(f"Error in getting latest directory path: {e}")

    def get_latest_model_path(self) -> str:
        """
        Returns the file path for the latest model.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Model is not available.")
            return os.path.join(latest_dir, self.model_dir_name, "model.pkl")
        except Exception as e:
            raise Exception(f"Error in getting latest model path: {e}")

    def get_latest_model_feature_names_file_path(self) -> str:
        """
        Returns the file path for the top features of the latest model.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception("Top features are not available.")
            return os.path.join(latest_dir, self.top_feature_dir_name, "model_feature_names.pkl")
        except Exception as e:
            raise Exception(f"Error in getting latest top features file path: {e}")

    def get_latest_save_dir_path(self) -> str:
        """
        Determines the next directory path for saving a new model version.
        """
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                return os.path.join(self.model_registry, "0")
            latest_dir_num = int(os.path.basename(latest_dir))
            return os.path.join(self.model_registry, str(latest_dir_num + 1))
        except Exception as e:
            raise Exception(f"Error in getting next save directory path: {e}")

    def get_latest_save_model_path(self) -> str:
        """
        Returns the file path to save a new model.
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, 'model.pkl')
        except Exception as e:
            raise Exception(f"Error in getting path to save the new model: {e}")

    def get_latest_save_model_feature_names_file_path(self) -> str:
        """
        Returns the file path to save the top features of the new model.
        """
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.top_feature_dir_name, 'model_feature_names.pkl')
        except Exception as e:
            raise Exception(f"Error in getting path to save top features: {e}")


class Predictor:
    """
    Class to make predictions using a trained model.
    """
    def __init__(self, model_resolver: ModelResolver):
        """
        Initializes the Predictor with the ModelResolver instance.
        """
        self.model_resolver = model_resolver
