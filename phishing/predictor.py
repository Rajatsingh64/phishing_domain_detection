import os
from typing import Optional


class ModelResolver:
    """Resolve the latest saved model and feature-name files in the registry."""

    def __init__(
        self,
        model_registry: str = "saved_models",
        model_dir_name: str = "model",
        top_feature_dir_name: str = "features_names",
    ):
        self.model_registry = model_registry
        self.model_dir_name = model_dir_name
        self.top_feature_dir_name = top_feature_dir_name

    def get_latest_dir_path(self) -> Optional[str]:
        """Return the version directory with the highest numeric name."""
        try:
            directory_names = os.listdir(self.model_registry)
            if not directory_names:
                return None

            version_numbers = [int(name) for name in directory_names]
            latest_version = max(version_numbers)
            return os.path.join(self.model_registry, str(latest_version))
        except Exception as exc:
            raise Exception(f"Error in getting the latest directory path: {exc}") from exc

    def get_latest_model_path(self) -> str:
        """Return the path to the latest saved model file."""
        try:
            latest_directory = self.get_latest_dir_path()
            if latest_directory is None:
                raise Exception("Model is not available.")
            return os.path.join(latest_directory, self.model_dir_name, "model.pkl")
        except Exception as exc:
            raise Exception(f"Error in getting the latest model path: {exc}") from exc

    def get_latest_model_feature_names_file_path(self) -> str:
        """Return the path to the latest feature-name file."""
        try:
            latest_directory = self.get_latest_dir_path()
            if latest_directory is None:
                raise Exception("Top features are not available.")
            return os.path.join(latest_directory, self.top_feature_dir_name, "model_feature_names.pkl")
        except Exception as exc:
            raise Exception(f"Error in getting the latest feature file path: {exc}") from exc

    def get_latest_save_dir_path(self) -> str:
        """Return the next model version directory for saving a new model."""
        try:
            latest_directory = self.get_latest_dir_path()
            if latest_directory is None:
                return os.path.join(self.model_registry, "0")

            latest_version_number = int(os.path.basename(latest_directory))
            return os.path.join(self.model_registry, str(latest_version_number + 1))
        except Exception as exc:
            raise Exception(f"Error in getting the next save directory path: {exc}") from exc

    def get_latest_save_model_path(self) -> str:
        """Return the path where the next model should be saved."""
        try:
            save_directory = self.get_latest_save_dir_path()
            return os.path.join(save_directory, self.model_dir_name, "model.pkl")
        except Exception as exc:
            raise Exception(f"Error in getting the save model path: {exc}") from exc

    def get_latest_save_model_feature_names_file_path(self) -> str:
        """Return the path where the next feature-name file should be saved."""
        try:
            save_directory = self.get_latest_save_dir_path()
            return os.path.join(save_directory, self.top_feature_dir_name, "model_feature_names.pkl")
        except Exception as exc:
            raise Exception(f"Error in getting the save feature file path: {exc}") from exc


class Predictor:
    """Use the latest saved model to classify a URL."""

    def __init__(self, model_resolver: ModelResolver):
        self.model_resolver = model_resolver
