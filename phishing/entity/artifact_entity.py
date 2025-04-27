from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Class to represent the artifacts generated during the data ingestion process.

    Attributes:
        feature_store_file_path (str): Path to the feature store file where the entire dataset is stored.
        train_file_path (str): Path to the training dataset after the split.
        test_file_path (str): Path to the testing dataset after the split.
    """

    feature_store_file_path: str
    train_file_path: str
    test_file_path: str


@dataclass
class DataValidationArtifact:
    
    report_yml_file_path: str


@dataclass
class DataTransformationArtifact:
    
    transformed_original_data_file_path:str
   
@dataclass
class ModelTrainingArtifact:

    model_file_path: str
    train_accuracy_score:float
    test_accuracy_score:float
    top_features_model_trained_file_path: str