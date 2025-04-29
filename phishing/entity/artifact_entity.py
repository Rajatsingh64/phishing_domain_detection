from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Represents outputs generated during the data ingestion process.
    """
    feature_store_file_path: str  # Path to complete dataset
    train_file_path: str          # Path to training dataset
    test_file_path: str           # Path to testing dataset


@dataclass
class DataValidationArtifact:
    """
    Represents outputs from the data validation step.
    """
    report_yml_file_path: str  # Path to validation report YAML


@dataclass
class DataTransformationArtifact:
    """
    Represents outputs from the data transformation step.
    """
    transformed_original_data_file_path: str  # Path to transformed dataset


@dataclass
class ModelTrainingArtifact:
    """
    Represents outputs from the model training step.
    """
    model_file_path: str                    # Path to saved model
    train_accuracy_score: float             # Training accuracy score
    test_accuracy_score: float              # Testing accuracy score
    model_feature_names_file_path: str      # Path to top features model


@dataclass
class ModelEvaluationArtifact:
    """
    Represents evaluation results for the trained model.
    """
    is_model_accepted: bool   # Whether model is accepted
    improved_accuracy: bool   # Whether model improved over previous


@dataclass
class ModelPusherArtifact:
    """
    Represents artifacts related to model pushing (deployment).
    """
    pusher_model_dir: str     # Directory where model is pushed
    saved_model_dir: str      # Directory where model is saved
