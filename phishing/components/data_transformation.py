from phishing.exception import PhishingException  # Corrected typo here
from phishing.logger import logging
import numpy as np
import pandas as pd
from phishing.entity import artifact_entity, config_entity
from imblearn.over_sampling import SMOTE
from phishing.config import TARGET_COLUMN
from phishing import utils
import os, sys


class DataTransformation:
    def __init__(self, 
                 data_transformation_config: config_entity.DataTransformationConfig, 
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        """
        Initialize the DataTransformation class.

        Args:
            data_transformation_config (DataTransformationConfig): Configuration for data transformation.
            data_ingestion_artifact (DataIngestionArtifact): Artifact containing paths to ingested data.
        """
        try:
            logging.info(f"{'>'*20} Data Transformation Started {'<'*20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise PhishingException(e, sys)  # Corrected typo here

    def remove_high_correlation_features(self, df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        """
        Remove highly correlated features from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            threshold (float): Correlation threshold above which features are dropped.

        Returns:
            pd.DataFrame: DataFrame with highly correlated features removed.
        """
        try:
            logging.info(f"Checking for highly correlated features.")

            # Step 1: Compute absolute correlation matrix
            corr_matrix = df.corr().abs()

            # Step 2: Select the upper triangle of the correlation matrix
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

            # Step 3: Identify features to drop
            to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
            logging.info(f"Removing highly correlated columns: {to_drop}")

            # Step 4: Drop selected features
            return df.drop(columns=to_drop)

        except Exception as e:
            raise PhishingException(e, sys)  # Corrected typo here

    def initiate_data_transformation(self) -> artifact_entity.DataTransformationArtifact:
        """
        Initiate the data transformation process.

        Returns:
            DataTransformationArtifact: Artifact containing paths to transformed datasets.
        """
        try:
            # Reading data
            logging.info(f"Reading original data as DataFrame for data transformation.")
            df = pd.read_csv(self.data_ingestion_artifact.feature_store_file_path)

            # Splitting input and target features
            logging.info(f"Splitting input and target features for transformation.")
            input_df = df.drop(TARGET_COLUMN, axis=1)
            df_target_feature = df[TARGET_COLUMN]
            
            # Apply SMOTE to original data only
            logging.info(f"Applying SMOTE for resampling to balance the target variable in original data.")
            smote = SMOTE(random_state=42)

            logging.info(f"Before resampling (Original data): Input shape {input_df.shape}, "
                         f"Target shape {df_target_feature.shape}")

            input_df, df_target_feature = smote.fit_resample(input_df, df_target_feature)

            logging.info(f"After resampling (Original data): Input shape {input_df.shape}, "
                         f"Target shape {df_target_feature.shape}")

            # Remove highly correlated features
            logging.info(f"Removing highly correlated features from original data.")
            input_df = self.remove_high_correlation_features(df=input_df, 
                                                             threshold=self.data_transformation_config.correlation_threshold)

            # Combine input features with targets 
            logging.info(f"Combining the resampled input features and target variables for original data.")
            original_data = pd.concat([input_df , df_target_feature], axis=1)

            # Save transformed datasets
            logging.info(f"Saving original data to transformation artifact folder.")
            #creating data_transformed directory if not available
            transformed_dir=os.path.dirname(self.data_transformation_config.transformed_original_data_file_path)
            os.makedirs(transformed_dir , exist_ok=True)
            original_data.to_csv(self.data_transformation_config.transformed_original_data_file_path , index=False , header=True)
            
            # Create and return DataTransformationArtifact
            data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                transformed_original_data_file_path=self.data_transformation_config.transformed_original_data_file_path,
            )

            logging.info(f"Data transformation artifact created successfully: {data_transformation_artifact}")
            return data_transformation_artifact

        except Exception as e:
            raise PhishingException(e, sys)  
