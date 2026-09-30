import os
import sys
import warnings
import pickle

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import auc, roc_curve

from phishing.exception import PhishingException
from phishing.logger import logging

warnings.filterwarnings("ignore")


def get_table_as_dataframe(client, table_id: str) -> pd.DataFrame:
    """Fetch a BigQuery table and return it as a pandas DataFrame."""
    try:
        query = f"SELECT * FROM `{table_id}`"
        return client.query(query).to_dataframe()
    except Exception as exc:
        raise Exception(exc) from exc


def convert_columns_to_float(dataframe: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """Convert non-excluded columns to float values."""
    try:
        for column_name in dataframe.columns:
            if column_name not in exclude_columns:
                dataframe[column_name] = dataframe[column_name].astype(float)
        return dataframe
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def writing_yml_file(file_path: str, data: dict) -> None:
    """Write a dictionary to a YAML file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(data, yaml_file)
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def save_object(file_path: str, obj: object) -> None:
    """Serialize a Python object and save it to disk."""
    try:
        logging.info("Saving object to disk.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as output_file:
            dill.dump(obj, output_file)
        logging.info("Object saved successfully.")
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def load_object(file_path: str) -> object:
    """Load a serialized Python object from disk."""
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as input_file:
            return dill.load(input_file)
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """Save a NumPy array to a binary file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as output_file:
            np.save(output_file, array)
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def load_numpy_array(file_path: str) -> np.ndarray:
    """Load a NumPy array from a binary file."""
    try:
        with open(file_path, "rb") as input_file:
            return np.load(input_file)
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def plot_and_save_feature_importances(model, feature_names: list, top_n: int = 25, save_path: str = None) -> list:
    """Plot and save the most important feature importances."""
    try:
        importances = model.feature_importances_
        top_indexes = importances.argsort()[::-1][:top_n]
        selected_features = [feature_names[index] for index in top_indexes]
        selected_importances = importances[top_indexes]

        plt.figure(figsize=(10, 8))
        sns.barplot(x=selected_importances, y=selected_features)
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        return selected_features
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def plot_and_save_model_evaluation(y_true, y_pred_proba, title: str = "", save_path: str = None) -> None:
    """Plot and save the ROC curve for a trained model."""
    try:
        figure, axis = plt.subplots(figsize=(8, 6))
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_pred_proba)
        roc_auc_value = auc(false_positive_rate, true_positive_rate)

        axis.plot(false_positive_rate, true_positive_rate, color="#1f77b4", lw=2,
                  label=f"ROC curve (AUC = {roc_auc_value:.2f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set_xlabel("False Positive Rate")
        axis.set_ylabel("True Positive Rate")
        axis.set_title(f"{title} Receiver Operating Characteristic (ROC)", fontsize=14)
        axis.legend(loc="lower right")
        axis.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def save_features_names(top_features: list, save_path: str) -> None:
    """Save the selected feature names to disk."""
    try:
        with open(save_path, "wb") as feature_file:
            pickle.dump(top_features, feature_file)
    except Exception as exc:
        raise PhishingException(exc, sys) from exc


def load_features_names(file_path: str) -> list:
    """Load feature names from a pickle file."""
    try:
        with open(file_path, "rb") as feature_file:
            return pickle.load(feature_file)
    except Exception as exc:
        raise PhishingException(exc, sys) from exc
