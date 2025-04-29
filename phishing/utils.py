# ------------------ Imports ------------------ #
import os
import sys
import dill
import yaml
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from phishing.logger import logging
from phishing.exception import PhishingException

# Ignore all warnings
warnings.filterwarnings("ignore")

# ------------------ Data Handling Functions ------------------ #

def get_table_as_dataframe(client, table_id: str) -> pd.DataFrame:
    """
    Fetches a table from BigQuery and returns it as a Pandas DataFrame.
    """
    try:
        query = f"SELECT * FROM `{table_id}`"
        return client.query(query).to_dataframe()
    except Exception as e:
        raise Exception(e)


def convert_columns_to_float(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Converts all columns (except excluded ones) in the dataframe to float.
    """
    try:
        for column in df.columns:
            if column not in exclude_columns:
                df[column] = df[column].astype(float)
        return df
    except Exception as e:
        raise PhishingException(e, sys)


def writing_yml_file(file_path: str, data: dict) -> None:
    """
    Writes a dictionary into a YAML (.yml) file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)
    except Exception as e:
        raise PhishingException(e, sys)

# ------------------ Object Save/Load Utilities ------------------ #

def save_object(file_path: str, obj: object) -> None:
    """
    Serializes and saves a Python object using dill.
    """
    try:
        logging.info("Saving object...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Object saved successfully.")
    except Exception as e:
        raise PhishingException(e, sys)


def load_object(file_path: str) -> object:
    """
    Loads and returns a Python object saved with dill.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise PhishingException(e, sys)


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to a binary file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise PhishingException(e, sys)


def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a binary file.
    """
    try:
        with open(file_path, "rb") as file:
            return np.load(file)
    except Exception as e:
        raise PhishingException(e, sys)

# ------------------ Feature Importance & Model Evaluation ------------------ #

def plot_and_save_feature_importances(model, feature_names: list, top_n: int = 25, save_path: str = None) -> list:
    """
    Plots and optionally saves the top N feature importances.
    """
    try:
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_importances, y=top_features)
        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        return top_features
    except Exception as e:
        raise PhishingException(e, sys)


def plot_and_save_model_evaluation(y_true, y_pred_proba, title: str = "", save_path: str = None) -> None:
    """
    Plots and optionally saves the ROC curve.
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{title} Receiver Operating Characteristic (ROC)', fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    except Exception as e:
        raise PhishingException(e, sys)

# ------------------ Top Features Save/Load ------------------ #

def save_features_names(top_features: list, save_path: str) -> None:
    """
    Saves a list of top features into a pickle file.
    """
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(top_features, f)
    except Exception as e:
        raise PhishingException(e, sys)


def load_features_names(file_path: str) -> list:
    """
    Loads a list of top features from a pickle file.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        raise PhishingException(e, sys)
