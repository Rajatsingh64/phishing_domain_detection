import os
import sys
import dill
import yaml
import pickle
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from phishing.logger import logging
from phishing.exception import PhishingException

# Ignore warnings
warnings.filterwarnings("ignore")


def get_table_as_dataframe(client, table_id: str) -> pd.DataFrame:
    """
    Fetches data from a BigQuery table and returns it as a Pandas DataFrame.

    Args:
        client: Initialized BigQuery client.
        table_id (str): Fully qualified table ID in the format 'project.dataset.table'.

    Returns:
        pd.DataFrame: Table data as a dataframe.
    """
    try:
        query = f"SELECT * FROM `{table_id}`"
        data = client.query(query).to_dataframe()
        return data
    except Exception as e:
        raise Exception(e)


def convert_columns_to_float(df: pd.DataFrame, exclude_columns: list) -> pd.DataFrame:
    """
    Converts all DataFrame columns to float type except specified columns.

    Args:
        df (pd.DataFrame): Input dataframe.
        exclude_columns (list): List of column names to exclude.

    Returns:
        pd.DataFrame: DataFrame with specified columns converted to float.
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

    Args:
        file_path (str): Path to save the YAML file.
        data (dict): Data dictionary to write.
    """
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)

        with open(file_path, "w") as file_writer:
            yaml.dump(data, file_writer)

    except Exception as e:
        raise PhishingException(e, sys)


def save_object(file_path: str, obj: object) -> None:
    """
    Serializes and saves a Python object using dill.

    Args:
        file_path (str): File path to save object.
        obj (object): Python object to serialize.
    """
    try:
        logging.info("Entered the save_object method.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method.")
    except Exception as e:
        raise PhishingException(e, sys)


def load_object(file_path: str) -> object:
    """
    Loads and returns a Python object from a file using dill.

    Args:
        file_path (str): Path from where to load the object.

    Returns:
        object: Loaded Python object.
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

    Args:
        file_path (str): Path to save the array.
        array (np.ndarray): NumPy array to save.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise PhishingException(e, sys)


def load_numpy_array(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a binary file.

    Args:
        file_path (str): Path to the saved array.

    Returns:
        np.ndarray: Loaded NumPy array.
    """
    try:
        with open(file_path, "rb") as file:
            data_array = np.load(file)
        return data_array

    except Exception as e:
        raise PhishingException(e, sys)


def plot_and_save_feature_importances(model, feature_names: list, top_n: int = 25, save_path: str = None) -> list:
    """
    Plots and optionally saves the top_n feature importances from a model.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): Names of features.
        top_n (int): Number of top features to display.
        save_path (str, optional): Path to save the plot.

    Returns:
        list: List of top feature names.
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
    Plots and optionally saves a ROC curve for model evaluation.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred_proba (array-like): Predicted positive class probabilities.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot.

    Returns:
        None
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


def save_top_features(top_features: list, save_path: str) -> None:
    """
    Saves a list of top features into a pickle file.

    Args:
        top_features (list): List of top features.
        save_path (str): Path to save the list.
    """
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(top_features, f)
    except Exception as e:
        raise PhishingException(e, sys)


def load_top_features(file_path: str) -> list:
    """
    Loads a list of top features from a pickle file.

    Args:
        file_path (str): Path to the saved features.

    Returns:
        list: Loaded list of top features.
    """
    try:
        with open(file_path, 'rb') as f:
            top_features = pickle.load(f)
        return top_features
    except Exception as e:
        raise PhishingException(e, sys)
