from typing import Tuple, Type

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import f1_score, precision_score, recall_score

from thoth.handler.BaseHandler import BaseHandler
from thoth.handler.DTHandler import DTHandler


@st.cache
def train_model(model, params: dict, train_x: pd.DataFrame, train_y: pd.Series):
    """Initialise and train a given model with the provided parameters and data

    Args:
        model: The model architecture to use
        params (dict): A parameter dictionary containing parameter_name: value pairs
        train_x (pd.DataFrame): The training data, should be of shape (n_samples, n_features)
        train_y (pd.Series): The training labels, should be of shape (n_samples)

    Returns:
        model: The trained model
    """
    return model(**params).fit(train_x, train_y)


@st.cache(show_spinner=False)
def load_process_data(dataset_name: str) -> Tuple[dict, pd.DataFrame]:
    """Loads and formats a dataset based on its name

    Args:
        dataset_name (str): The name of the dataset to load and process

    Returns:
        Tuple[dict, pd.DataFrame]: A tuple of the dataset metadata dict and the data
    """
    dataloaders = {
        "Breast Cancer": load_breast_cancer,
        "Iris": load_iris,
        "Wine": load_wine,
    }
    dataloader = dataloaders.get(dataset_name)
    dataset = dataloader()
    dataset["DESCR"] = dataset["DESCR"].split(":", 1)[1]
    data = pd.DataFrame(dataset.pop("data"), columns=dataset["feature_names"])
    labels = pd.Series(dataset.pop("target")).map(
        {i: name for i, name in enumerate(dataset["target_names"])}
    )
    data = pd.DataFrame(labels, columns=["label"]).join(data)
    return (
        dataset,
        data,
    )


@st.cache
def get_metrics(clf, x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Evaluate the performance of a classifier on a given dataset

    Args:
        clf: The trained classifier to evaluate
        x (pd.DataFrame): The input data
        y (pd.Series): Lables for each sample in the input data

    Returns:
        pd.DataFrame: A DataFrame containing the Precision, Recall and F1 scores
            Macro average is used for multiclass datasets, and micro average is used
            for binary classification.
    """
    average = "macro" if len(np.unique(y)) > 2 else "micro"
    metrics = {
        "Precision": precision_score(y, clf.predict(x), average=average),
        "Recall": recall_score(y, clf.predict(x), average=average),
        "F1": f1_score(y, clf.predict(x), average=average),
    }
    return pd.DataFrame(metrics, index=[0])


def get_handler(handler_name: str) -> Type[BaseHandler]:
    """Returns the appropriate Handler based off the name of the article

    Args:
        handler_name (str): The name of the handler to return

    Returns:
        Type[BaseHandler]: The appropriate page handler
    """
    handlers = {"Decision Tree": DTHandler()}
    return handlers.get(handler_name, DTHandler())
