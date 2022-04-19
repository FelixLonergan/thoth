import sys
from typing import Any, Dict, Tuple, Type, TypeVar, Union, cast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import f1_score, precision_score, recall_score

# Protocol is only available in the typing module for python 3.8+
if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


class ScikitModel(Protocol):
    """A protocol for compatible scikit-learn models

    See the scikit-learn documentation for details
    [here](https://scikit-learn.org/stable/developers/develop.html#apis-of-scikit-learn-objects)
    """

    def fit(
        self: "ScikitModelT",
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, pd.DataFrame, np.ndarray],
    ) -> "ScikitModelT":
        """Fits the model to a dataset"""
        ...

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict with the model on a dataset"""
        ...


ScikitModelT = TypeVar("ScikitModelT", bound=ScikitModel)


@st.cache
def train_model(
    model: Type[ScikitModelT],
    params: Dict[str, Any],
    train_x: Union[pd.DataFrame, np.ndarray],
    train_y: Union[pd.Series, np.ndarray],
) -> ScikitModelT:
    """Initialise and train a given scikit-learn model with the provided parameters and data

    Args:
        model: The model architecture to use
        params (dict): A parameter dictionary containing parameter_name: value pairs
        train_x (pd.DataFrame): The training data, should be of shape (n_samples, n_features)
        train_y (pd.Series): The training labels, should be of shape (n_samples)

    Returns:
        model: The trained model
    """
    return model(**params).fit(train_x, train_y)  # type: ignore


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
    dataloader = dataloaders[dataset_name]
    dataset = cast(Dict[str, Any], dataloader())
    dataset["DESCR"] = dataset["DESCR"].split(":", 1)[1]
    data = pd.DataFrame(dataset.pop("data"), columns=dataset["feature_names"])
    labels = pd.Series(dataset.pop("target")).map(
        dict(enumerate(dataset["target_names"]))
    )
    data = pd.DataFrame(labels, columns=["label"]).join(data)
    return (
        dataset,
        data,
    )


@st.cache
def get_metrics(
    clf: ScikitModel,
    x: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
) -> pd.DataFrame:
    """Evaluate the performance of a scikit-learn predictor on a given dataset

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
