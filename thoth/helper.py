from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
import pandas as pd
import numpy as np
import streamlit as st
from thoth.handler.DTHandler import DTHandler


@st.cache
def train_model(model, params, train_x, train_y):
    return model(**params).fit(train_x, train_y)


@st.cache(show_spinner=False)
def load_process_data(dataset_name):
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
def get_metrics(clf, x, y):
    average = "macro" if len(np.unique(y)) > 2 else "micro"
    metrics = {
        "Precision": precision_score(y, clf.predict(x), average=average),
        "Recall": recall_score(y, clf.predict(x), average=average),
        "F1": f1_score(y, clf.predict(x), average=average),
    }
    return pd.DataFrame(metrics, index=[0])


def get_handler(handler_name):
    handlers = {"Decision Tree": DTHandler()}
    return handlers.get(handler_name, DTHandler())
