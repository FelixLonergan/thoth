import pandas as pd
from sklearn.datasets import (
    make_blobs,
    make_circles,
    make_moons,
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
)

from thoth import SEED

FEATURE_NAMES = ["Feature_1", "Feature_2"]
N_SAMPLES = 1000


def get_moons():
    x, y = make_moons(N_SAMPLES, noise=0.2, shuffle=True, random_state=SEED)
    data = pd.DataFrame(x, columns=FEATURE_NAMES)
    data["label"] = y
    dataset = {
        "target_names": ["0", "1"],
        "feature_names": FEATURE_NAMES,
    }
    return dataset, data


def get_blobs():
    x, y = make_blobs(
        N_SAMPLES, n_features=2, cluster_std=3, shuffle=True, random_state=SEED
    )
    data = pd.DataFrame(x, columns=FEATURE_NAMES)
    data["label"] = y
    dataset = {
        "target_names": ["0", "1"],
        "feature_names": FEATURE_NAMES,
    }
    return dataset, data


def get_circles():
    x, y = make_circles(
        N_SAMPLES, noise=0.1, factor=0.8, shuffle=True, random_state=SEED
    )
    data = pd.DataFrame(x, columns=FEATURE_NAMES)
    data["label"] = y
    dataset = {
        "target_names": ["0", "1"],
        "feature_names": FEATURE_NAMES,
    }
    return dataset, data


def get_classification():
    x, y = make_classification(
        N_SAMPLES,
        n_features=2,
        n_clusters_per_class=1,
        n_informative=2,
        n_redundant=0,
        flip_y=0.1,
        class_sep=0.5,
        shuffle=True,
        random_state=SEED,
    )
    data = pd.DataFrame(x, columns=FEATURE_NAMES)
    data["label"] = y
    dataset = {
        "target_names": ["0", "1"],
        "feature_names": FEATURE_NAMES,
    }
    return dataset, data


def get_breast_cancer():
    dataset = load_breast_cancer()
    dataset["DESCR"] = dataset["DESCR"].split(":", 1)[1]
    data = pd.DataFrame(dataset.pop("data"), columns=dataset["feature_names"])
    labels = pd.Series(dataset.pop("target")).map(
        {i: name for i, name in enumerate(dataset["target_names"])}
    )
    data = pd.DataFrame(labels, columns=["label"]).join(data)
    dataset.pop("filename")
    return dataset, data


def get_iris():
    dataset = load_iris()
    dataset["DESCR"] = dataset["DESCR"].split(":", 1)[1]
    data = pd.DataFrame(dataset.pop("data"), columns=dataset["feature_names"])
    labels = pd.Series(dataset.pop("target")).map(
        {i: name for i, name in enumerate(dataset["target_names"])}
    )
    data = pd.DataFrame(labels, columns=["label"]).join(data)
    dataset.pop("filename")
    return dataset, data


def get_wine():
    dataset = load_wine()
    dataset["DESCR"] = dataset["DESCR"].split(":", 1)[1]
    data = pd.DataFrame(dataset.pop("data"), columns=dataset["feature_names"])
    labels = pd.Series(dataset.pop("target")).map(
        {i: name for i, name in enumerate(dataset["target_names"])}
    )
    data = pd.DataFrame(labels, columns=["label"]).join(data)
    dataset.pop("filename")
    return dataset, data
