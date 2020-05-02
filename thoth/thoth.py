#%%
import os

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
import altair as alt

SEED = 42


@st.cache
def train_model(params, train_x, train_y):
    return DecisionTreeClassifier(**params).fit(train_x, train_y)


@st.cache
def load_process_data(dataloader):
    dataset = dataloader()
    data = pd.DataFrame(dataset["data"], columns=dataset["feature_names"])
    labels = pd.Series(dataset["target"]).map(
        {i: name for i, name in enumerate(dataset["target_names"])}
    )
    data = pd.DataFrame(labels, columns=["label"]).join(data)
    return (
        dataset,
        data,
    )


#%%
@st.cache
def get_metrics(clf, x, y):
    average = "macro" if len(np.unique(y)) > 2 else "micro"
    metrics = {
        "Precision": precision_score(y, clf.predict(x), average=average),
        "Recall": recall_score(y, clf.predict(x), average=average),
        "F1": f1_score(y, clf.predict(x), average=average),
        # "ROC AUC": roc_auc_score(
        #     y, clf.predict_proba(x), average=average, multi_class="ovo"
        # ),
    }
    return pd.DataFrame(metrics, index=[0])


# TODO Improve resolution of images
# TODO Improve caching
@st.cache(show_spinner=False)
def tree_plot(dt: DecisionTreeClassifier, iris: dict):
    # plt.plot(10, 10)
    dot_data = export_graphviz(
        dt,
        out_file=None,
        rounded=True,
        filled=True,
        class_names=iris["target_names"],
        feature_names=iris["feature_names"],
    )
    graph = graphviz.Source(dot_data)
    print(dot_data)
    graph.render(filename="temp", format="png")
    img = Image.open("temp.png")
    os.remove("temp.png")
    os.remove("temp")
    return img


st.title("Decision Trees")

dataloaders = {
    "Breast Cancer": load_breast_cancer,
    "Iris": load_iris,
    "Wine": load_wine,
}
loader_name = st.selectbox("Choose a Dataset", list(dataloaders.keys()))
dataloader = dataloaders.get(loader_name, load_iris)

dataset, data = load_process_data(dataloader)
train_x, test_x, train_y, test_y = train_test_split(
    data.drop("label", axis=1), dataset["target"], train_size=0.8
)

st.dataframe(data, width=10000)
st.header("Simple Data Exploration")
feat = st.selectbox("Feature", data.columns)

chart = (
    alt.Chart(data)
    .mark_bar(opacity=0.8)
    .encode(x=alt.X(feat, bin=feat != "label"), y="count()", color="label",)
)

st.altair_chart(chart, use_container_width=True)

st.write("## Parameter Selection")
params = {"random_state": SEED}
params["criterion"] = st.selectbox("Criterion", ["gini", "entropy"])
params["max_depth"] = st.slider("Max Depth", min_value=1, max_value=30, value=5)
params["min_impurity_decrease"] = st.slider(
    "Min Impurity Decrease", min_value=0.0, max_value=0.2, step=0.001, format="%.3f"
)

# * Advanced parameters
if st.checkbox("Show advanced options"):
    params["splitter"] = st.selectbox("Splitter", ["best", "random"])
    if st.checkbox("Balance classes"):
        params["class_weight"] = "balanced"
    params["min_samples_split"] = st.slider(
        "Min Samples per Split", min_value=0.01, max_value=1.0, step=0.01
    )
    params["max_features"] = st.slider(
        "Number of Features to Consider at Each Split",
        min_value=1,
        max_value=len(dataset["feature_names"]),
        value=len(dataset["feature_names"]),
    )
dt = train_model(params, train_x, train_y)

train_metrics = get_metrics(dt, train_x, train_y).rename(index={0: "Train"})
test_metrics = get_metrics(dt, test_x, test_y).rename(index={0: "Test"})
st.write("## Performance Metrics", train_metrics.append(test_metrics))

st.write("## View Tree")
with st.spinner("Plotting tree..."):
    st.image(tree_plot(dt, dataset), use_column_width=True)

st.write("## Tree Parameters", dt.get_params())

# %%
