import copy

import altair as alt
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.neighbors import KNeighborsClassifier

import thoth.helper as helper
from thoth.handler.BaseHandler import BaseHandler


class KNNHandler(BaseHandler):
    """Page handler for the Knn article (short name 'knn')
    """

    def __init__(self):
        super().__init__("knn")
        self.data_options = ["Moons", "Blobs", "Circles", "Classification"]
        self.summary = pd.DataFrame(
            {
                "Attribute": ["Power", "Interpretability", "Simplicity"],
                "Score": [2.5, 4, 4],
            },
        )

    def render_eda(self, index=0):
        super().render_eda(index=self.data_options.index("Moons"))
        chart = (
            alt.Chart(self.data)
            .mark_circle()
            .encode(
                x="Feature_1:Q",
                y="Feature_2:Q",
                color="label:N",
                tooltip=["label", "Feature_1", "Feature_2"],
            )
        ).properties(height=400, title=f"Feature_1 vs. Feature_2")
        st.altair_chart(chart, use_container_width=True)

    def render_playground(self):
        st.header("Model Playground")
        st.write(self.get_section("playground"))
        st.subheader("Parameter Selection")

        params = {}
        params["n_neighbors"] = st.number_input(
            "Number of Neighbours (k):", value=5, min_value=1
        )
        params["metric"] = st.selectbox(
            "Distance Metric:", ["euclidean", "manhattan", "chebyshev", "minkowski"]
        )
        if params["metric"] == "minkowski":
            params["p"] = st.number_input("Minkowski Power (p):", value=3, min_value=1)

        params["weights"] = st.selectbox(
            "Weight of Neighbour:", ["uniform", "distance"]
        )
        knn = helper.train_model(
            KNeighborsClassifier, params, self.train_x, self.train_y
        )

        train_metrics = helper.get_metrics(knn, self.train_x, self.train_y).rename(
            index={0: "Train"}
        )
        test_metrics = helper.get_metrics(knn, self.test_x, self.test_y).rename(
            index={0: "Test"}
        )
        st.subheader("Performance Metrics")
        st.write(train_metrics.append(test_metrics))

        st.subheader("View Decision Boundary")
        plt.figure(dpi=30)
        plot_decision_regions(
            self.data.drop("label", axis=1).values,
            self.data["label"].values,
            clf=knn,
            colors="#7093b9,#f79d46,#e97978",
        )
        plt.xticks([])
        plt.yticks([])
        st.pyplot()
        st.subheader("k-NN Parameters")
        st.write(knn.get_params())
