from thoth.handler.generic import Handler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import os
from PIL import Image
from thoth.helper import load_process_data, train_model, get_metrics
import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd

SEED = 42


class DTHandler(Handler):
    def __init__(self):
        super().__init__("dt")
        self.data_options = ["Breast Cancer", "Iris", "Wine"]
        self.summary = pd.DataFrame(
            {
                "Attribute": ["Power", "Interpretability", "Simplicity"],
                "Score": [3, 5, 4],
            },
        )

    def render_playground(self):
        st.header("Model Playground")
        st.subheader("Parameter Selection")
        params = {"random_state": SEED}
        params["criterion"] = st.selectbox("Criterion", ["gini", "entropy"])
        params["max_depth"] = st.slider("Max Depth", min_value=1, max_value=30, value=5)
        params["min_impurity_decrease"] = st.slider(
            "Min Impurity Decrease",
            min_value=0.0,
            max_value=0.2,
            step=0.001,
            format="%.3f",
        )

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
                max_value=len(self.dataset["feature_names"]),
                value=len(self.dataset["feature_names"]),
            )

        dt = train_model(DecisionTreeClassifier, params, self.train_x, self.train_y)

        train_metrics = get_metrics(dt, self.train_x, self.train_y).rename(
            index={0: "Train"}
        )
        test_metrics = get_metrics(dt, self.test_x, self.test_y).rename(
            index={0: "Test"}
        )
        st.subheader("Performance Metrics")
        st.write(train_metrics.append(test_metrics))

        st.subheader("View Tree")
        with st.spinner("Plotting tree..."):
            st.image(self.tree_plot(dt, self.dataset), use_column_width=True)

        st.subheader("Tree Parameters")
        st.write(dt.get_params())

    @staticmethod
    @st.cache(show_spinner=False)
    def tree_plot(dt: DecisionTreeClassifier, dataset: dict):
        # plt.plot(10, 10)
        dot_data = export_graphviz(
            dt,
            out_file=None,
            rounded=True,
            filled=True,
            class_names=dataset["target_names"],
            feature_names=dataset["feature_names"],
        )
        graph = graphviz.Source(dot_data)
        graph.render(filename="temp", format="png")
        img = Image.open("temp.png")
        os.remove("temp.png")
        os.remove("temp")
        return img
