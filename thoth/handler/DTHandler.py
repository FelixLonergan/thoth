import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import thoth.helper as helper
from thoth import SEED
from thoth.handler.BaseHandler import BaseHandler


class DTHandler(BaseHandler):
    """Page handler for the Decision Tree article (short name 'dt')
    """

    def __init__(self):
        super().__init__("dt")
        self.data_options = ["Breast Cancer", "Iris", "Wine"]
        self.summary = pd.DataFrame(
            {
                "Attribute": ["Power", "Interpretability", "Simplicity"],
                "Score": [2, 5, 4],
            },
        )

    def render_eda(self, index=0):
        return super().render_eda(index=self.data_options.index("Iris"))

    def render_playground(self):
        st.header("Model Playground")
        st.write(self.get_section("playground"))
        st.subheader("Parameter Selection")
        params = {"random_state": SEED}
        params["criterion"] = st.selectbox(
            "Splitting criterion:", ["gini", "entropy"], index=1
        )
        params["max_depth"] = st.slider(
            "Maximum tree depth:", min_value=1, max_value=10, value=5
        )
        params["min_samples_split"] = st.slider(
            "Minimum number of samples required to split",
            min_value=2,
            max_value=len(self.train_x),
            step=1,
        )

        if st.checkbox("Show advanced options"):
            params["splitter"] = st.selectbox(
                "How to select feature to split by:", ["best", "random"]
            )
            params["min_impurity_decrease"] = st.slider(
                f"Minimum decrease in {params['criterion']} required to perform a split:",
                min_value=0.0,
                max_value=0.5,
                step=0.001,
                format="%.3f",
            )
            if st.checkbox("Balance classes inversely proportional to their frequency"):
                params["class_weight"] = "balanced"
            params["max_features"] = st.slider(
                "Number of features to consider at each split (randomly selected at each branch):",
                min_value=1,
                max_value=len(self.dataset["feature_names"]),
                value=len(self.dataset["feature_names"]),
            )

        dt = helper.train_model(
            DecisionTreeClassifier, params, self.train_x, self.train_y
        )
        train_metrics = helper.get_metrics(dt, self.train_x, self.train_y).rename(
            index={0: "Train"}
        )
        test_metrics = helper.get_metrics(dt, self.test_x, self.test_y).rename(
            index={0: "Test"}
        )
        st.subheader("Performance Metrics")
        st.write(train_metrics.append(test_metrics))

        st.subheader("View Tree")
        with st.spinner("Plotting tree"):
            tree_dot = export_graphviz(
                dt,
                out_file=None,
                rounded=True,
                filled=True,
                class_names=self.dataset["target_names"],
                feature_names=self.dataset["feature_names"],
            )
            st.graphviz_chart(tree_dot, use_container_width=True)

        st.subheader("Tree Parameters")
        st.write(dt.get_params())
