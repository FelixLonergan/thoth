from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, export_graphviz

from .. import SEED, utils
from .base_handler import BaseHandler


class DTHandler(BaseHandler):
    """Page handler for the Decision Tree article (short name 'dt')"""

    ARTICLE_TITLE = "Decision Trees"
    DATASETS = ["Breast Cancer", "Iris", "Wine"]
    SUMMARY = pd.DataFrame(
        {
            "Attribute": ["Power", "Interpretability", "Simplicity"],
            "Score": [2, 5, 4],
        },
    )
    NAME = "dt"

    def render_eda(self, dataset_index: Optional[int] = None) -> None:
        if dataset_index is None:
            dataset_index = self.DATASETS.index("Iris")
        return super().render_eda(dataset_index=dataset_index)

    def render_playground(self) -> None:
        st.header("Model Playground")
        st.write(self.get_section("playground"))
        st.subheader("Parameter Selection")

        if any(
            data is None
            for data in (self.train_x, self.train_y, self.test_x, self.test_y)
        ):
            raise ValueError(
                "A dataset must be chosen before the playground can be rendered!"
            )

        params: Dict[str, Any] = {
            "random_state": SEED,
            "criterion": st.selectbox(
                "Splitting criterion:", ["gini", "entropy"], index=1
            ),
            "max_depth": st.slider(
                "Maximum tree depth:", min_value=1, max_value=10, value=5
            ),
            "min_samples_split": st.slider(
                "Minimum number of samples required to split",
                min_value=2,
                max_value=len(self.train_x),
                step=1,
            ),
        }

        with st.expander("Advanced parameters"):
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

        decision_tree = utils.train_model(
            DecisionTreeClassifier, params, self.train_x, self.train_y
        )

        train_metrics = utils.get_metrics(decision_tree, self.train_x, self.train_y)
        train_metrics = train_metrics.set_axis(["Train"], axis="index")
        test_metrics = utils.get_metrics(decision_tree, self.test_x, self.test_y)
        test_metrics = test_metrics.set_axis(["Test"], axis="index")

        st.subheader("Performance Metrics")
        st.write(train_metrics.append(test_metrics))

        st.subheader("View Tree")
        with st.spinner("Plotting tree"):
            tree_dot = export_graphviz(
                decision_tree,
                out_file=None,
                rounded=True,
                filled=True,
                class_names=self.dataset["target_names"],
                feature_names=self.dataset["feature_names"],
            )
            st.graphviz_chart(tree_dot, use_container_width=True)

        st.subheader("Tree Parameters")
        st.write(decision_tree.get_params())
