import inspect
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from .. import SEED, utils

HANDLER_REGISTRY = {}


class BaseHandler(ABC):
    """Abstract base class to handle article specific elements of app"""

    ARTICLE_TITLE: ClassVar[str]
    """The formatted name of the article"""

    DATASETS: ClassVar[List[str]]
    """List of dataset names available in the article"""

    SUMMARY: ClassVar[pd.DataFrame]
    """A set of values defining the properties of the ML method"""

    NAME: ClassVar[str]
    """The programatic name of the handler"""

    def __init__(self) -> None:
        super().__init__()
        self.dataset: Dict[str, Any]
        self.data: pd.DataFrame
        self.train_x: np.ndarray
        self.test_x: np.ndarray
        self.train_y: np.ndarray
        self.test_y: np.ndarray
        self.text_path = Path(__file__).parent.parent.joinpath(
            "static", "text", self.NAME
        )

    def __init_subclass__(cls) -> None:
        if not inspect.isabstract(cls):
            HANDLER_REGISTRY[cls.ARTICLE_TITLE] = cls

        return super().__init_subclass__()

    def render_page(self) -> None:
        """Main method for rendering the entire page"""

        st.title(self.ARTICLE_TITLE)

        self.render_summary()
        with st.expander("Introduction", expanded=True):
            st.write(self.get_section("intro"), unsafe_allow_html=True)

        self.render_eda()
        self.render_playground()

    @st.cache(show_spinner=False)
    def get_section(self, section: str) -> str:
        """Retrieves the contents of a markdown file and returns them as a string

        Each article has the article text stored in markdown files. These are located
        in `text/<article_name>/<section>.md`

        Args:
            section (str): The name of the section to retrieve the markdown for

        Returns:
            The markdown for the required section
        """
        with open(f"{self.text_path}/{section}.md", "r") as file:
            return file.read()

    def render_summary(self) -> None:
        """Create and render a chart showing basic qualities of the handler's ML method"""
        chart = (
            alt.Chart(self.SUMMARY)
            .mark_bar()
            .encode(
                y="Attribute:N",
                x="Score:Q",
                color=alt.Color("Attribute", legend=None),
                tooltip=["Attribute", "Score"],
            )
            .properties(title=f"{self.ARTICLE_TITLE} as a Machine Learning Model")
        )
        st.altair_chart(chart, use_container_width=True)

    @abstractmethod
    def render_playground(self) -> None:
        """Generates and renders the interactive playground for the handler's ML method

        The playground consists of two sections. The first involves choosing the parameters
        of the model, while the second presents relevant plots and metrics.
        """
        raise NotImplementedError

    def render_eda(self, dataset_index: Optional[int] = None) -> None:
        """Generate and render the data selection and exploration section of the article

        Each handler defines some datasets to choose from, and this function renders these options,
        and displays some interactive graphs to explore the data.

        Args:
            dataset_index: If supplied, specifies the index of the default dataset.
        """
        # * Dataset Selection
        st.header("Data Selection and Exploration")
        st.write(self.get_section("eda"))
        dataset_name = st.selectbox(
            "Choose a Dataset", self.DATASETS, index=dataset_index or 0
        )

        with st.spinner("Loading dataset"):
            self.dataset, self.data = utils.load_process_data(dataset_name)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data.drop("label", axis=1),
            self.data["label"],
            train_size=0.8,
            stratify=self.data["label"],
            shuffle=True,
            random_state=SEED,
        )

        # Optionally display dataset information
        with st.expander("Dataset details"):
            st.write(self.dataset["DESCR"])
        st.write(self.data)

        # * EDA
        st.subheader("Simple Exploratory Data Analysis (EDA)")

        # Class Balance
        class_chart = (
            alt.Chart(self.data)
            .mark_bar()
            .encode(
                y=alt.Y("label", axis=alt.Axis(title="Class")),
                x=alt.X("count()", axis=alt.Axis(title="Count")),
                color=alt.Color("label", legend=None),
                tooltip=["label", "count()"],
            )
            .properties(title="Class Distribution")
        )
        st.altair_chart(class_chart, use_container_width=True)

        feature = st.selectbox("Feature", self.data.drop("label", axis=1).columns)

        buffer = 0.1 * (max(self.data[feature]) - min(self.data[feature]))
        density_chart = (
            alt.Chart(self.data)
            .transform_density(
                density=feature,
                groupby=["label"],  # type: ignore
                steps=1000,  # type: ignore
                extent=[
                    min(self.data[feature]) - buffer,
                    max(self.data[feature]) + buffer,
                ],  # type: ignore
            )
            .mark_area()
            .encode(
                alt.X("value:Q", axis=alt.Axis(title=f"{feature}")),
                alt.Y("density:Q", axis=alt.Axis(title="Density")),
                color=alt.Color(
                    "label", legend=alt.Legend(orient="bottom", title="Class")
                ),
                opacity=alt.OpacityValue(0.8),
                tooltip=["label", "density:Q"],
            )
            .properties(title=f"Distribution of {feature} for each class")
        )
        st.altair_chart(density_chart, use_container_width=True)
