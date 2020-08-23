import os.path
from abc import ABC, abstractmethod

import altair as alt
import streamlit as st
from sklearn.model_selection import train_test_split

import thoth.helper as helper
from thoth import SEED


class BaseHandler(ABC):
    """Abstract base class to handle article specific elements of app

    Args:
        name (str): The short name of the article
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.data_options = []
        self.dataset = None
        self.data = None
        self.train_x = None
        self.test_x = None
        self.train_y = None
        self.test_y = None
        self.summary = None

        cwd = os.path.abspath(os.path.dirname(__file__))
        self.text_path = os.path.join(cwd, f"../../text/{name}")

    @st.cache(show_spinner=False)
    def get_section(self, section: str) -> str:
        """Retrieves the contents of a markdown file and returns them as a string

        Each article has the article text stored in markdown files. These are located
        in `text/<article_name>/<section>.md`

        Args:
            section (str): The name of the section to retrieve the markdown for

        Returns:
            str: The markdown for the required section
        """
        with open(f"{self.text_path}/{section}.md", "r") as file:
            return file.read()

    def get_summary(self) -> alt.Chart:
        """Create and return an altair chart showing the basic qualities of the handler's ML method

        Returns:
            alt.Chart: The attribute summary chart
        """
        return (
            alt.Chart(self.summary)
            .mark_bar()
            .encode(
                alt.X("Score:Q", scale=alt.Scale(domain=(0, 5))),
                y="Attribute:N",
                tooltip=["Attribute", "Score"],
                color="Attribute",
            )
            .properties(title="Decision Trees as a Machine Learning Model")
            .configure_axis(grid=False, tickCount=5)
        )

    @abstractmethod
    def render_playground(self) -> None:
        """Generates and renders the interactive playground for the handler's ML method

        The playground consists of two sections. The first involves choosing the parameters
        of the model, while the second presents relevant plots and metrics.
        """

    def render_eda(self, index=0):
        """Generate and render the data selection and exploration section of the article

        Each handler defines some datasets to choose from, and this function renders these options,
        and displays some interactive graphs to explore the data.

        Args:
            index (int, optional): The index of the default dataset. Defaults to 0.
        """
        # * Dataset Selection
        st.header("Data Selection and Exploration")
        st.write(self.get_section("eda"))
        dataset_name = st.selectbox("Choose a Dataset", self.data_options, index=index)

        with st.spinner("Loading dataset"):
            self.dataset, self.data = helper.load_process_data(dataset_name)

        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data.drop("label", axis=1),
            self.data["label"],
            train_size=0.8,
            stratify=self.data["label"],
            shuffle=True,
            random_state=SEED,
        )

        # Optionally display dataset information
        if self.dataset.get("DESCR"):
            if st.checkbox("Display dataset information"):
                st.write(self.dataset["DESCR"])
                st.write("---")

        st.write(self.data)

        # * EDA
        st.subheader("Simple Exploratory Data Analysis (EDA)")

        # Class Balance
        class_chart = (
            alt.Chart(self.data)
            .mark_bar()
            .encode(
                y=alt.Y("label:N", axis=alt.Axis(title="Class")),
                x=alt.X("count()", axis=alt.Axis(title="Count")),
                color=alt.Color("label:N", legend=None),
                tooltip=["label", "count()"],
            )
            .properties(title="Class Distribution")
        )
        st.altair_chart(class_chart, use_container_width=True)

        feature = st.selectbox("Feature", self.data.drop("label", axis=1).columns)

        density_chart = (
            alt.Chart(self.data)
            .transform_density(
                density=feature,
                groupby=["label"],
                steps=1000,
                extent=[0.9 * min(self.data[feature]), 1.1 * max(self.data[feature])],
            )
            .mark_area()
            .encode(
                alt.X("value:Q", axis=alt.Axis(title=f"{feature}")),
                alt.Y("density:Q", axis=alt.Axis(title="Density")),
                color=alt.Color(
                    "label:N", legend=alt.Legend(orient="bottom", title="Class")
                ),
                opacity=alt.OpacityValue(0.8),
                tooltip=["label", "density:Q"],
            )
            .properties(title=f"Distribution of {feature} for each class")
        )
        st.altair_chart(density_chart, use_container_width=True)
