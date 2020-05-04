import os.path
from abc import ABC, abstractmethod
import streamlit as st
from sklearn.model_selection import train_test_split
from thoth.helper import load_process_data
import altair as alt

SEED = 42


class Handler(ABC):
    def __init__(self, name):
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
        with open(f"{self.text_path}/{section}.md", "r") as f:
            return f.read()

    def get_summary(self):
        return (
            alt.Chart(self.summary)
            .mark_bar()
            .encode(
                y="Attribute:N",
                x="Score:Q",
                # color="Attribute",
                tooltip=["Attribute", "Score"],
            )
            .properties(title="Decision Trees as a Machine Learning Model")
        )

    @abstractmethod
    def render_playground(self):
        pass

    def render_eda(self):
        # * Dataset Selection
        st.write("---")
        st.header("Data Selection and Exploration")
        dataset_name = st.selectbox("Choose a Dataset", self.data_options)
        self.dataset, self.data = load_process_data(dataset_name)
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data.drop("label", axis=1),
            self.data["label"],
            train_size=0.8,
            stratify=self.data["label"],
            shuffle=True,
            random_state=SEED,
        )

        # Optionally display dataset information
        if st.checkbox("Display dataset information"):
            st.write(self.dataset["DESCR"].split(":", 1)[1])
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
                color="label",
                tooltip=["label", "count()"],
            )
            .properties(title="Class Distribution")
        )
        st.altair_chart(class_chart, use_container_width=True)

        feat = st.selectbox("Feature", self.data.drop("label", axis=1).columns)
        density_chart = (
            alt.Chart(self.data)
            .transform_density(
                density=feat,
                groupby=["label"],
                steps=1000,
                # counts=True,
                extent=[min(self.data[feat]), max(self.data[feat])],
            )
            .mark_area()
            .encode(
                alt.X(f"value:Q", axis=alt.Axis(title=f"{feat}")),
                alt.Y("density:Q", axis=alt.Axis(title="Density")),
                alt.Color("label:N"),
                tooltip=["label", "density:Q"],
            )
            .properties(title=f"Distribution of {feat} for each class")
        )
        st.altair_chart(density_chart, use_container_width=True)
