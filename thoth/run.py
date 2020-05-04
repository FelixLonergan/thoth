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
from handler.generic import Handler
from handler.DTHandler import DTHandler
from helper import *

st.title("Decision Trees")
handler = DTHandler()
st.write(handler.get_section("intro"))
st.altair_chart(handler.get_summary(), use_container_width=True)

handler.render_eda()

handler.render_playground()
