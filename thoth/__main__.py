import streamlit as st
from thoth.helper import get_handler

article = st.sidebar.selectbox(
    "Choose a Machine Learning method", ["Decision Trees"], index=0,
)

st.title(article)
handler = get_handler(article)
st.write(handler.get_section("intro"))
st.altair_chart(handler.get_summary(), use_container_width=True)

handler.render_eda()

handler.render_playground()
