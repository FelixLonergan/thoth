import streamlit as st

from thoth.helper import get_handler

st.sidebar.markdown("# Thoth")
article = st.sidebar.selectbox(
    "Choose a Machine Learning method",
    ["Decision Trees", "k-Nearest Neighbours"],
    index=0,
)
show_text = st.sidebar.checkbox("Show article text", value=True)
st.title(article)
handler = get_handler(article)

if show_text:
    st.altair_chart(handler.get_summary(), use_container_width=True)
    st.write(handler.get_section("intro"), unsafe_allow_html=True)
    st.write("---")
handler.render_eda()

handler.render_playground()
