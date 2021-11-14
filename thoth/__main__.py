import streamlit as st

from thoth.helper import get_handler

ARTICLES = ["Decision Trees"]


def main():
    """Main entry point for the Thoth application that handles overall page structure"""
    st.sidebar.title("Thoth")
    article = st.sidebar.selectbox(
        "Choose a Machine Learning method",
        ARTICLES,
        index=0,
    )

    st.title(article)
    handler = get_handler(article)

    st.altair_chart(handler.get_summary(), use_container_width=True)
    st.write(handler.get_section("intro"), unsafe_allow_html=True)

    st.write("---")

    handler.render_eda()
    handler.render_playground()


if __name__ == "__main__":
    main()
