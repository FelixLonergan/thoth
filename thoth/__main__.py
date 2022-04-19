import streamlit as st

from .handler import HANDLER_REGISTRY


def main() -> None:
    """Main entry point for the Thoth application that handles overall page structure"""
    st.sidebar.title("Thoth")
    article = st.sidebar.selectbox(
        "Choose a Machine Learning method",
        list(HANDLER_REGISTRY.keys()),
        index=0,
    )

    handler_cls = HANDLER_REGISTRY[article]
    handler = handler_cls()
    handler.render_page()


if __name__ == "__main__":
    main()
