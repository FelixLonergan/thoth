from pathlib import Path

import streamlit as st
from PIL import Image

from .handler import HANDLER_REGISTRY


def main() -> None:
    """Main entry point for the Thoth application that handles overall page structure"""
    favicon_path = Path(__file__).parent.joinpath("static", "favicon.ico")
    with Image.open(favicon_path) as img:
        st.set_page_config(
            page_title="Thoth",
            page_icon=img,
            menu_items={
                "Report a bug": "https://github.com/FelixLonergan/thoth/issues",
            },
        )

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
