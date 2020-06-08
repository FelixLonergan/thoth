# Thoth 𓅝

Thoth is designed to be an interactive explanation of a number of common Machine Learning methods. Built upon [Streamlit](https://www.streamlit.io/), Thoth offers an intuitive way to understand and experiment with fundamental AI tools and methods.

## Installation

To get started with Thoth, first clone the repository using

```bash
git clone https://github.com/FelixLonergan/Thoth.git
```

Package management for this project is handled using the excellent [Poetry](https://python-poetry.org/) library which can be installed with the following command.

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
```

Once you have the repository cloned and poetry installed you will need to install the dependencies of Thoth. You can do this by navigating to the Thoth directory and running

```bash
poetry install
```

By default poetry will create a new virtual environment and install all the required dependencies there. If you are already inside a virtual environment when you run `poetry install` the dependencies will be installed there (see the [Poetry documentation](https://python-poetry.org/docs/) for more information on how Poetry handles virtual environments).

Once you are inside a virtual environment with all the appropriate dependencies installed, simply run the following command from within the repository directory and Thoth should open in a new browser tab.

```bash
streamlit run run.py
```
