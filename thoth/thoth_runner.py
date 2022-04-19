"""This is the script that is run by ``streamlit run``

It simply runs the ``thoth`` module as ``__main__``
"""

import runpy

runpy.run_module("thoth", run_name="__main__", alter_sys=True)
