import subprocess
import sys
from pathlib import Path

import click


@click.command()
def thoth() -> None:
    """Runs the Thoth application"""
    run_path = Path(__file__).parent.joinpath("thoth_runner.py")
    args = [sys.executable, "-m", "streamlit", "run", str(run_path)]
    subprocess.run(args, cwd=Path(__file__).parent, check=True)
