from importlib.metadata import PackageNotFoundError, version
from typing import Optional

try:
    __version__: Optional[str] = version("thoth")
except PackageNotFoundError:
    __version__ = None

SEED = 42
