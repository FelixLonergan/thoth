import sys
from typing import TYPE_CHECKING, Any, Optional

# importlib is only available in python 3.8+
if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    metadata: Any = None

if metadata is None:
    __version__: Optional[str] = None
else:
    try:
        __version__ = metadata.version("thoth")
    except metadata.PackageNotFoundError:
        __version__ = None

SEED = 42
