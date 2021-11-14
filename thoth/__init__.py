import pkg_resources as _pkg_resources

try:
    _dist = _pkg_resources.get_distribution("thoth")
except _pkg_resources.DistributionNotFound:
    __version__ = None
else:
    __version__ = _dist.version

SEED = 42
