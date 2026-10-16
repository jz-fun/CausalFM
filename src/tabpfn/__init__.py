from importlib.metadata import version

# Import estimator wrappers only when requested.  The CausalFM PyTorch core
# does not require the scikit-learn estimator/preprocessing dependencies.

try:
    __version__ = version(__name__)
except ImportError:
    __version__ = "unknown"

__all__ = [
    "TabPFNClassifier",
    "TabPFNRegressor",
    "__version__",
    "display_debug_info",
]


def __getattr__(name):
    if name == "TabPFNClassifier":
        from tabpfn.classifier import TabPFNClassifier

        return TabPFNClassifier
    if name == "TabPFNRegressor":
        from tabpfn.regressor import TabPFNRegressor

        return TabPFNRegressor
    if name == "display_debug_info":
        from tabpfn.misc.debug_versions import display_debug_info

        return display_debug_info
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
