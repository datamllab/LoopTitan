"""Research code for training and scaling looped sparse models.

`LoopedLM` is the public name of the model class; `RecursiveLM` is the same
class object, kept so existing scripts keep working (no warning).

Model dependencies are optional: importing the package/control tools does not
import PyTorch, TorchTitan, Transformers or tracking services.
"""

from importlib import import_module

__all__ = ["ModelConfig", "ModelOutput", "LoopedLM", "RecursiveLM", "DecodeCache"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".cache" if name == "DecodeCache" else ".model", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
