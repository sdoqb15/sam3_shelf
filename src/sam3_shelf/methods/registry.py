"""Small registry shared by reference methods and user-defined method plugins."""

import importlib

_METHODS = {}


def register_method(name):
    def decorate(method):
        if name in _METHODS:
            raise ValueError(f"Method already registered: {name}")
        _METHODS[name] = method
        return method

    return decorate


def get_method(config):
    from . import joint  # noqa: F401 - register built-in methods

    if config.get("method_plugin"):
        importlib.import_module(config["method_plugin"])
    name = config["method"]
    if name not in _METHODS:
        raise ValueError(f"Unknown method {name!r}. Registered: {sorted(_METHODS)}")
    return _METHODS[name]()
