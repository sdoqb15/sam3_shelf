"""Method plugins define trainable modules and losses, not experiment pipelines."""

from .registry import get_method, register_method

__all__ = ["get_method", "register_method"]
