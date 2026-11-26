"""Shared scope signal for native likelihood fallback."""


class NativeInputUnsupported(ValueError):
    """The native kernel cannot represent this valid PyTorch input."""
