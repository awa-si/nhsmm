"""Shared scope signal for optional research-only native likelihoods."""


class NativeInputUnsupported(ValueError):
    """The native kernel cannot represent this valid PyTorch input."""
