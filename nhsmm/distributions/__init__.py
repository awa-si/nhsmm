from . import default as _default

# Keep functional calls in the distribution module bound to its canonical nnF import.
_default.F = _default.nnF

from .default import Initial, Duration, Transition, Emission

__all__ = [
    'Initial',
    'Emission',
    'Duration',
    'Transition',
]
