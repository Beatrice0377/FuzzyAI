"""Backends: executors of :class:`~fuzzyai.plans.InferencePlan` objects.

Only the backend :class:`~fuzzyai.backends.base.Backend` protocol lives here.
Concrete backends are optional: ``fuzzyai.backends.transformers`` requires the
``transformers`` extra and is deliberately NOT imported by this package.
"""

from fuzzyai.backends.base import Backend

__all__ = ["Backend"]
