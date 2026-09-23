"""Backends: executors of :class:`~probvenance.plans.InferencePlan` objects.

Only the backend :class:`~probvenance.backends.base.Backend` protocol lives here.
Concrete backends are optional: ``probvenance.backends.transformers`` requires the
``transformers`` extra and is deliberately NOT imported by this package.
"""

from probvenance.backends.base import Backend

__all__ = ["Backend"]
