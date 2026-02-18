"""ractogateway.gateway — Unified Gateway Runner.

Orchestrates prompt compilation, adapter selection, tool injection, and
response standardization into a single high-level interface.
"""

from ractogateway.gateway.runner import Gateway

__all__ = ["Gateway"]
