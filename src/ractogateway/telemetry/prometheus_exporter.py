"""PrometheusExporter — HTTP server exposing a ``/metrics`` endpoint.

Requires: ``pip install ractogateway[prometheus]``

Example::

    from ractogateway.telemetry import GatewayMetricsMiddleware, PrometheusExporter

    metrics = GatewayMetricsMiddleware()
    exporter = PrometheusExporter(port=8000)
    exporter.start()

    # Prometheus can now scrape http://localhost:8000/metrics
    # ...

    exporter.stop()
"""

from __future__ import annotations

import threading
from typing import Any


def _require_prometheus() -> Any:
    try:
        import prometheus_client  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "prometheus_client is required for PrometheusExporter. "
            "Install with:  pip install ractogateway[prometheus]"
        ) from exc
    return prometheus_client


class PrometheusExporter:
    """Start an HTTP server that exposes all registered Prometheus metrics.

    The server listens on ``http://0.0.0.0:<port>/metrics`` and responds
    with the standard Prometheus text exposition format.  This is designed
    to be used alongside :class:`~ractogateway.telemetry.GatewayMetricsMiddleware`
    but works with any metrics registered in the global registry.

    Parameters
    ----------
    port:
        HTTP port to listen on.  Defaults to ``8000``.
    registry:
        Custom ``prometheus_client.CollectorRegistry``.  When ``None``
        the global ``REGISTRY`` is used.

    Example::

        from ractogateway.telemetry import GatewayMetricsMiddleware, PrometheusExporter

        metrics = GatewayMetricsMiddleware()
        exporter = PrometheusExporter(port=9090)
        exporter.start()
        # Prometheus can now scrape http://localhost:9090/metrics
        exporter.stop()

    Requires: ``pip install ractogateway[prometheus]``
    """

    def __init__(
        self,
        port: int = 8000,
        *,
        registry: Any | None = None,
    ) -> None:
        self._port = port
        self._registry = registry
        self._server: Any | None = None
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the metrics HTTP server in a background daemon thread.

        Calling ``start()`` a second time on an already-running exporter
        is a no-op.
        """
        prom = _require_prometheus()
        with self._lock:
            if self._server is not None:
                return
            kw: dict[str, Any] = {"port": self._port, "addr": ""}
            if self._registry is not None:
                kw["registry"] = self._registry
            server, _ = prom.start_http_server(**kw)
            self._server = server

    def stop(self) -> None:
        """Shut down the metrics HTTP server.

        Safe to call even if the server was never started.
        """
        with self._lock:
            if self._server is not None:
                self._server.shutdown()
                self._server = None

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def port(self) -> int:
        """The configured HTTP port."""
        return self._port

    @property
    def is_running(self) -> bool:
        """``True`` when the HTTP server is active."""
        with self._lock:
            return self._server is not None

    def generate_latest(self) -> str:
        """Return current metrics in Prometheus text format.

        No HTTP server required — useful for testing or embedding the
        metrics in a custom endpoint::

            text = exporter.generate_latest()
            assert "process_resident_memory_bytes" in text

        Returns
        -------
        str
            UTF-8 decoded Prometheus text exposition string.
        """
        prom = _require_prometheus()
        if self._registry is not None:
            raw: bytes = prom.generate_latest(self._registry)
        else:
            raw = prom.generate_latest()
        return raw.decode()
