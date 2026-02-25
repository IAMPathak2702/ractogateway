"""OllamaServerManager — start and stop an Ollama server from Python.

Useful when you want to control the Ollama process lifecycle (custom port,
programmatic startup/shutdown) directly from your application code.

Usage::

    from ractogateway import ollama_developer_kit as local

    with local.OllamaServerManager(port=11500) as srv:
        kit = local.Chat(model="llama3.2", base_url=srv.base_url)
        response = kit.chat(local.ChatConfig(user_message="Hello!"))
        print(response.content)

Or manually::

    srv = OllamaServerManager(port=11500)
    srv.start()
    ...
    srv.stop()
"""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from typing import Any


class OllamaServerManager:
    """Manage the lifecycle of an Ollama server subprocess.

    The server is started with the ``OLLAMA_HOST`` environment variable set to
    ``{host}:{port}``, which makes Ollama listen on the requested address.

    Parameters
    ----------
    host:
        Bind address.  Defaults to ``"127.0.0.1"`` (localhost only).
    port:
        TCP port for the Ollama REST API.  Defaults to ``11434`` (the standard
        Ollama port).  Change this to run multiple Ollama instances or avoid
        conflicts with an already-running server.
    startup_timeout:
        Seconds to wait for the server to become ready after starting the
        subprocess.  Raises :class:`TimeoutError` if the server doesn't
        respond within this window.
    ollama_bin:
        Path to the ``ollama`` executable.  Defaults to ``"ollama"`` (looked
        up via PATH).

    Attributes
    ----------
    base_url : str
        The full ``http://{host}:{port}`` URL of the managed server.  Use this
        to construct a :class:`~ractogateway.ollama_developer_kit.kit.OllamaDeveloperKit`::

            kit = local.Chat(model="llama3.2", base_url=srv.base_url)

    Examples
    --------
    **Context manager** (recommended — guarantees cleanup)::

        with OllamaServerManager(port=11500) as srv:
            kit = local.Chat(model="llama3.2", base_url=srv.base_url)
            print(kit.chat(local.ChatConfig(user_message="Hi")).content)

    **Manual start / stop**::

        srv = OllamaServerManager(port=11500)
        srv.start()
        try:
            ...
        finally:
            srv.stop()
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 11434,
        startup_timeout: float = 30.0,
        ollama_bin: str = "ollama",
    ) -> None:
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.ollama_bin = ollama_bin
        self._proc: subprocess.Popen[bytes] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        """Return ``http://{host}:{port}``."""
        return f"http://{self.host}:{self.port}"

    @property
    def is_running(self) -> bool:
        """``True`` when the subprocess is alive."""
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> "OllamaServerManager":
        """Start the Ollama server subprocess.

        Returns *self* so that the call can be chained::

            srv = OllamaServerManager(port=11500).start()

        Raises
        ------
        RuntimeError
            If the server is already running.
        FileNotFoundError
            If the ``ollama`` binary cannot be found.
        TimeoutError
            If the server does not become ready within *startup_timeout* seconds.
        """
        if self.is_running:
            raise RuntimeError(
                f"OllamaServerManager is already running on {self.base_url}."
            )

        env = {**os.environ, "OLLAMA_HOST": f"{self.host}:{self.port}"}
        self._proc = subprocess.Popen(
            [self.ollama_bin, "serve"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(self._atexit_cleanup)
        self._wait_until_ready()
        return self

    def stop(self) -> None:
        """Stop the Ollama server subprocess gracefully.

        Sends SIGTERM first; if the process doesn't exit within 5 seconds,
        SIGKILL is used.  Silently does nothing if the server is not running.
        """
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        self._proc = None

    def pull(self, model: str) -> None:
        """Pull *model* from the Ollama library into the running server.

        Equivalent to running ``ollama pull <model>`` in a shell, but
        scoped to the server managed by this instance via ``OLLAMA_HOST``.

        Parameters
        ----------
        model:
            Model name, e.g. ``"llama3.2"``, ``"nomic-embed-text"``.

        Raises
        ------
        RuntimeError
            If the server is not running.
        subprocess.CalledProcessError
            If ``ollama pull`` exits with a non-zero status.
        """
        if not self.is_running:
            raise RuntimeError(
                "OllamaServerManager is not running.  Call .start() first."
            )
        env = {**os.environ, "OLLAMA_HOST": f"{self.host}:{self.port}"}
        subprocess.run(
            [self.ollama_bin, "pull", model],
            env=env,
            check=True,
        )

    def list_models(self) -> list[str]:
        """Return the names of locally-available models on this server.

        Uses a lightweight HTTP request to the ``/api/tags`` endpoint instead
        of a subprocess so it works even when ``ollama`` CLI is not on PATH.

        Returns
        -------
        list[str]
            Model names (e.g. ``["llama3.2:latest", "nomic-embed-text:latest"]``).
        """
        import json
        import urllib.request

        url = f"{self.base_url}/api/tags"
        with urllib.request.urlopen(url, timeout=5) as resp:  # noqa: S310
            data: Any = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "OllamaServerManager":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wait_until_ready(self) -> None:
        """Poll ``/api/tags`` until the server responds or the timeout expires."""
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + self.startup_timeout
        url = f"{self.base_url}/api/tags"
        last_exc: Exception | None = None

        while time.monotonic() < deadline:
            # Check whether the subprocess died unexpectedly
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"Ollama subprocess exited with code {self._proc.returncode} "
                    "before becoming ready.  Is the 'ollama' binary installed?"
                )
            try:
                with urllib.request.urlopen(url, timeout=1) as _resp:  # noqa: S310
                    return  # server is ready
            except Exception as exc:
                last_exc = exc
                time.sleep(0.25)

        self.stop()
        raise TimeoutError(
            f"Ollama server at {self.base_url} did not become ready within "
            f"{self.startup_timeout}s.  Last error: {last_exc}"
        )

    def _atexit_cleanup(self) -> None:
        """Registered with :func:`atexit` to ensure the subprocess is stopped."""
        self.stop()

    def __repr__(self) -> str:
        status = "running" if self.is_running else "stopped"
        return f"OllamaServerManager(base_url={self.base_url!r}, status={status!r})"
