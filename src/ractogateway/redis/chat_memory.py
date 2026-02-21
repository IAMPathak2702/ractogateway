"""Sliding-window conversation memory stored in Redis.

Each conversation is kept as a Redis List of JSON-encoded ``{"role", "content"}``
message dicts.  The list is capped at ``max_turns * 2`` entries (one user + one
assistant message per turn) using ``LTRIM`` after every ``append()``.

Why Redis Lists?
----------------
Redis Lists support O(1) push and O(n) range reads — perfect for maintaining a
bounded conversation history that must be accessible to multiple server replicas.
Unlike in-memory approaches, the history survives rolling deployments and can be
shared between a web server and a background worker.

Compatibility with ``ChatConfig.history``
-----------------------------------------
:meth:`get_history` returns ``list[dict[str, str]]`` with ``"role"`` and
``"content"`` keys.  This is the exact format used by all three provider
adapters under the hood.  You can pass the result directly to
``ChatConfig(history=memory.get_history(conv_id))`` after wrapping each dict in
your ``Message`` model.

Example::

    from ractogateway.redis import RedisChatMemory, ChatMemoryConfig

    memory = RedisChatMemory(
        url="redis://localhost:6379/0",
        config=ChatMemoryConfig(max_turns=20, ttl_seconds=1800),
    )

    # Store messages as the conversation progresses:
    memory.append("conv_abc", "user", "What is the capital of France?")
    memory.append("conv_abc", "assistant", "Paris.")

    # Retrieve history to pass back into the kit:
    history = memory.get_history("conv_abc")
    # → [{"role": "user", "content": "What is the capital of France?"},
    #    {"role": "assistant", "content": "Paris."}]
"""

from __future__ import annotations

import json
from typing import Any

from ractogateway.redis._models import ChatMemoryConfig


def _require_redis() -> Any:
    try:
        import redis as redis_lib
    except ImportError as exc:
        raise ImportError(
            "The 'redis' package is required for RedisChatMemory. "
            "Install it with:  pip install ractogateway[redis]"
        ) from exc
    return redis_lib


class RedisChatMemory:
    """Shared, bounded conversation history backed by Redis.

    Parameters
    ----------
    url:
        Redis connection URL.  Ignored when *client* is provided.
    client:
        Pre-built ``redis.Redis`` client.
    config:
        :class:`~ractogateway.redis.ChatMemoryConfig` controlling turn limit,
        TTL, and key namespace.  Defaults are applied when ``None``.
    """

    def __init__(
        self,
        *,
        url: str = "redis://localhost:6379/0",
        client: Any | None = None,
        config: ChatMemoryConfig | None = None,
    ) -> None:
        self._url = url
        self._provided_client = client
        self._config = config or ChatMemoryConfig()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _client(self) -> Any:
        if self._provided_client is not None:
            return self._provided_client
        return _require_redis().from_url(self._url, decode_responses=True)

    def _key(self, conversation_id: str) -> str:
        return f"{self._config.key_prefix}:{conversation_id}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, conversation_id: str, role: str, content: str) -> None:
        """Append a message to the conversation history.

        After appending, the list is trimmed to the last
        ``config.max_turns * 2`` messages (oldest dropped first).  If a TTL is
        configured, it is refreshed on every append so the window slides with
        activity.

        Parameters
        ----------
        conversation_id:
            Opaque identifier for the conversation (e.g. session UUID).
        role:
            The message author: ``"user"``, ``"assistant"``, or ``"system"``.
        content:
            Text content of the message.
        """
        key = self._key(conversation_id)
        payload = json.dumps({"role": role, "content": content})
        max_messages = self._config.max_turns * 2
        cli = self._client()

        with cli.pipeline(transaction=True) as pipe:
            pipe.rpush(key, payload)
            # Keep only the most-recent max_messages entries.
            pipe.ltrim(key, -max_messages, -1)
            if self._config.ttl_seconds is not None:
                pipe.expire(key, int(self._config.ttl_seconds))
            pipe.execute()

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        """Return all stored messages as a list of ``{"role", "content"}`` dicts.

        The list is ordered oldest-first, matching the ``ChatConfig.history``
        convention.

        Returns an empty list when the conversation does not exist or has expired.
        """
        key = self._key(conversation_id)
        raw_messages: list[str] = self._client().lrange(key, 0, -1)
        result: list[dict[str, str]] = []
        for raw in raw_messages:
            try:
                msg: dict[str, str] = json.loads(raw)
                result.append(msg)
            except (json.JSONDecodeError, TypeError):
                # Skip corrupted entries rather than crashing.
                continue
        return result

    def clear(self, conversation_id: str) -> None:
        """Delete the conversation history from Redis."""
        self._client().delete(self._key(conversation_id))

    def count(self, conversation_id: str) -> int:
        """Return the number of messages stored for this conversation.

        Returns ``0`` when the conversation does not exist.
        """
        return int(self._client().llen(self._key(conversation_id)))

    def __repr__(self) -> str:  # pragma: no cover
        cfg = self._config
        return (
            f"RedisChatMemory(max_turns={cfg.max_turns}, "
            f"ttl={cfg.ttl_seconds}s, "
            f"prefix={cfg.key_prefix!r})"
        )
