"""Read-only SQL guard — blocks write operations before execution."""

from __future__ import annotations

import re

from ractogateway.pipelines.sql_analyst._models import ReadOnlyViolationError

# Matches any SQL keyword that performs a write or schema-altering operation.
_WRITE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|EXEC(?:UTE)?|"
    r"MERGE|CALL|GRANT|REVOKE|REPLACE|UPSERT)\b",
    re.IGNORECASE,
)


class ReadOnlySQLGuard:
    """Validates that a SQL string contains only read (SELECT) operations.

    Uses a keyword regex to detect any DML/DDL statement that would modify
    data or schema.  Call :meth:`check` before executing any LLM-generated
    SQL when ``force_read_only=True``.

    Example::

        ReadOnlySQLGuard.check("SELECT * FROM users")   # passes silently
        ReadOnlySQLGuard.check("DROP TABLE users")       # raises ReadOnlyViolationError
    """

    @staticmethod
    def check(sql: str) -> None:
        """Raise :exc:`ReadOnlyViolationError` if *sql* contains a write operation.

        Parameters
        ----------
        sql:
            The SQL string to validate.

        Raises
        ------
        ReadOnlyViolationError
            If the SQL contains a disallowed keyword.
        """
        match = _WRITE_PATTERN.search(sql)
        if match:
            raise ReadOnlyViolationError(
                f"Generated SQL contains a disallowed write operation: "
                f"'{match.group().upper()}'. Only SELECT queries are permitted "
                f"when force_read_only=True."
            )
