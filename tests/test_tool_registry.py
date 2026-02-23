from __future__ import annotations

import pytest

from ractogateway.tools.registry import ToolRegistry, tool


def test_tool_decorator_registers_into_positional_registry() -> None:
    registry = ToolRegistry()

    @tool(registry)
    def get_weather(city: str) -> str:
        """Return the current weather for a city."""
        return f"Sunny in {city}"

    assert "get_weather" in registry
    assert registry.get_callable("get_weather") is get_weather
    schema = registry.get_schema("get_weather")
    assert schema is not None
    assert schema.name == "get_weather"
    assert schema.description == "Return the current weather for a city."
    assert [p.name for p in schema.parameters] == ["city"]
    assert list(registry.tools.keys()) == ["get_weather"]


def test_tool_decorator_registers_into_keyword_registry() -> None:
    registry = ToolRegistry()

    @tool(registry=registry)
    def get_time(timezone: str) -> str:
        """Return current time for a timezone."""
        return timezone

    assert "get_time" in registry
    assert registry.get_callable("get_time") is get_time


def test_tool_decorator_with_registry_supports_name_description_overrides() -> None:
    registry = ToolRegistry()

    @tool(registry, name="weather_now", description="Weather lookup")
    def weather(city: str) -> str:
        """Internal weather helper."""
        return city

    assert "weather" not in registry
    assert "weather_now" in registry
    schema = registry.get_schema("weather_now")
    assert schema is not None
    assert schema.name == "weather_now"
    assert schema.description == "Weather lookup"
    assert registry.get_callable("weather_now") is weather


def test_tool_rejects_invalid_first_argument() -> None:
    with pytest.raises(TypeError, match="callable or ToolRegistry"):
        tool(123)


def test_tool_rejects_conflicting_positional_and_keyword_registry() -> None:
    registry_a = ToolRegistry()
    registry_b = ToolRegistry()

    with pytest.raises(TypeError, match="different registries"):
        tool(registry_a, registry=registry_b)
