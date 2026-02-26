"""Chart specification and deterministic Plotly figure builder.

No LLM calls — chart type is either specified by the user or inferred from
DataFrame column dtypes using simple heuristics.

Requires: ``pip install ractogateway[pipelines-sql-viz]``
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------


def _require_plotly() -> Any:
    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError(
            "The 'plotly' package is required for chart generation. "
            "Install it with:  pip install ractogateway[pipelines-sql-viz]"
        ) from exc
    return px


# ---------------------------------------------------------------------------
# ChartSpec model
# ---------------------------------------------------------------------------

_VALID_CHART_TYPES = frozenset(
    {"bar", "line", "scatter", "pie", "histogram", "box", "area", "heatmap", "violin", "funnel"}
)


class ChartSpec(BaseModel):
    """Specification for a Plotly chart.

    Pass to ``SQLAnalystPipeline`` as ``chart=ChartSpec(...)`` or as a plain
    ``dict`` (e.g. ``chart={"chart_type": "bar", "x": "customer", "y": "revenue"}``).
    Use ``chart="auto"`` to let the pipeline infer the best chart type from the
    DataFrame's column dtypes with no extra LLM call.

    Supported chart types
    ---------------------
    ``bar`` · ``line`` · ``scatter`` · ``pie`` · ``histogram`` · ``box`` ·
    ``area`` · ``heatmap`` · ``violin`` · ``funnel``

    Example::

        from ractogateway.pipelines import SQLAnalystPipeline, ChartSpec

        result = pipeline.run(
            user_query="Top 5 customers by revenue?",
            ...,
            chart=ChartSpec(chart_type="bar", x="customer_name", y="revenue",
                            title="Top 5 Customers"),
        )
        result.plotly_figure.show()
    """

    chart_type: str
    x: str | None = None
    y: str | list[str] | None = None
    color: str | None = None
    title: str = ""
    x_label: str | None = None
    y_label: str | None = None
    orientation: Literal["v", "h"] | None = None

    @field_validator("chart_type")
    @classmethod
    def _validate_chart_type(cls, v: str) -> str:
        normalized = v.lower()
        if normalized not in _VALID_CHART_TYPES:
            raise ValueError(
                f"chart_type must be one of {sorted(_VALID_CHART_TYPES)}, got {v!r}"
            )
        return normalized


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _labels(spec: ChartSpec) -> dict[str, str]:
    out: dict[str, str] = {}
    if spec.x_label and spec.x:
        out[spec.x] = spec.x_label
    if spec.y_label and isinstance(spec.y, str) and spec.y:
        out[spec.y] = spec.y_label
    return out


def _xy_kw(spec: ChartSpec, kw: dict[str, Any]) -> dict[str, Any]:
    if spec.x:
        kw["x"] = spec.x
    if spec.y:
        kw["y"] = spec.y
    if spec.color:
        kw["color"] = spec.color
    lb = _labels(spec)
    if lb:
        kw["labels"] = lb
    return kw


# ---------------------------------------------------------------------------
# Chart builders — one function per chart type
# ---------------------------------------------------------------------------


def _build_bar(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    _xy_kw(spec, kw)
    if spec.orientation:
        kw["orientation"] = spec.orientation
    return px.bar(df, **kw)


def _build_line(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    _xy_kw(spec, kw)
    return px.line(df, **kw)


def _build_scatter(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    _xy_kw(spec, kw)
    return px.scatter(df, **kw)


def _build_pie(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    if spec.x:
        kw["names"] = spec.x
    if spec.y:
        kw["values"] = spec.y
    return px.pie(df, **kw)


def _build_histogram(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    if spec.x:
        kw["x"] = spec.x
    if spec.color:
        kw["color"] = spec.color
    lb = _labels(spec)
    if lb:
        kw["labels"] = lb
    return px.histogram(df, **kw)


def _build_box(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    _xy_kw(spec, kw)
    return px.box(df, **kw)


def _build_area(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    _xy_kw(spec, kw)
    return px.area(df, **kw)


def _build_heatmap(px: Any, df: Any, spec: ChartSpec) -> Any:
    return px.imshow(df, title=spec.title)


def _build_violin(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    if spec.x:
        kw["x"] = spec.x
    if spec.y:
        kw["y"] = spec.y
    if spec.color:
        kw["color"] = spec.color
    return px.violin(df, **kw)


def _build_funnel(px: Any, df: Any, spec: ChartSpec) -> Any:
    kw: dict[str, Any] = {"title": spec.title}
    if spec.x:
        kw["x"] = spec.x
    if spec.y:
        kw["y"] = spec.y
    return px.funnel(df, **kw)


# ---------------------------------------------------------------------------
# Dispatch map — add new chart types here
# ---------------------------------------------------------------------------

_CHART_MAP: dict[str, Any] = {
    "bar":       _build_bar,
    "line":      _build_line,
    "scatter":   _build_scatter,
    "pie":       _build_pie,
    "histogram": _build_histogram,
    "box":       _build_box,
    "area":      _build_area,
    "heatmap":   _build_heatmap,
    "violin":    _build_violin,
    "funnel":    _build_funnel,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_figure(df: Any, spec: ChartSpec) -> Any:
    """Build a Plotly ``Figure`` from *df* using the deterministic *spec*.

    Parameters
    ----------
    df:
        A pandas ``DataFrame`` to visualise.
    spec:
        A :class:`ChartSpec` describing chart type, axes, title, etc.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ImportError
        If ``plotly`` is not installed.
    ValueError
        If ``spec.chart_type`` is not in :data:`_CHART_MAP`.
    """
    px = _require_plotly()
    builder = _CHART_MAP.get(spec.chart_type)
    if builder is None:
        raise ValueError(
            f"Unknown chart_type: {spec.chart_type!r}. "
            f"Supported: {sorted(_CHART_MAP)}"
        )
    return builder(px, df, spec)


def infer_spec(df: Any, pd_module: Any) -> ChartSpec | None:
    """Infer the best :class:`ChartSpec` from *df* column dtypes — no LLM needed.

    Heuristic decision tree:

    1. datetime + numeric  → **line** chart (trend over time)
    2. categorical + numeric, ≤6 unique values → **pie** chart
    3. categorical + numeric, >6 unique values → **bar** chart
    4. 2+ numeric columns  → **scatter** plot
    5. 1 numeric column    → **histogram**
    6. Otherwise           → ``None``

    Parameters
    ----------
    df:
        A pandas ``DataFrame`` (may be the raw SQL result or the pandas
        analysis output).
    pd_module:
        The pandas module (passed in to avoid re-importing).

    Returns
    -------
    ChartSpec | None
    """
    if df is None:
        return None
    try:
        if not hasattr(df, "dtypes") or df.empty or len(df.columns) == 0:
            return None
    except Exception:  # noqa: BLE001
        return None

    cols: list[str] = list(df.columns)
    dtypes = df.dtypes
    pd = pd_module

    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(dtypes[c])]
    cat_cols = [
        c for c in cols
        if pd.api.types.is_string_dtype(dtypes[c]) or pd.api.types.is_object_dtype(dtypes[c])
    ]
    date_cols = [c for c in cols if pd.api.types.is_datetime64_any_dtype(dtypes[c])]

    # 1. datetime + numeric → line
    if date_cols and numeric_cols:
        return ChartSpec(
            chart_type="line",
            x=date_cols[0],
            y=numeric_cols[0],
            title="Trend over Time",
        )

    # 2 & 3. categorical + numeric → pie (few) or bar (many)
    if cat_cols and numeric_cols:
        try:
            n_unique = int(df[cat_cols[0]].nunique())
        except Exception:  # noqa: BLE001
            n_unique = 999
        if n_unique <= 6:
            return ChartSpec(
                chart_type="pie",
                x=cat_cols[0],
                y=numeric_cols[0],
                title="Distribution",
            )
        return ChartSpec(
            chart_type="bar",
            x=cat_cols[0],
            y=numeric_cols[0],
            title="Comparison",
        )

    # 4. 2+ numerics → scatter
    if len(numeric_cols) >= 2:
        return ChartSpec(
            chart_type="scatter",
            x=numeric_cols[0],
            y=numeric_cols[1],
            title="Correlation",
        )

    # 5. single numeric → histogram
    if len(numeric_cols) == 1:
        return ChartSpec(
            chart_type="histogram",
            x=numeric_cols[0],
            title="Distribution",
        )

    return None
