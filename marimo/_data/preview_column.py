# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

from typing import Any

import narwhals.stable.v2 as nw

from marimo import _loggers
from marimo._data.charts import ChartBuilder, get_chart_builder
from marimo._data.models import ColumnStats, DataType
from marimo._data.sql_summaries import (
    get_column_type,
    get_sql_stats,
)
from marimo._dependencies.dependencies import DependencyManager
from marimo._messaging.notification import (
    ColumnPreview,
    DataColumnPreviewNotification,
)
from marimo._plugins.ui._impl.tables.table_manager import (
    FieldType,
    TableManager,
)
from marimo._plugins.ui._impl.tables.utils import get_table_manager_or_none
from marimo._runtime.commands import PreviewDatasetColumnCommand
from marimo._sql.engines.types import EngineCatalog, QueryEngine
from marimo._sql.sql_quoting import quote_qualified_name, quote_sql_identifier
from marimo._sql.utils import wrapped_sql
from marimo._utils.narwhals_utils import (
    downgrade_narwhals_df_to_v1,
    is_narwhals_string_type,
    is_narwhals_time_type,
)

LOGGER = _loggers.marimo_logger()

CHART_MAX_ROWS = 20_000
VEGAFUSION_ERROR = "Too many rows, vegafusion required to render charts"
VEGAFUSION_MISSING_PACKAGES = ["vegafusion", "vl_convert_python"]
ALTAIR_ERROR = "Altair is required to render charts"
ALTAIR_MISSING_PACKAGES = ["altair"]


def get_table_manager(item: object) -> TableManager[Any] | None:
    try:
        table = get_table_manager_or_none(item)
        return table
    except Exception as e:
        LOGGER.warning(
            "Failed to get table manager for item %s",
            item,
            exc_info=e,
        )
        return None


def get_column_preview_dataset(
    table: TableManager[Any],
    table_name: str,
    column_name: str,
) -> ColumnPreview:
    """
    Get a preview of the column in the dataset.

    This may return a chart and aggregation stats of the column.
    """

    try:
        table_rows = table.get_num_rows(force=True)
        if table_rows == 0:
            return ColumnPreview(
                error="Table is empty",
            )
        try:
            stats = table.get_stats(column_name)
        except BaseException as e:
            # Catch-all: some libraries like Polars have bugs and raise
            # BaseExceptions, which shouldn't crash the kernel
            LOGGER.warning(
                "Failed to get stats for column %s in table %s",
                column_name,
                table_name,
                exc_info=e,
            )
            stats = ColumnStats()

        # We require altair to render the chart
        error = None
        missing_packages = None
        if not DependencyManager.altair.has():
            error, missing_packages = ALTAIR_ERROR, ALTAIR_MISSING_PACKAGES
        else:
            # Check for special characters that can't be escaped easily
            # (e.g. backslash, quotes)
            for char in ["\\", '"', "'"]:
                if char in str(column_name):
                    error = (
                        f"Column names with `{char}` are not supported "
                        "in charts. Consider renaming the column."
                    )
                    break

        # Get the chart for the column
        chart_spec = None
        chart_code = None

        if error is None:
            try:
                (
                    chart_spec,
                    chart_code,
                    error,
                    missing_packages,
                ) = _get_altair_chart(
                    table_name, column_name, table, stats, table_rows
                )
            except Exception as e:
                error = str(e)
                LOGGER.warning(
                    "Failed to get chart for column %s in table %s",
                    column_name,
                    table_name,
                    exc_info=e,
                )
                chart_spec, chart_code = None, None

        return ColumnPreview(
            chart_spec=chart_spec,
            chart_code=chart_code,
            error=error,
            missing_packages=missing_packages,
            stats=stats,
        )

    except Exception as e:
        LOGGER.warning(
            "Failed to get column preview for column %s in table %s",
            column_name,
            table_name,
            exc_info=e,
        )
        return ColumnPreview(error=str(e), missing_packages=None)


def get_column_preview_for_dataframe(
    item: object,
    request: PreviewDatasetColumnCommand,
) -> DataColumnPreviewNotification | None:
    """
    Finds the table manager for the item and gets the column preview.
    """
    column_name = request.column_name
    table_name = request.table_name

    table = get_table_manager(item)
    if table is None:
        return None

    column_preview = get_column_preview_dataset(table, table_name, column_name)
    return DataColumnPreviewNotification(
        table_name=table_name,
        column_name=column_name,
        chart_spec=column_preview.chart_spec,
        chart_code=column_preview.chart_code,
        stats=column_preview.stats,
        error=column_preview.error,
        missing_packages=column_preview.missing_packages,
    )


def get_column_preview_for_duckdb(
    *,
    fully_qualified_table_name: str,
    column_name: str,
) -> DataColumnPreviewNotification | None:
    DependencyManager.duckdb.require(why="previewing DuckDB columns")

    column_type = get_column_type(fully_qualified_table_name, column_name)
    stats = get_sql_stats(fully_qualified_table_name, column_name, column_type)

    # Generate Altair chart
    chart_spec = None
    chart_code = None
    error = None
    missing_packages = None
    should_limit_to_10_items = True

    total_rows = stats.total
    if total_rows is not None and DependencyManager.altair.has():
        try:
            if total_rows <= CHART_MAX_ROWS:
                relation = wrapped_sql(
                    f"SELECT {column_name} FROM {fully_qualified_table_name}",
                    connection=None,
                )
                chart_spec = _get_chart_spec(
                    column_data=relation,
                    column_type=column_type,
                    column_name=column_name,
                    chart_builder=get_chart_builder(
                        column_type, should_limit_to_10_items
                    ),
                )
            else:
                error, missing_packages = (
                    VEGAFUSION_ERROR,
                    VEGAFUSION_MISSING_PACKAGES,
                )
        except Exception as e:
            LOGGER.warning(f"Failed to generate Altair chart: {e!s}")

    return DataColumnPreviewNotification(
        table_name=fully_qualified_table_name,
        column_name=column_name,
        chart_spec=chart_spec,
        chart_code=chart_code,
        stats=stats,
        error=error,
        missing_packages=missing_packages,
    )


def get_column_preview_for_connection(
    *,
    engine: QueryEngine[Any],
    request: PreviewDatasetColumnCommand,
) -> DataColumnPreviewNotification:
    """Get a column preview for a table reached through a user SQL connection.

    Stats are computed with aggregate queries on the connection itself, and
    chart data is pulled with a hard row cap so whole columns are never
    fetched.
    """
    column_name = request.column_name
    database = request.database or ""
    schema = request.schema or ""
    schema_path = request.schema_path or None
    table_name = request.table_name
    locator = request.fully_qualified_table_name or table_name
    dialect = engine.dialect.lower()

    qualified_table = _qualify_connection_table_name(
        engine,
        database=database,
        schema=schema,
        schema_path=schema_path,
        table_name=table_name,
    )
    quoted_column = quote_sql_identifier(column_name, dialect=dialect)

    column_type = _resolve_connection_column_type(
        engine,
        database=database,
        schema=schema,
        schema_path=schema_path,
        table_name=table_name,
        qualified_table=qualified_table,
        column_name=column_name,
        fallback_type=request.column_type,
    )

    def _notify(
        *,
        stats: ColumnStats | None = None,
        chart_spec: str | None = None,
        error: str | None = None,
        missing_packages: list[str] | None = None,
    ) -> DataColumnPreviewNotification:
        return DataColumnPreviewNotification(
            request_id=request.request_id,
            table_name=locator,
            column_name=column_name,
            chart_spec=chart_spec,
            chart_code=None,
            stats=stats,
            error=error,
            missing_packages=missing_packages,
        )

    if column_type is None:
        return _notify(
            error=f"Unable to determine the type of column {column_name}",
        )

    # Aggregates run in the database; this never pulls the column itself.
    stats = _get_connection_sql_stats(
        engine,
        qualified_table=qualified_table,
        quoted_column=quoted_column,
        column_type=column_type,
    )

    chart_spec, error, missing_packages = _get_connection_chart(
        engine,
        qualified_table=qualified_table,
        column_name=column_name,
        column_type=column_type,
        total=stats.total,
        nulls=stats.nulls,
    )

    return _notify(
        stats=stats,
        chart_spec=chart_spec,
        error=error,
        missing_packages=missing_packages,
    )


def _qualify_connection_table_name(
    engine: QueryEngine[Any],
    *,
    database: str,
    schema: str,
    schema_path: list[str] | None,
    table_name: str,
) -> str:
    """Build a dialect-quoted, fully qualified table name for a connection."""
    dialect = engine.dialect.lower()

    if schema_path:
        namespace_parts = [*schema_path]
    else:
        namespace_parts = [schema] if schema else []

    parts: list[str] = []
    if database:
        default_database: str | None = None
        if isinstance(engine, EngineCatalog):
            try:
                default_database = engine.get_default_database()
            except Exception:
                LOGGER.debug(
                    "Failed to get default database for %s engine",
                    dialect,
                    exc_info=True,
                )
        # Omit the catalog when it is the connection's current one: some
        # dialects (e.g. Postgres) reject 3-part names for it.
        if database != default_database:
            parts.append(database)

    parts.extend(part for part in namespace_parts if part)
    parts.append(table_name)
    # Snowflake/StarRocks normalize unquoted identifiers in their catalog, so
    # use the engine's own quoting rules for those dialects.
    if dialect in ("snowflake", "starrocks"):
        engine_quote = getattr(engine, "_quote_identifier", None)
        if callable(engine_quote):
            return ".".join(engine_quote(part) for part in parts)
    return quote_qualified_name(*parts, dialect=dialect)


def _resolve_connection_column_type(
    engine: QueryEngine[Any],
    *,
    database: str,
    schema: str,
    schema_path: list[str] | None,
    table_name: str,
    qualified_table: str,
    column_name: str,
    fallback_type: DataType | None,
) -> DataType | None:
    """Determine the column type using the connection, then the UI hint."""
    if isinstance(engine, EngineCatalog):
        try:
            details = engine.get_table_details(
                table_name=table_name,
                schema_name=schema,
                database_name=database,
                schema_path=schema_path,
            )
        except Exception:
            LOGGER.debug(
                "Catalog failed to resolve column %s",
                column_name,
                exc_info=True,
            )
            details = None
        if details is not None:
            for column in details.columns:
                if column.name == column_name:
                    return column.type

    inferred = _infer_connection_column_type(
        engine, qualified_table, column_name
    )
    if inferred is not None:
        return inferred

    return fallback_type


def _infer_connection_column_type(
    engine: QueryEngine[Any], qualified_table: str, column_name: str
) -> DataType | None:
    """Infer the type from a zero-row SELECT's result schema."""
    dialect = engine.dialect.lower()
    quoted_column = quote_sql_identifier(column_name, dialect=dialect)
    query = _bounded_select_query(
        dialect,
        f"SELECT {quoted_column} FROM {qualified_table}",
        0,
    )
    try:
        result = engine.execute(query)
        frame = nw.from_native(result, pass_through=True)
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        dtype = frame.collect_schema()[column_name]
    except Exception:
        LOGGER.debug(
            "Failed to infer type of column %s", column_name, exc_info=True
        )
        return None

    if is_narwhals_string_type(dtype):
        return "string"
    if dtype == nw.Boolean:
        return "boolean"
    if dtype == nw.Duration:
        return "number"
    if dtype.is_integer():
        return "integer"
    if is_narwhals_time_type(dtype):
        return "time"
    if dtype == nw.Date:
        return "date"
    if dtype == nw.Datetime or dtype.is_temporal():
        return "datetime"
    if dtype.is_numeric():
        return "number"
    return "unknown"


def _execute_first_row(
    engine: QueryEngine[Any], query: str
) -> tuple[Any, ...] | None:
    """Execute a query and return its first row as a plain tuple."""
    result = engine.execute(query)
    if result is None:
        return None

    try:
        frame = nw.from_native(result, pass_through=True)
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        if frame.shape[0] == 0:
            return None
        return tuple(frame.row(0))
    except Exception:
        pass

    fetchone = getattr(result, "fetchone", None)
    if callable(fetchone):
        row = fetchone()
        return tuple(row) if row is not None else None
    return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _get_connection_sql_stats(
    engine: QueryEngine[Any],
    *,
    qualified_table: str,
    quoted_column: str,
    column_type: DataType,
) -> ColumnStats:
    """Run portable aggregate queries against the connection."""
    base_query = f"""
    SELECT
        COUNT(*) as col_count,
        COUNT(DISTINCT {quoted_column}) as col_unique,
        SUM(CASE WHEN {quoted_column} IS NULL THEN 1 ELSE 0 END) as col_nulls
    FROM {qualified_table}
    """
    base_row = _execute_first_row(engine, base_query)
    if base_row is None:
        raise ValueError("Failed to compute column statistics")

    total, unique, null_count = base_row
    stats = ColumnStats(
        total=_as_int(total),
        unique=_as_int(unique),
        nulls=_as_int(null_count),
    )

    if stats.total == 0:
        return stats

    if column_type in ("integer", "number"):
        row = _best_effort_row(
            engine,
            f"""
            SELECT
                MIN({quoted_column}) as col_min,
                MAX({quoted_column}) as col_max,
                AVG({quoted_column}) as col_mean
            FROM {qualified_table}
            """,
        )
        if row is not None:
            stats.min, stats.max, stats.mean = (
                row[0],
                row[1],
                row[2],
            )
    elif column_type in ("date", "datetime", "time"):
        row = _best_effort_row(
            engine,
            f"""
            SELECT
                MIN({quoted_column}) as col_min,
                MAX({quoted_column}) as col_max
            FROM {qualified_table}
            """,
        )
        if row is not None:
            stats.min, stats.max = row[0], row[1]
    elif column_type == "boolean":
        true_expr, false_expr = _boolean_count_expressions(
            quoted_column, engine.dialect.lower()
        )
        row = _best_effort_row(
            engine,
            f"""
            SELECT {true_expr} as col_true, {false_expr} as col_false
            FROM {qualified_table}
            """,
        )
        if row is not None:
            stats.true = _as_int(row[0])
            stats.false = _as_int(row[1])

    return stats


def _best_effort_row(
    engine: QueryEngine[Any], query: str
) -> tuple[Any, ...] | None:
    """Run an optional stats query, degrading silently if the dialect rejects it."""
    try:
        return _execute_first_row(engine, query)
    except Exception:
        LOGGER.warning(
            "Optional statistics query failed for %s engine",
            engine.dialect,
            exc_info=True,
        )
        return None


def _boolean_count_expressions(
    quoted_column: str, dialect: str
) -> tuple[str, str]:
    # T-SQL bit columns can't be used as bare boolean expressions
    if dialect in ("mssql", "sqlserver"):
        return (
            f"SUM(CASE WHEN {quoted_column} = 1 THEN 1 ELSE 0 END)",
            f"SUM(CASE WHEN {quoted_column} = 0 THEN 1 ELSE 0 END)",
        )
    return (
        f"SUM(CASE WHEN {quoted_column} = TRUE THEN 1 ELSE 0 END)",
        f"SUM(CASE WHEN {quoted_column} = FALSE THEN 1 ELSE 0 END)",
    )


def _bounded_select_query(dialect: str, select_query: str, limit: int) -> str:
    """Render a row-limited SELECT for engines without standard LIMIT."""
    dialect = dialect.lower()
    if dialect in ("mssql", "sqlserver"):
        return select_query.replace("SELECT ", f"SELECT TOP {limit} ", 1)
    if dialect in ("oracle", "oracledb", "db2", "db2i"):
        return f"{select_query} FETCH FIRST {limit} ROWS ONLY"
    return f"{select_query} LIMIT {limit}"


def _get_connection_chart(
    engine: QueryEngine[Any],
    *,
    qualified_table: str,
    column_name: str,
    column_type: DataType,
    total: int | None,
    nulls: int | None,
) -> tuple[str | None, str | None, list[str] | None]:
    """Build the Altair chart from a bounded sample read through the connection.

    Returns chart_spec, error, missing_packages.
    """
    # Geometry/unknown columns can't be charted
    if column_type in ("unknown", "geometry"):
        return None, None, None

    if total is None or total == 0:
        return None, "Table is empty", None

    if nulls is not None and nulls == total:
        return None, "Column contains only null values", None

    if total > CHART_MAX_ROWS:
        return None, VEGAFUSION_ERROR, VEGAFUSION_MISSING_PACKAGES

    if not DependencyManager.altair.has():
        return None, ALTAIR_ERROR, ALTAIR_MISSING_PACKAGES

    # Same identifier restrictions as local and DuckDB previews
    for char in ["\\", '"', "'"]:
        if char in str(column_name):
            return (
                None,
                (
                    f"Column names with `{char}` are not supported in charts. "
                    "Consider renaming the column."
                ),
                None,
            )

    dialect = engine.dialect.lower()
    quoted_column = quote_sql_identifier(column_name, dialect=dialect)
    query = _bounded_select_query(
        dialect,
        f"SELECT {quoted_column} FROM {qualified_table}",
        CHART_MAX_ROWS,
    )

    try:
        result = engine.execute(query)
        column_data = nw.from_native(result, pass_through=True)
        if isinstance(column_data, nw.LazyFrame):
            column_data = column_data.collect()
        if column_data.shape[0] == 0:
            return None, "Table is empty", None
        column_data = _sanitize_data(column_data, column_name)
        if isinstance(column_data, nw.LazyFrame):
            column_data = column_data.collect()
        chart_spec = _get_chart_spec(
            column_data=downgrade_narwhals_df_to_v1(column_data),
            column_type=column_type,
            column_name=column_name,
            chart_builder=get_chart_builder(
                column_type, should_limit_to_10_items=True
            ),
        )
        return chart_spec, None, None
    except Exception as e:
        LOGGER.warning(
            "Failed to generate chart for column %s via %s connection",
            column_name,
            dialect,
            exc_info=e,
        )
        return None, None, None


def _get_altair_chart(
    table_name: str,
    column_name: str,
    table: TableManager[Any],
    stats: ColumnStats,
    table_rows: int | None,
) -> tuple[str | None, str | None, str | None, list[str] | None]:
    """
    Get an Altair chart for a column.

    Returns:
        chart_spec, chart_code, error, missing_packages
    """
    # We require altair to render the chart
    if not DependencyManager.altair.has() or not table.supports_altair():
        return None, None, ALTAIR_ERROR, ALTAIR_MISSING_PACKAGES

    from altair import MaxRowsError

    (column_type, _external_type) = table.get_field_type(column_name)

    # Nested/unknown dtypes (e.g. Polars Struct/List) can't be serialized
    # through the dataframe interchange path, so skip charting silently.
    # Geometry columns get no chart either.
    if column_type in ("unknown", "geometry"):
        return None, None, None, None

    if stats.total == 0:
        return None, None, "Table is empty", None

    if (
        table_rows is not None
        and table_rows > CHART_MAX_ROWS
        and not (
            DependencyManager.vegafusion.has()
            and DependencyManager.vl_convert_python.has()
        )
    ):
        # If we don't have vegafusion, we can't render charts for large tables
        return None, None, VEGAFUSION_ERROR, VEGAFUSION_MISSING_PACKAGES

    # For categorical columns with more than 10 unique values,
    # we limit the chart to 10 items
    should_limit_to_10_items = False
    if (
        column_type == "string"
        and stats.unique is not None
        and stats.unique > 10
    ):
        should_limit_to_10_items = True

    chart_builder = get_chart_builder(column_type, should_limit_to_10_items)
    code = chart_builder.altair_code_with_comment(
        table_name, column_name, simple=True
    )

    # Filter the data to the column we want
    column_data = table.select_columns([column_name]).data
    column_data = _sanitize_data(column_data, column_name)
    if isinstance(column_data, nw.LazyFrame):
        column_data = column_data.collect()

    error: str | None = None
    missing_packages: list[str] | None = None

    # We may not know number of rows, so we can check for max rows error
    try:
        chart_spec = _get_chart_spec(
            # Downgrade to v1 since altair doesn't support v2 yet
            # This is validated with our tests, so if the tests pass with this
            # removed, we can remove the downgrade.
            column_data=downgrade_narwhals_df_to_v1(column_data),
            column_type=column_type,
            column_name=column_name,
            chart_builder=chart_builder,
        )
    except MaxRowsError:
        chart_spec = None
        error, missing_packages = VEGAFUSION_ERROR, VEGAFUSION_MISSING_PACKAGES

    return chart_spec, code, error, missing_packages


def _get_chart_spec(
    *,
    column_data: Any,
    column_type: FieldType,
    column_name: str,
    chart_builder: ChartBuilder,
) -> str:
    import altair as alt

    # If we have vegafusion and vl-convert-python, use it
    if (
        DependencyManager.vegafusion.has()
        and DependencyManager.vl_convert_python.has()
    ):
        with alt.data_transformers.enable("vegafusion"):
            return chart_builder.altair_json(
                column_data,
                column_name,
            )

    # Date types don't serialize well to csv,
    # so we don't transform them
    dont_use_csv = (
        column_type == "date"
        or column_type == "datetime"
        or column_type == "time"
    )
    if dont_use_csv:
        # Default max_rows is 5_000, but we can support more.
        with alt.data_transformers.enable("default", max_rows=CHART_MAX_ROWS):
            return chart_builder.altair_json(
                column_data,
                column_name,
            )
    with alt.data_transformers.enable("marimo_inline_csv"):
        return chart_builder.altair_json(
            column_data,
            column_name,
        )


def _sanitize_data(
    column_data: nw.DataFrame[Any] | nw.LazyFrame[Any] | Any, column_name: str
) -> nw.DataFrame[Any] | nw.LazyFrame[Any] | Any:
    """
    Sanitize data for vegafusion.
    Vegafusion doesn't support all data types so we convert them to supported types.
    """
    try:
        frame = column_data.lazy()
        col = nw.col(column_name)
        dtype = column_data.collect_schema()[column_name]

        if dtype == nw.Categorical or dtype == nw.Enum:
            column_data = frame.with_columns(col.cast(nw.String))
        # Int128 and UInt128 are not supported by datafusion
        elif dtype == nw.Int128:
            column_data = frame.with_columns(col.cast(nw.Int64))
        elif dtype == nw.UInt128:
            column_data = frame.with_columns(col.cast(nw.UInt64))
        elif dtype == nw.Duration:
            # Convert Duration to numeric values for better charting support
            try:
                result = (
                    frame.select(
                        col.min().alias("min"), col.max().alias("max")
                    )
                    .collect()
                    .rows(named=True)[0]
                )
                min_value = result["min"]
                max_value = result["max"]
                if min_value is not None and max_value is not None:
                    diff = max_value - min_value
                    total_seconds = diff.total_seconds()
                    if total_seconds >= 604800:
                        # Use weeks if range is at least a week
                        column_data = frame.with_columns(
                            (col.dt.total_seconds() / 604800).alias(
                                column_name
                            )
                        )
                    elif total_seconds >= 86400:
                        # Use days if range is at least a day
                        column_data = frame.with_columns(
                            (col.dt.total_seconds() / 86400).alias(column_name)
                        )
                    elif total_seconds >= 3600:
                        # Use hours if range is at least an hour
                        column_data = frame.with_columns(
                            (col.dt.total_seconds() / 3600).alias(column_name)
                        )
                    elif total_seconds >= 60:
                        # Use minutes if range is at least a minute
                        column_data = frame.with_columns(
                            col.dt.total_minutes().alias(column_name)
                        )
                    elif total_seconds >= 1:
                        # Use seconds if range is at least a second
                        column_data = frame.with_columns(
                            col.dt.total_seconds().alias(column_name)
                        )
                    elif total_seconds >= 0.001:
                        # Use milliseconds if range is at least a millisecond
                        column_data = frame.with_columns(
                            col.dt.total_milliseconds().alias(column_name)
                        )
                    elif total_seconds >= 0.000001:
                        # Use microseconds if range is at least a microsecond
                        column_data = frame.with_columns(
                            col.dt.total_microseconds().alias(column_name)
                        )
                    elif total_seconds >= 0.000000001:
                        # Use nanoseconds if range is at least a nanosecond
                        column_data = frame.with_columns(
                            col.dt.total_nanoseconds().alias(column_name)
                        )
            except Exception as e:
                LOGGER.warning("Failed to infer duration precision: %s", e)
                column_data = frame.with_columns(
                    col.dt.total_seconds().alias(column_name)
                )
    except Exception as e:
        LOGGER.warning(f"Failed to sanitize dtypes: {e}")
    return column_data
