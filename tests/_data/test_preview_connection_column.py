# Copyright 2026 Marimo. All rights reserved.
from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest

from marimo._data.models import (
    DataTable,
    DataTableColumn,
    DataType,
)
from marimo._data.preview_column import (
    CHART_MAX_ROWS,
    _bounded_select_query,
    _qualify_connection_table_name,
    get_column_preview_for_connection,
)
from marimo._dependencies.dependencies import DependencyManager
from marimo._runtime.commands import PreviewDatasetColumnCommand
from marimo._sql.engines.duckdb import DuckDBEngine
from marimo._sql.engines.sqlalchemy import SQLAlchemyEngine
from marimo._sql.engines.types import (
    EngineCatalog,
    InferenceConfig,
    QueryEngine,
)
from marimo._types.ids import VariableName
from tests.utils import assert_serialize_roundtrip

HAS_SQLALCHEMY = DependencyManager.sqlalchemy.has()
HAS_DUCKDB = DependencyManager.duckdb.has()
HAS_POLARS = DependencyManager.polars.has()
HAS_ALTAIR = DependencyManager.altair.has()


def _request(
    *,
    column_name: str,
    column_type: DataType | None = None,
    database: str = "",
    schema: str = "main",
    schema_path: list[str] | None = None,
    table_name: str = "tbl",
    engine: str = "my_conn",
    request_id: str | None = "req-1",
) -> PreviewDatasetColumnCommand:
    locator_parts = [
        *(schema_path if schema_path else ([schema] if schema else [])),
        table_name,
    ]
    locator = (
        ".".join([database, *locator_parts])
        if database
        else ".".join(locator_parts)
    )
    return PreviewDatasetColumnCommand(
        request_id=cast("Any", request_id),
        source_type="connection",
        source=engine,
        table_name=table_name,
        column_name=column_name,
        fully_qualified_table_name=locator,
        engine=engine,
        database=database,
        schema=schema,
        schema_path=schema_path,
        column_type=column_type,
    )


class FakeCatalogEngine(EngineCatalog[Any], QueryEngine[Any]):
    """Engine whose catalog returns scripted table details."""

    def __init__(
        self,
        *,
        dialect: str = "fake",
        default_database: str | None = None,
        details: DataTable | None = None,
    ) -> None:
        super().__init__(connection=object(), engine_name=None)
        self._dialect = dialect
        self._default_database = default_database
        self._details = details

    @property
    def source(self) -> str:
        return "fake"

    @property
    def dialect(self) -> str:
        return self._dialect

    @staticmethod
    def is_compatible(_var: Any) -> bool:
        return False

    @property
    def inference_config(self) -> InferenceConfig:
        return InferenceConfig(
            auto_discover_schemas=False,
            auto_discover_tables=False,
            auto_discover_columns=False,
        )

    def get_default_database(self) -> str | None:
        return self._default_database

    def get_default_schema(self) -> str | None:
        return None

    def get_databases(self, **_kwargs: Any) -> Any:
        raise NotImplementedError

    def get_schemas(self, **_kwargs: Any) -> Any:
        raise NotImplementedError

    def get_tables_in_schema(self, **_kwargs: Any) -> Any:
        raise NotImplementedError

    def get_table_details(self, **_kwargs: Any) -> DataTable | None:
        return self._details

    def execute(self, query: str) -> Any:
        raise NotImplementedError


def _catalog_table(*columns: tuple[str, DataType]) -> DataTable:
    return DataTable(
        source_type="connection",
        source="fake",
        name="tbl",
        num_rows=None,
        num_columns=len(columns),
        variable_name=None,
        columns=[
            DataTableColumn(
                name=name,
                type=col_type,
                external_type=col_type,
                sample_values=[],
            )
            for name, col_type in columns
        ],
        primary_keys=[],
        indexes=[],
    )


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def _polars_frame(total: int, unique: int, nulls: int):
    import polars as pl

    return pl.DataFrame(
        {"count": [total], "unique": [unique], "null_count": [nulls]}
    )


# --------------------------------------------------------------------------- #
# Qualified table names                                                        #
# --------------------------------------------------------------------------- #


def test_qualify_omits_default_postgres_database() -> None:
    engine = FakeCatalogEngine(dialect="postgresql", default_database="app")
    qualified = _qualify_connection_table_name(
        engine,
        database="app",
        schema="public",
        schema_path=None,
        table_name="users",
    )
    assert qualified == '"public"."users"'


def test_qualify_includes_other_postgres_database() -> None:
    engine = FakeCatalogEngine(dialect="postgresql", default_database="app")
    qualified = _qualify_connection_table_name(
        engine,
        database="analytics",
        schema="public",
        schema_path=None,
        table_name="users",
    )
    assert qualified == '"analytics"."public"."users"'


def test_qualify_schema_is_mysql_database() -> None:
    # MySQL exposes sibling databases as schemas; the top-level database
    # node is always the current database, so only the schema is qualified
    engine = FakeCatalogEngine(dialect="mysql", default_database="app")
    qualified = _qualify_connection_table_name(
        engine,
        database="app",
        schema="logs",
        schema_path=None,
        table_name="events",
    )
    assert qualified == "`logs`.`events`"


def test_qualify_duckdb_attached_database() -> None:
    engine = FakeCatalogEngine(dialect="duckdb", default_database="memory")
    current = _qualify_connection_table_name(
        engine,
        database="memory",
        schema="main",
        schema_path=None,
        table_name="tbl",
    )
    attached = _qualify_connection_table_name(
        engine,
        database="lake",
        schema="main",
        schema_path=None,
        table_name="tbl",
    )
    assert current == '"main"."tbl"'
    assert attached == '"lake"."main"."tbl"'


def test_qualify_schemaless_connection() -> None:
    engine = FakeCatalogEngine(
        dialect="clickhouse", default_database="default"
    )
    same_db = _qualify_connection_table_name(
        engine,
        database="default",
        schema="",
        schema_path=None,
        table_name="events",
    )
    other_db = _qualify_connection_table_name(
        engine,
        database="warehouse",
        schema="",
        schema_path=None,
        table_name="events",
    )
    assert same_db == "`events`"
    assert other_db == "`warehouse`.`events`"


def test_qualify_nested_schema_path() -> None:
    engine = FakeCatalogEngine(dialect="trino", default_database="cat")
    qualified = _qualify_connection_table_name(
        engine,
        database="cat",
        schema="deep",
        schema_path=["ns", "deep"],
        table_name="tbl",
    )
    assert qualified == '"ns"."deep"."tbl"'


def test_qualify_snowflake_uses_engine_identifier_rules() -> None:
    engine = FakeCatalogEngine(
        dialect="snowflake", default_database="mydb"
    )
    # Mirror SQLAlchemyEngine._quote_identifier: unquoted names are raw
    engine._quote_identifier = lambda identifier: identifier  # type: ignore[attr-defined]

    qualified = _qualify_connection_table_name(
        engine,
        database="other_db",
        schema="public",
        schema_path=None,
        table_name="users",
    )
    assert qualified == "other_db.public.users"


def test_bounded_select_query_dialects() -> None:
    base = "SELECT x FROM t"
    assert _bounded_select_query("postgres", base, 100).endswith("LIMIT 100")
    assert _bounded_select_query("duckdb", base, 100).endswith("LIMIT 100")
    assert (
        _bounded_select_query("mssql", base, 100) == "SELECT TOP 100 x FROM t"
    )
    assert (
        _bounded_select_query("oracle", base, 100)
        == "SELECT x FROM t FETCH FIRST 100 ROWS ONLY"
    )


# --------------------------------------------------------------------------- #
# End-to-end previews against real engines                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_connection_preview_sqlite_stats() -> None:
    import sqlalchemy as sa

    sa_engine = sa.create_engine("sqlite:///:memory:")
    with sa_engine.begin() as conn:
        conn.execute(
            sa.text("CREATE TABLE tbl (id INTEGER, name TEXT, score REAL)")
        )
        conn.execute(
            sa.text(
                "INSERT INTO tbl VALUES (1, 'a', 1.0), (2, 'a', NULL), "
                "(3, NULL, 3.0)"
            )
        )

    engine = SQLAlchemyEngine(sa_engine, cast(VariableName, "my_conn"))

    result = get_column_preview_for_connection(
        engine=engine, request=_request(column_name="id")
    )
    assert result.request_id == "req-1"
    assert result.table_name == "main.tbl"
    assert result.error is None
    assert result.stats is not None
    assert result.stats.total == 3
    assert result.stats.unique == 3
    assert result.stats.nulls == 0
    assert result.stats.min == 1
    assert result.stats.max == 3
    assert result.stats.mean == 2.0
    assert_serialize_roundtrip(result)

    string_result = get_column_preview_for_connection(
        engine=engine, request=_request(column_name="name")
    )
    assert string_result.stats is not None
    assert string_result.stats.total == 3
    assert string_result.stats.unique == 1
    assert string_result.stats.nulls == 1
    # Only count/unique/nulls are portable for strings
    assert string_result.stats.mean is None


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
@pytest.mark.skipif(not HAS_ALTAIR, reason="altair not installed")
def test_connection_preview_sqlite_chart() -> None:
    from marimo._plugins.ui._impl.charts.altair_transformer import (
        register_transformers,
    )

    register_transformers()

    import sqlalchemy as sa

    sa_engine = sa.create_engine("sqlite:///:memory:")
    with sa_engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE tbl (id INTEGER)"))
        conn.execute(sa.text("INSERT INTO tbl VALUES (1), (2), (3), (4), (5)"))

    engine = SQLAlchemyEngine(sa_engine, cast(VariableName, "my_conn"))
    result = get_column_preview_for_connection(
        engine=engine, request=_request(column_name="id")
    )
    assert result.error is None
    assert result.chart_spec is not None
    # SQL connections never return copy-paste dataframe chart code
    assert result.chart_code is None


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_connection_preview_empty_table() -> None:
    import sqlalchemy as sa

    sa_engine = sa.create_engine("sqlite:///:memory:")
    with sa_engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE tbl (id INTEGER)"))

    engine = SQLAlchemyEngine(sa_engine, cast(VariableName, "my_conn"))
    result = get_column_preview_for_connection(
        engine=engine, request=_request(column_name="id")
    )
    assert result.stats is not None
    assert result.stats.total == 0
    assert result.error == "Table is empty"
    assert result.chart_spec is None


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_connection_preview_all_null_column() -> None:
    import sqlalchemy as sa

    sa_engine = sa.create_engine("sqlite:///:memory:")
    with sa_engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE tbl (id INTEGER)"))
        conn.execute(sa.text("INSERT INTO tbl VALUES (NULL), (NULL), (NULL)"))

    engine = SQLAlchemyEngine(sa_engine, cast(VariableName, "my_conn"))
    result = get_column_preview_for_connection(
        engine=engine, request=_request(column_name="id")
    )
    assert result.stats is not None
    assert result.stats.total == 3
    assert result.stats.nulls == 3
    assert result.error == "Column contains only null values"
    assert result.chart_spec is None


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_connection_preview_failed_query() -> None:
    import sqlalchemy as sa

    sa_engine = sa.create_engine("sqlite:///:memory:")
    with sa_engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE other (id INTEGER)"))

    engine = SQLAlchemyEngine(sa_engine, cast(VariableName, "my_conn"))
    from sqlalchemy.exc import OperationalError

    with pytest.raises(OperationalError, match="no such table"):
        get_column_preview_for_connection(
            engine=engine,
            request=_request(column_name="id", column_type="integer"),
        )


@pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb not installed")
@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_connection_preview_duckdb_connection() -> None:
    """User DuckDB connections expose tables as source_type=connection."""
    import duckdb

    connection = duckdb.connect()
    try:
        connection.execute(
            "CREATE TABLE tbl AS SELECT range AS id FROM range(100)"
        )
        engine = DuckDBEngine(connection, cast(VariableName, "duck_conn"))

        # DuckDBEngine.get_table_details is unimplemented, so the type must
        # be inferred from a zero-row query
        result = get_column_preview_for_connection(
            engine=engine,
            request=_request(
                column_name="id", database="memory", schema="main"
            ),
        )
        assert result.error is None
        assert result.stats is not None
        assert result.stats.total == 100
        assert result.stats.unique == 100
        assert result.stats.nulls == 0
        assert result.stats.min == 0
        assert result.stats.max == 99
        assert result.table_name == "memory.main.tbl"
        assert_serialize_roundtrip(result)
    finally:
        connection.close()


# --------------------------------------------------------------------------- #
# Degraded states with scripted engines                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_connection_preview_large_table_degrades() -> None:
    def execute(query: str) -> Any:
        assert "LIMIT" not in query
        return _polars_frame(CHART_MAX_ROWS + 1, CHART_MAX_ROWS + 1, 0)

    engine = FakeCatalogEngine(
        dialect="postgres",
        default_database="app",
        details=_catalog_table(("id", "integer")),
    )
    # Scripted execute lives on the QueryEngine mixin
    engine.execute = execute  # type: ignore[method-assign]

    result = get_column_preview_for_connection(
        engine=engine,
        request=_request(column_name="id", database="app"),
    )
    assert result.stats is not None
    assert result.stats.total == CHART_MAX_ROWS + 1
    assert (
        result.error == "Too many rows, vegafusion required to render charts"
    )
    assert result.missing_packages == ["vegafusion", "vl_convert_python"]
    assert result.chart_spec is None


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_connection_preview_geometry_skips_chart() -> None:
    engine = FakeCatalogEngine(
        dialect="postgres",
        default_database="app",
        details=_catalog_table(("geom", "geometry")),
    )

    def execute(_query: str) -> Any:
        return _polars_frame(3, 3, 0)

    engine.execute = execute  # type: ignore[method-assign]

    result = get_column_preview_for_connection(
        engine=engine,
        request=_request(column_name="geom", database="app"),
    )
    assert result.error is None
    assert result.chart_spec is None
    assert result.stats is not None
    assert result.stats.total == 3


@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
def test_connection_preview_boolean_stats_best_effort() -> None:
    import polars as pl

    engine = FakeCatalogEngine(
        dialect="postgres",
        default_database="app",
        details=_catalog_table(("flag", "boolean")),
    )

    def execute(query: str) -> Any:
        if "COUNT(*)" in query:
            return _polars_frame(4, 2, 1)
        # Simulate a dialect that rejects TRUE/FALSE predicates
        if "= TRUE" in query:
            raise RuntimeError("syntax error")
        return pl.DataFrame({"flag": [True, False, True, None]})

    engine.execute = execute  # type: ignore[method-assign]

    result = get_column_preview_for_connection(
        engine=engine,
        request=_request(column_name="flag", database="app"),
    )
    assert result.error is None
    assert result.stats is not None
    assert result.stats.total == 4
    assert result.stats.unique == 2
    assert result.stats.nulls == 1
    # Optional extras degrade to None instead of failing the preview
    assert result.stats.true is None
    assert result.stats.false is None


def test_connection_preview_unknown_column_type() -> None:
    engine = FakeCatalogEngine(dialect="weird", default_database="db")

    def execute(_query: str) -> Any:
        raise RuntimeError("unsupported")

    engine.execute = execute  # type: ignore[method-assign]

    result = get_column_preview_for_connection(
        engine=engine,
        request=_request(column_name="col", column_type=None, database="db"),
    )
    assert result.stats is None
    assert result.error is not None
    assert result.error.startswith("Unable to determine the type")


def test_connection_preview_uses_column_type_hint() -> None:
    import polars as pl

    engine = FakeCatalogEngine(dialect="weird", default_database="db")

    def execute(query: str) -> Any:
        if "COUNT(*)" in query:
            return pl.DataFrame(
                {"count": [1], "unique": [1], "null_count": [0]}
            )
        return pl.DataFrame({"col": ["a"]})

    engine.execute = execute  # type: ignore[method-assign]

    result = get_column_preview_for_connection(
        engine=engine,
        request=_request(column_name="col", column_type="string"),
    )
    assert result.error is None
    assert result.stats is not None
    assert result.stats.total == 1


# --------------------------------------------------------------------------- #
# Callback dispatch                                                            #
# --------------------------------------------------------------------------- #


def _callbacks(engine: Any, error: str | None = None) -> Any:
    from marimo._runtime.callbacks.datasets import DatasetCallbacks

    class FakeKernel:
        def get_sql_connection(self, _name: str) -> tuple[Any, str | None]:
            return engine, error

    return DatasetCallbacks(FakeKernel())  # type: ignore[arg-type]


async def test_callback_connection_unavailable() -> None:
    from marimo._runtime.callbacks import datasets as datasets_module

    callbacks = _callbacks(None, error="Engine not found")
    notifications: list[Any] = []
    with patch.object(
        datasets_module,
        "broadcast_notification",
        side_effect=notifications.append,
    ):
        await callbacks.preview_dataset_column(
            _request(column_name="id", database="app", schema="public")
        )

    assert len(notifications) == 1
    assert notifications[0].request_id == "req-1"
    assert "unavailable" in notifications[0].error
    assert notifications[0].table_name == "app.public.tbl"


async def test_callback_catalog_only_degrades() -> None:
    from unittest.mock import MagicMock

    from marimo._runtime.callbacks import datasets as datasets_module

    engine = MagicMock(spec=EngineCatalog)
    callbacks = _callbacks(engine)
    notifications: list[Any] = []
    with patch.object(
        datasets_module,
        "broadcast_notification",
        side_effect=notifications.append,
    ):
        await callbacks.preview_dataset_column(
            _request(column_name="id", database="app", schema="public")
        )

    assert len(notifications) == 1
    assert "cannot run SQL queries" in notifications[0].error


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="sqlalchemy not installed")
@pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
async def test_callback_connection_query_failure() -> None:
    import sqlalchemy as sa

    from marimo._runtime.callbacks import datasets as datasets_module

    sa_engine = sa.create_engine("sqlite:///:memory:")
    engine = SQLAlchemyEngine(sa_engine, cast(VariableName, "my_conn"))
    callbacks = _callbacks(engine)
    notifications: list[Any] = []
    with patch.object(
        datasets_module,
        "broadcast_notification",
        side_effect=notifications.append,
    ):
        await callbacks.preview_dataset_column(_request(column_name="id"))

    assert len(notifications) == 1
    assert notifications[0].request_id == "req-1"
    assert notifications[0].error is not None
