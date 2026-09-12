# Copyright 2026 Marimo. All rights reserved.
import asyncio
import threading
from pathlib import Path

from marimo._export.exporter import AutoExporter


def _list_temp_files(directory: Path) -> list[Path]:
    return list(directory.glob("*.tmp"))


async def test_newer_auto_export_supersedes_an_older_write(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "marimo._export.exporter.notebook_output_dir",
        lambda _path: tmp_path,
    )
    exporter = AutoExporter()
    try:
        older_stage_started = threading.Event()
        release_older_stage = threading.Event()
        original_stage = AutoExporter._stage_file_sync

        def stage_file(tmp_file: Path, content: str) -> None:
            if content == "older":
                older_stage_started.set()
                assert release_older_stage.wait(timeout=2)
            original_stage(tmp_file, content)

        monkeypatch.setattr(exporter, "_stage_file_sync", stage_file)
        older = exporter.reserve_revision("notebook.py", "html")
        older_task = asyncio.create_task(
            exporter.save_html("notebook.py", "older", revision=older)
        )
        assert await asyncio.to_thread(older_stage_started.wait, 2)

        newer = exporter.reserve_revision("notebook.py", "html")
        newer_task = asyncio.create_task(
            exporter.save_html("notebook.py", "newer", revision=newer)
        )
        release_older_stage.set()

        assert not await older_task
        assert await newer_task
        assert (tmp_path / "notebook.html").read_text() == "newer"
        # Staged temp files are either published or removed.
        assert await asyncio.to_thread(_list_temp_files, tmp_path) == []
    finally:
        exporter.cleanup()


async def test_different_formats_have_independent_revisions(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "marimo._export.exporter.notebook_output_dir",
        lambda _path: tmp_path,
    )
    exporter = AutoExporter()
    try:
        html_rev = exporter.reserve_revision("notebook.py", "html")
        md_rev = exporter.reserve_revision("notebook.py", "md")

        assert await exporter.save_html(
            "notebook.py", "html-1", revision=html_rev
        )
        assert await exporter.save_md("notebook.py", "md-1", revision=md_rev)

        # A newer HTML revision must not invalidate the committed Markdown.
        html_rev_2 = exporter.reserve_revision("notebook.py", "html")
        assert not await exporter.save_html(
            "notebook.py", "html-stale", revision=html_rev
        )
        assert await exporter.save_html(
            "notebook.py", "html-2", revision=html_rev_2
        )

        assert (tmp_path / "notebook.html").read_text() == "html-2"
        assert (tmp_path / "notebook.md").read_text() == "md-1"
    finally:
        exporter.cleanup()


async def test_stale_revision_is_dropped_without_touching_the_file(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "marimo._export.exporter.notebook_output_dir",
        lambda _path: tmp_path,
    )
    exporter = AutoExporter()
    try:
        revision = exporter.reserve_revision("notebook.py", "ipynb")
        exporter.reserve_revision("notebook.py", "ipynb")  # supersedes

        assert not await exporter.save_ipynb(
            "notebook.py", "stale", revision=revision
        )
        assert not (tmp_path / "notebook.ipynb").exists()
        assert await asyncio.to_thread(_list_temp_files, tmp_path) == []
    finally:
        exporter.cleanup()


async def test_empty_export_is_not_published(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "marimo._export.exporter.notebook_output_dir",
        lambda _path: tmp_path,
    )
    exporter = AutoExporter()
    try:
        revision = exporter.reserve_revision("notebook.py", "md")
        assert not await exporter.save_md("notebook.py", "", revision=revision)
        assert not (tmp_path / "notebook.md").exists()
        assert await asyncio.to_thread(_list_temp_files, tmp_path) == []
    finally:
        exporter.cleanup()


async def test_cleanup_discards_inflight_revisions_but_allows_new_exports(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "marimo._export.exporter.notebook_output_dir",
        lambda _path: tmp_path,
    )
    exporter = AutoExporter()
    revision = exporter.reserve_revision("notebook.py", "html")
    exporter.cleanup()

    # A revision reserved before shutdown is discarded.
    assert not await exporter.save_html(
        "notebook.py", "before-shutdown", revision=revision
    )
    assert not (tmp_path / "notebook.html").exists()

    # A later export works again from a clean slate (reopened notebook).
    exporter = AutoExporter()
    monkeypatch.setattr(
        "marimo._export.exporter.notebook_output_dir",
        lambda _path: tmp_path,
    )
    try:
        new_revision = exporter.reserve_revision("notebook.py", "html")
        assert new_revision == 1
        assert await exporter.save_html(
            "notebook.py", "after-reopen", revision=new_revision
        )
        assert (tmp_path / "notebook.html").read_text() == "after-reopen"
    finally:
        exporter.cleanup()


async def test_cleanup_during_staging_discards_the_write(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "marimo._export.exporter.notebook_output_dir",
        lambda _path: tmp_path,
    )
    exporter = AutoExporter()
    stage_started = threading.Event()
    release_stage = threading.Event()
    original_stage = AutoExporter._stage_file_sync

    def stage_file(tmp_file: Path, content: str) -> None:
        stage_started.set()
        assert release_stage.wait(timeout=2)
        original_stage(tmp_file, content)

    monkeypatch.setattr(exporter, "_stage_file_sync", stage_file)
    revision = exporter.reserve_revision("notebook.py", "html")
    write_task = asyncio.create_task(
        exporter.save_html("notebook.py", "in-flight", revision=revision)
    )
    assert await asyncio.to_thread(stage_started.wait, 2)

    # Server shutdown while the write is in flight must discard the result.
    exporter.cleanup()
    release_stage.set()
    assert not await write_task
    assert not (tmp_path / "notebook.html").exists()
    assert await asyncio.to_thread(_list_temp_files, tmp_path) == []

    # The exporter is reusable from a clean epoch afterwards.
    try:
        new_revision = exporter.reserve_revision("notebook.py", "html")
        assert new_revision == 1
        assert await exporter.save_html(
            "notebook.py", "after-reopen", revision=new_revision
        )
        assert (tmp_path / "notebook.html").read_text() == "after-reopen"
    finally:
        exporter.cleanup()
