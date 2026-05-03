"""Extra tests for mempalace.config to cover remaining gaps."""

import json
import multiprocessing
import os
import queue
import sqlite3
import sys
import time

import pytest

from mempalace.config import MempalaceConfig
from mempalace.knowledge_graph import KnowledgeGraph


def _set_test_home(monkeypatch, home):
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _wait_for_file(path: str, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.01)
    return os.path.exists(path)


def _hold_palace_write_lock_for_migration(
    palace_path: str, home: str, ready_path: str, release_path: str
) -> None:
    os.environ["HOME"] = home
    os.environ["USERPROFILE"] = home
    from mempalace.palace import palace_write_lock

    with palace_write_lock(palace_path, blocking=True, timeout=5.0, purpose="kg_test_holder"):
        with open(ready_path, "w", encoding="utf-8"):
            pass
        while not os.path.exists(release_path):
            time.sleep(0.01)


def _import_mcp_server_for_kg_migration(
    palace_path: str, home: str, started_path: str, result_queue
) -> None:
    os.environ["HOME"] = home
    os.environ["USERPROFILE"] = home
    os.environ["MEMPALACE_PALACE_PATH"] = palace_path
    os.environ["MEMPALACE_MCP_WRITE_LOCK_TIMEOUT"] = "0"
    with open(started_path, "w", encoding="utf-8"):
        pass
    try:
        from mempalace import mcp_server

        result_queue.put({"db_path": mcp_server._kg.db_path})
    except Exception as exc:
        result_queue.put({"error": repr(exc)})


def test_config_bad_json(tmp_path):
    """Bad JSON in config file falls back to empty."""
    (tmp_path / "config.json").write_text("not json", encoding="utf-8")
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.palace_path  # still returns default


def test_people_map_from_file(tmp_path):
    (tmp_path / "people_map.json").write_text(json.dumps({"bob": "Robert"}), encoding="utf-8")
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.people_map == {"bob": "Robert"}


def test_people_map_bad_json(tmp_path):
    (tmp_path / "people_map.json").write_text("bad", encoding="utf-8")
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.people_map == {}


def test_people_map_missing(tmp_path):
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.people_map == {}


def test_topic_wings_default(tmp_path):
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert isinstance(cfg.topic_wings, list)
    assert "emotions" in cfg.topic_wings


def test_hall_keywords_default(tmp_path):
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert isinstance(cfg.hall_keywords, dict)
    assert "technical" in cfg.hall_keywords


def test_init_idempotent(tmp_path):
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    cfg.init()
    cfg.init()  # second call should not overwrite
    with open(tmp_path / "config.json") as f:
        data = json.load(f)
    assert "palace_path" in data


def test_save_people_map(tmp_path):
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    result = cfg.save_people_map({"alice": "Alice Smith"})
    assert result.exists()
    with open(result) as f:
        data = json.load(f)
    assert data["alice"] == "Alice Smith"


def test_env_mempal_palace_path(tmp_path):
    """MEMPAL_PALACE_PATH (legacy) should also work."""
    os.environ.pop("MEMPALACE_PALACE_PATH", None)
    raw = "/legacy/path"
    os.environ["MEMPAL_PALACE_PATH"] = raw
    try:
        cfg = MempalaceConfig(config_dir=str(tmp_path))
        # palace_path is normalized via abspath + expanduser — compare
        # against the normalized form so the test is portable between
        # POSIX (no-op) and Windows (prepends current drive letter).
        assert cfg.palace_path == os.path.abspath(os.path.expanduser(raw))
    finally:
        del os.environ["MEMPAL_PALACE_PATH"]


def test_collection_name_from_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"collection_name": "custom_col"}), encoding="utf-8"
    )
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.collection_name == "custom_col"


def test_kg_migration_copies_legacy_to_selected_palace(tmp_path, monkeypatch):
    """Legacy KG is copied into the selected palace and legacy remains."""
    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    palace = tmp_path / "palace"
    palace.mkdir()

    legacy_kg = home / ".mempalace" / "knowledge_graph.sqlite3"
    graph = KnowledgeGraph(db_path=str(legacy_kg))
    graph.add_entity("TestEntity")
    graph.close()

    from mempalace.mcp_server import _migrate_legacy_kg

    result = _migrate_legacy_kg(str(palace))
    expected = str(palace / "knowledge_graph.sqlite3")
    assert result == expected
    assert os.path.isfile(result)
    assert os.path.isfile(legacy_kg)

    conn = sqlite3.connect(result)
    row = conn.execute("SELECT name FROM entities WHERE name='TestEntity'").fetchone()
    conn.close()
    assert row is not None


def test_kg_migration_waits_for_lock_before_import_creates_kg(tmp_path, monkeypatch):
    """Import-time legacy migration must not time out into an empty selected KG."""
    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    palace = tmp_path / "palace"
    palace.mkdir()

    legacy_kg = home / ".mempalace" / "knowledge_graph.sqlite3"
    graph = KnowledgeGraph(db_path=str(legacy_kg))
    graph.add_entity("MigratedEntity")
    graph.close()

    ctx = multiprocessing.get_context("spawn")
    ready = str(tmp_path / "migration-lock-ready")
    release = str(tmp_path / "migration-lock-release")
    started = str(tmp_path / "mcp-import-started")
    result_queue = ctx.Queue()
    holder = ctx.Process(
        target=_hold_palace_write_lock_for_migration,
        args=(str(palace), str(home), ready, release),
    )
    importer = ctx.Process(
        target=_import_mcp_server_for_kg_migration,
        args=(str(palace), str(home), started, result_queue),
    )

    holder.start()
    try:
        assert _wait_for_file(ready), "holder did not acquire palace write lock"
        importer.start()
        assert _wait_for_file(started), "importer did not start"
        try:
            early = result_queue.get(timeout=0.3)
        except queue.Empty:
            early = None
        assert early is None, f"migration completed before lock release: {early}"
        assert not (palace / "knowledge_graph.sqlite3").exists()
    finally:
        with open(release, "w", encoding="utf-8"):
            pass
        holder.join(timeout=5)

    importer.join(timeout=30)
    assert holder.exitcode == 0
    assert importer.exitcode == 0
    result = result_queue.get(timeout=1)
    assert result == {"db_path": str(palace / "knowledge_graph.sqlite3")}

    conn = sqlite3.connect(result["db_path"])
    row = conn.execute("SELECT name FROM entities WHERE name='MigratedEntity'").fetchone()
    conn.close()
    assert row is not None


def test_kg_migration_does_not_overwrite_selected_palace(tmp_path, monkeypatch):
    """Existing selected-palace KG wins over legacy KG."""
    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    palace = tmp_path / "palace"
    palace.mkdir()

    selected_kg = palace / "knowledge_graph.sqlite3"
    selected = KnowledgeGraph(db_path=str(selected_kg))
    selected.add_entity("SelectedEntity")
    selected.close()

    legacy_kg = home / ".mempalace" / "knowledge_graph.sqlite3"
    legacy = KnowledgeGraph(db_path=str(legacy_kg))
    legacy.add_entity("LegacyEntity")
    legacy.close()

    from mempalace.mcp_server import _migrate_legacy_kg

    result = _migrate_legacy_kg(str(palace))
    assert result == str(selected_kg)

    conn = sqlite3.connect(result)
    selected_row = conn.execute("SELECT name FROM entities WHERE name='SelectedEntity'").fetchone()
    legacy_row = conn.execute("SELECT name FROM entities WHERE name='LegacyEntity'").fetchone()
    conn.close()
    assert selected_row is not None
    assert legacy_row is None


def test_kg_new_install_uses_selected_palace_path(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    palace = tmp_path / "palace"
    palace.mkdir()

    from mempalace.mcp_server import _migrate_legacy_kg

    assert _migrate_legacy_kg(str(palace)) == str(palace / "knowledge_graph.sqlite3")


def test_kg_migration_corrupt_legacy_db_returns_target_without_crash(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    palace = tmp_path / "palace"
    palace.mkdir()

    legacy_kg = home / ".mempalace" / "knowledge_graph.sqlite3"
    legacy_kg.parent.mkdir(parents=True, exist_ok=True)
    legacy_kg.write_bytes(b"not a sqlite database")

    from mempalace.mcp_server import _migrate_legacy_kg

    result = _migrate_legacy_kg(str(palace))
    assert result == str(palace / "knowledge_graph.sqlite3")


def test_kg_migration_sidecar_cleanup_on_failure(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    palace = tmp_path / "palace"
    palace.mkdir()

    legacy_kg = home / ".mempalace" / "knowledge_graph.sqlite3"
    legacy_graph = KnowledgeGraph(db_path=str(legacy_kg))
    legacy_graph.add_entity("Legacy")
    legacy_graph.close()

    from mempalace import mcp_server

    def _failing_replace(*_args, **_kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "replace", _failing_replace)
    result = mcp_server._migrate_legacy_kg(str(palace))
    assert result == str(palace / "knowledge_graph.sqlite3")

    base = str(palace / "knowledge_graph.sqlite3")
    assert not os.path.exists(base + ".migrating")
    assert not os.path.exists(base + ".migrating-wal")
    assert not os.path.exists(base + ".migrating-shm")


def test_kg_migration_same_path_legacy_and_target_is_noop(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    palace = home / ".mempalace"
    palace.mkdir(parents=True, exist_ok=True)

    legacy = palace / "knowledge_graph.sqlite3"
    graph = KnowledgeGraph(db_path=str(legacy))
    graph.add_entity("SamePath")
    graph.close()

    from mempalace.mcp_server import _migrate_legacy_kg

    assert _migrate_legacy_kg(str(palace)) == str(palace / "knowledge_graph.sqlite3")


@pytest.mark.skipif(sys.platform == "win32", reason="chmod simulation is non-portable on Windows")
def test_kg_migration_readonly_legacy_does_not_crash(tmp_path, monkeypatch):
    home = tmp_path / "home"
    _set_test_home(monkeypatch, home)
    palace = tmp_path / "palace"
    palace.mkdir()

    legacy_kg = home / ".mempalace" / "knowledge_graph.sqlite3"
    legacy_kg.parent.mkdir(parents=True, exist_ok=True)
    graph = KnowledgeGraph(db_path=str(legacy_kg))
    graph.add_entity("LegacyReadOnly")
    graph.close()
    legacy_kg.chmod(0o000)

    from mempalace.mcp_server import _migrate_legacy_kg

    result = _migrate_legacy_kg(str(palace))
    assert result == str(palace / "knowledge_graph.sqlite3")
