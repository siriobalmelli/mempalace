"""
test_mcp_server.py — Tests for the MCP server tool handlers and dispatch.

Tests each tool handler directly (unit-level) and the handle_request
dispatch layer (integration-level). Uses isolated palace + KG fixtures
via monkeypatch to avoid touching real data.
"""

from datetime import datetime
import inspect
import json
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time

import pytest


PALACE_WRITE_TOOLS = {
    "mempalace_add_drawer",
    "mempalace_delete_drawer",
    "mempalace_update_drawer",
    "mempalace_kg_add",
    "mempalace_kg_invalidate",
    "mempalace_diary_write",
}

TOOL_ARG_DEFAULTS = {
    "query": "hello",
    "limit": 1,
    "offset": 0,
    "last_n": 3,
    "max_hops": 1,
    "wing": "team",
    "room": "notes",
    "drawer_id": "drawer_unit_abc",
    "entity": "Alice",
    "subject": "Alice",
    "predicate": "likes",
    "object": "chess",
    "agent_name": "agent",
    "entry": "daily update",
    "start_room": "overview",
    "tunnel_id": "tunnel_1",
    "source_wing": "backend",
    "source_room": "planning",
    "target_wing": "frontend",
    "target_room": "design",
    "content": "Sample diary entry",
}


def _required_tool_args(tool_name: str, mcp_server):
    schema = mcp_server.TOOLS[tool_name]["input_schema"]
    args = {key: TOOL_ARG_DEFAULTS.get(key, "value") for key in schema.get("required", [])}
    if tool_name == "mempalace_update_drawer":
        args.setdefault("content", "updated content")
        args.setdefault("wing", "team")
    return args


class _StubCollection:
    def get(self, *args, **kwargs):
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, *args, **kwargs):
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def count(self):
        return 0

    def upsert(self, *args, **kwargs):
        return None

    def add(self, *args, **kwargs):
        return None

    def update(self, *args, **kwargs):
        return None

    def delete(self, *args, **kwargs):
        return None


class _StubKG:
    def query_entity(self, *args, **kwargs):
        return []

    def timeline(self, *args, **kwargs):
        return []

    def stats(self):
        return {"entities": 0, "triples": 0}

    def add_triple(self, *args, **kwargs):
        return "triple_id"

    def invalidate(self, *args, **kwargs):
        return None


def _install_readonly_backends(monkeypatch, mcp_server):
    stub_col = _StubCollection()
    monkeypatch.setattr(mcp_server, "_get_collection", lambda *args, **kwargs: stub_col)
    monkeypatch.setattr(mcp_server, "search_memories", lambda *args, **kwargs: {"results": []})
    monkeypatch.setattr(mcp_server, "_fetch_all_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr(mcp_server, "_get_cached_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr(mcp_server, "_kg", _StubKG())
    monkeypatch.setattr(mcp_server, "traverse", lambda *args, **kwargs: [])
    monkeypatch.setattr(mcp_server, "find_tunnels", lambda *args, **kwargs: [])
    monkeypatch.setattr(mcp_server, "graph_stats", lambda *args, **kwargs: {})
    monkeypatch.setattr(mcp_server, "create_tunnel", lambda *args, **kwargs: {"created": True})
    monkeypatch.setattr(mcp_server, "list_tunnels", lambda *args, **kwargs: [])
    monkeypatch.setattr(mcp_server, "delete_tunnel", lambda *args, **kwargs: {"deleted": True})
    monkeypatch.setattr(mcp_server, "follow_tunnels", lambda *args, **kwargs: [])
    monkeypatch.setattr(mcp_server, "_wal_log", lambda *args, **kwargs: None)


def _patch_mcp_server(monkeypatch, config, kg):
    """Patch the mcp_server module globals to use test fixtures."""
    from mempalace import mcp_server

    old_kg = getattr(mcp_server, "_kg", None)
    if old_kg is not kg and hasattr(old_kg, "close"):
        old_kg.close()
    monkeypatch.setattr(mcp_server, "_config", config)
    monkeypatch.setattr(mcp_server, "_kg", kg)


def _get_collection(palace_path, create=False):
    """Helper to get collection from test palace.

    Returns (client, collection) so callers can clean up the client
    when they are done.
    """
    import chromadb

    client = chromadb.PersistentClient(path=palace_path)
    if create:
        return (
            client,
            client.get_or_create_collection("mempalace_drawers", metadata={"hnsw:space": "cosine"}),
        )
    return client, client.get_collection("mempalace_drawers")


def _get_mp_context():
    return multiprocessing.get_context("spawn" if os.name == "nt" else "fork")


def _hold_palace_write_lock_process(palace_path: str, home: str, ready: str, release: str):
    os.environ["HOME"] = home
    os.environ["USERPROFILE"] = home
    from mempalace.palace import palace_write_lock

    with palace_write_lock(palace_path, blocking=True, timeout=5.0, purpose="mcp_test_holder"):
        open(ready, "w").close()
        for _ in range(1000):
            if os.path.exists(release):
                return
            time.sleep(0.01)


def _mcp_add_drawer_process(palace_path: str, home: str, started: str, result_queue):
    os.environ["HOME"] = home
    os.environ["USERPROFILE"] = home
    os.environ["MEMPALACE_PALACE_PATH"] = palace_path
    os.environ["MEMPALACE_MCP_WRITE_LOCK_TIMEOUT"] = "5"
    from mempalace.mcp_server import tool_add_drawer

    open(started, "w").close()
    try:
        result_queue.put(
            tool_add_drawer(
                wing="integration",
                room="locks",
                content="cross process mcp write lock integration drawer",
            )
        )
    except Exception as exc:
        result_queue.put({"success": False, "error": repr(exc)})


def _wait_for_file(path: str, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.01)
    return os.path.exists(path)


# ── Internal test helpers ───────────────────────────────────────────────


def _run_tool_with_external_write_lock(
    monkeypatch,
    config,
    kg,
    tmp_path,
    tool,
    tool_args,
    *,
    timeout_env="0.05",
):
    """Run a tool while another process holds the write lock."""
    _patch_mcp_server(monkeypatch, config, kg)
    if timeout_env is not None:
        monkeypatch.setenv("MEMPALACE_MCP_WRITE_LOCK_TIMEOUT", timeout_env)

    ctx = _get_mp_context()
    suffix = tool.__name__
    ready = str(tmp_path / f"holder-ready-{suffix}")
    release = str(tmp_path / f"holder-release-{suffix}")
    holder = ctx.Process(
        target=_hold_palace_write_lock_process,
        args=(config.palace_path, os.environ["HOME"], ready, release),
    )
    holder.start()
    result = None
    try:
        assert _wait_for_file(ready), "holder did not acquire lock"
        result = tool(**tool_args)
    finally:
        open(release, "w").close()
        holder.join(timeout=5)
    assert holder.exitcode == 0
    return result


def _get_isolated_mcp_server(monkeypatch, config, kg, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", config.palace_path)
    import importlib

    from mempalace import mcp_server

    old_kg = getattr(mcp_server, "_kg", None)
    if hasattr(old_kg, "close"):
        old_kg.close()
    mcp_server = importlib.reload(mcp_server)
    _patch_mcp_server(monkeypatch, config, kg)
    return mcp_server


# ── Protocol Layer ──────────────────────────────────────────────────────


class TestHandleRequest:
    def test_initialize(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "initialize", "id": 1, "params": {}})
        assert resp["result"]["serverInfo"]["name"] == "mempalace"
        assert resp["id"] == 1

    def test_initialize_negotiates_client_version(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request(
            {
                "method": "initialize",
                "id": 1,
                "params": {"protocolVersion": "2025-11-25"},
            }
        )
        assert resp["result"]["protocolVersion"] == "2025-11-25"

    def test_initialize_negotiates_older_supported_version(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request(
            {
                "method": "initialize",
                "id": 1,
                "params": {"protocolVersion": "2025-03-26"},
            }
        )
        assert resp["result"]["protocolVersion"] == "2025-03-26"

    def test_initialize_unknown_version_falls_back_to_latest(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request(
            {
                "method": "initialize",
                "id": 1,
                "params": {"protocolVersion": "9999-12-31"},
            }
        )
        from mempalace.mcp_server import SUPPORTED_PROTOCOL_VERSIONS

        assert resp["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[0]

    def test_initialize_missing_version_uses_oldest(self):
        from mempalace.mcp_server import handle_request, SUPPORTED_PROTOCOL_VERSIONS

        resp = handle_request({"method": "initialize", "id": 1, "params": {}})
        assert resp["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSIONS[-1]

    def test_notifications_initialized_returns_none(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "notifications/initialized", "id": None, "params": {}})
        assert resp is None

    def test_ping_returns_empty_result(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "ping", "id": 11, "params": {}})
        assert resp["id"] == 11
        assert resp["result"] == {}

    def test_tools_list(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "tools/list", "id": 2, "params": {}})
        tools = resp["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "mempalace_status" in names
        assert "mempalace_search" in names
        assert "mempalace_add_drawer" in names
        assert "mempalace_kg_add" in names

    def test_palace_arg_expands_tilde_in_subprocess(self, tmp_path):
        """Import-time --palace handling must not preserve a literal tilde."""
        home = tmp_path / "home"
        home.mkdir()
        env = os.environ.copy()
        env["HOME"] = str(home)
        env["USERPROFILE"] = str(home)
        proc = subprocess.run(
            [sys.executable, "-m", "mempalace.mcp_server", "--palace", "~/mempalace-smoke"],
            input=(
                '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
                '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"mempalace_add_drawer","arguments":{"wing":"smoke","room":"general","content":"tilde palace smoke drawer"}}}\n'
                '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"mempalace_status","arguments":{}}}\n'
            ),
            text=True,
            capture_output=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
        status = responses[-1]
        body = json.loads(status["result"]["content"][0]["text"])
        assert "palace_path" not in body
        assert (home / "mempalace-smoke" / "chroma.sqlite3").exists()
        assert not (home / "~" / "mempalace-smoke").exists()

    def test_null_arguments_does_not_hang(self, monkeypatch, config, palace_path, seeded_kg):
        """Sending arguments: null should return a result, not hang (#394)."""
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import handle_request

        _client, _col = _get_collection(palace_path, create=True)
        del _client
        resp = handle_request(
            {
                "method": "tools/call",
                "id": 10,
                "params": {"name": "mempalace_status", "arguments": None},
            }
        )
        assert "error" not in resp
        assert resp["result"] is not None

    def test_unknown_tool(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request(
            {
                "method": "tools/call",
                "id": 3,
                "params": {"name": "nonexistent_tool", "arguments": {}},
            }
        )
        assert resp["error"]["code"] == -32601

    def test_unknown_method(self):
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "unknown/method", "id": 4, "params": {}})
        assert resp["error"]["code"] == -32601

    def test_any_notification_returns_none(self):
        """All notifications/* methods should return None (no response)."""
        from mempalace.mcp_server import handle_request

        for method in [
            "notifications/initialized",
            "notifications/cancelled",
            "notifications/progress",
            "notifications/roots/list_changed",
        ]:
            resp = handle_request({"method": method, "params": {}})
            assert resp is None, f"{method} should return None"

    def test_unknown_method_no_id_returns_none(self):
        """Messages without id (notifications) must never get a response."""
        from mempalace.mcp_server import handle_request

        resp = handle_request({"method": "unknown/thing", "params": {}})
        assert resp is None

    def test_malformed_method_none(self):
        """method=None or missing should not crash."""
        from mempalace.mcp_server import handle_request

        # Explicit None
        resp = handle_request({"method": None, "params": {}})
        assert resp is None  # no id → no response

        # Missing method entirely
        resp = handle_request({"params": {}})
        assert resp is None

        # method=None with id → should return error, not crash
        resp = handle_request({"method": None, "id": 99, "params": {}})
        assert resp["error"]["code"] == -32601

    def test_tools_call_dispatches(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import handle_request

        # Create a collection so status works
        _client, _col = _get_collection(palace_path, create=True)
        del _client

        resp = handle_request(
            {
                "method": "tools/call",
                "id": 5,
                "params": {"name": "mempalace_status", "arguments": {}},
            }
        )
        assert "result" in resp
        content = json.loads(resp["result"]["content"][0]["text"])
        assert "total_drawers" in content


# ── Read Tools ──────────────────────────────────────────────────────────


class TestReadTools:
    def test_status_cold_start_no_collection(self, monkeypatch, config, palace_path, kg):
        """Status on a valid palace with no ChromaDB collection yet (#830).

        After `mempalace init`, chroma.sqlite3 exists but the mempalace_drawers
        collection has not been created (no mine or add_drawer yet).  Status
        should return total_drawers: 0, not 'No palace found'.
        """
        import chromadb

        _patch_mcp_server(monkeypatch, config, kg)
        # Create the DB file (init does this) but NOT the collection
        client = chromadb.PersistentClient(path=palace_path)
        del client
        from mempalace.mcp_server import tool_status

        result = tool_status()
        assert "error" not in result, f"cold-start should not error: {result}"
        assert result["total_drawers"] == 0

    def test_status_empty_palace(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_status

        result = tool_status()
        assert result["total_drawers"] == 0
        assert result["wings"] == {}

    def test_status_with_data(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_status

        result = tool_status()
        assert result["total_drawers"] == 4
        assert "project" in result["wings"]
        assert "notes" in result["wings"]

    def test_status_handles_none_metadata_without_partial(
        self, monkeypatch, config, palace_path, kg
    ):
        """tool_status must not crash or go partial when the metadata cache
        returns a ``None`` entry — palaces can contain drawers with no
        metadata (older mining paths, third-party writes). Before the guard,
        ``m.get("wing")`` raised AttributeError mid-tally and the result
        carried ``"error"`` + ``"partial": True`` even though the data was
        perfectly fetchable."""
        from unittest.mock import patch as _patch

        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_status

        # Inject a metadata cache where one entry is None
        with _patch("mempalace.mcp_server._get_collection") as mock_get_col:
            fake_col = type("C", (), {"count": lambda self: 2})()
            mock_get_col.return_value = fake_col
            with _patch(
                "mempalace.mcp_server._get_cached_metadata",
                return_value=[{"wing": "proj", "room": "r"}, None],
            ):
                result = tool_status()

        # The None-metadata drawer falls under 'unknown/unknown' — no crash,
        # no partial flag.
        assert "error" not in result
        assert result.get("partial") is not True
        assert result["total_drawers"] == 2
        assert result["wings"].get("proj") == 1
        assert result["wings"].get("unknown") == 1

    def test_list_wings(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_list_wings

        result = tool_list_wings()
        assert result["wings"]["project"] == 3
        assert result["wings"]["notes"] == 1

    def test_list_rooms_all(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_list_rooms

        result = tool_list_rooms()
        assert "backend" in result["rooms"]
        assert "frontend" in result["rooms"]
        assert "planning" in result["rooms"]

    def test_list_rooms_filtered(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_list_rooms

        result = tool_list_rooms(wing="project")
        assert "backend" in result["rooms"]
        assert "planning" not in result["rooms"]

    def test_get_taxonomy(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_get_taxonomy

        result = tool_get_taxonomy()
        assert result["taxonomy"]["project"]["backend"] == 2
        assert result["taxonomy"]["project"]["frontend"] == 1
        assert result["taxonomy"]["notes"]["planning"] == 1

    def test_no_palace_returns_error(self, monkeypatch, config, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_status

        result = tool_status()
        assert "error" in result


# ── Search Tool ─────────────────────────────────────────────────────────


class TestSearchTool:
    def test_search_basic(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_search

        result = tool_search(query="JWT authentication tokens")
        assert "results" in result
        assert len(result["results"]) > 0
        # Top result should be the auth drawer
        top = result["results"][0]
        assert "JWT" in top["text"] or "authentication" in top["text"].lower()

    def test_search_with_wing_filter(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_search

        result = tool_search(query="planning", wing="notes")
        assert all(r["wing"] == "notes" for r in result["results"])

    def test_search_with_room_filter(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_search

        result = tool_search(query="database", room="backend")
        assert all(r["room"] == "backend" for r in result["results"])

    def test_search_min_similarity_backwards_compat(
        self, monkeypatch, config, palace_path, seeded_collection, kg
    ):
        """Old min_similarity param still works via backwards-compat shim."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_search

        # Old name should work
        result = tool_search(query="JWT", min_similarity=0.5)
        assert "results" in result

        result = tool_search(query="JWT", min_similarity=-0.1)
        assert "error" in result

        result = tool_search(query="JWT", min_similarity=1.1)
        assert "error" in result

        # Old name takes precedence when both provided
        result_strict = tool_search(query="JWT", max_distance=999.0, min_similarity=0.9)
        result_loose = tool_search(query="JWT", max_distance=0.01, min_similarity=0.1)
        assert len(result_strict["results"]) <= len(result_loose["results"])

    def test_list_rooms_rejects_invalid_wing(self, monkeypatch, config, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "_get_collection", lambda: pytest.fail())

        result = mcp_server.tool_list_rooms(wing="../etc/passwd")
        assert "error" in result

    def test_search_rejects_invalid_room(self, monkeypatch, config, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "search_memories", lambda: pytest.fail())

        result = mcp_server.tool_search(query="JWT", room="../backend")
        assert "error" in result

    def test_list_drawers_rejects_invalid_wing(self, monkeypatch, config, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "_get_collection", lambda: pytest.fail())

        result = mcp_server.tool_list_drawers(wing="../notes")
        assert "error" in result

    def test_find_tunnels_rejects_invalid_wing(self, monkeypatch, config, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "_get_collection", lambda: pytest.fail())

        result = mcp_server.tool_find_tunnels(wing_a="../project")
        assert "error" in result

    def test_wal_redacts_sensitive_fields(self, monkeypatch, config, kg, tmp_path):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        wal_file = tmp_path / "write_log.jsonl"
        monkeypatch.setattr(mcp_server, "_WAL_FILE", wal_file)

        mcp_server._wal_log(
            "test",
            {"content": "secret note", "query": "private search", "safe": "ok"},
        )

        entry = json.loads(wal_file.read_text().strip())
        assert entry["palace_path"] == config.palace_path
        assert entry["params"]["content"].startswith("[REDACTED")
        assert entry["params"]["query"].startswith("[REDACTED")
        assert entry["params"]["safe"] == "ok"

    def test_wal_includes_palace_path(self, monkeypatch, config, kg, tmp_path):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        wal_file = tmp_path / "write_log.jsonl"
        monkeypatch.setattr(mcp_server, "_WAL_FILE", wal_file)

        mcp_server._wal_log("test", {"safe": "ok"})

        entry = json.loads(wal_file.read_text().strip())
        assert entry["palace_path"] == config.palace_path
        assert entry["params"] == {"safe": "ok"}

    def test_wal_concurrent_entries_are_valid_jsonl(self, monkeypatch, config, kg, tmp_path):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        wal_file = tmp_path / "write_log.jsonl"
        monkeypatch.setattr(mcp_server, "_WAL_FILE", wal_file)

        def write_entry(i):
            mcp_server._wal_log("test", {"safe": i, "content": f"secret-{i}"})

        threads = [threading.Thread(target=write_entry, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        lines = wal_file.read_text().splitlines()
        assert len(lines) == 20
        for line in lines:
            entry = json.loads(line)
            assert entry["palace_path"] == config.palace_path
            assert entry["params"]["content"].startswith("[REDACTED")

    def test_mcp_import_survives_uncreatable_wal_dir(self, config, tmp_path):
        home_file = tmp_path / "home-is-a-file"
        home_file.write_text("not a directory", encoding="utf-8")

        env = os.environ.copy()
        env["HOME"] = str(home_file)
        env["USERPROFILE"] = str(home_file)
        env["MEMPALACE_PALACE_PATH"] = config.palace_path
        code = (
            "from mempalace import mcp_server; "
            "mcp_server._wal_log('test', {'safe': 'ok'}); "
            "print('ok')"
        )

        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 0, result.stderr


class TestToolsInvariants:
    def test_every_tool_has_handler(self, monkeypatch, config, kg, tmp_path):
        mcp_server = _get_isolated_mcp_server(monkeypatch, config, kg, tmp_path)

        for tool_name, entry in mcp_server.TOOLS.items():
            assert callable(entry["handler"]), f"{tool_name} handler is not callable"

    def test_every_tool_has_description(self, monkeypatch, config, kg, tmp_path):
        mcp_server = _get_isolated_mcp_server(monkeypatch, config, kg, tmp_path)

        for tool_name, entry in mcp_server.TOOLS.items():
            desc = entry.get("description")
            assert isinstance(desc, str), f"{tool_name} description must be a string"
            assert desc, f"{tool_name} description must not be empty"

    def test_every_tool_has_input_schema_with_type(self, monkeypatch, config, kg, tmp_path):
        mcp_server = _get_isolated_mcp_server(monkeypatch, config, kg, tmp_path)

        for tool_name, entry in mcp_server.TOOLS.items():
            assert "input_schema" in entry, f"{tool_name} missing input_schema"
            schema = entry["input_schema"]
            assert schema["type"] == "object", f"{tool_name} input schema must be object"

    def test_handler_params_subset_of_schema_properties(self, monkeypatch, config, kg, tmp_path):
        mcp_server = _get_isolated_mcp_server(monkeypatch, config, kg, tmp_path)

        for tool_name, entry in mcp_server.TOOLS.items():
            schema = entry["input_schema"]
            properties = set(schema.get("properties", {}).keys())
            for param_name, param in inspect.signature(entry["handler"]).parameters.items():
                if param.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                    continue
                assert (
                    param_name in properties
                ), f"{tool_name} has param '{param_name}' not in schema properties"

    def test_schema_required_subset_of_properties(self, monkeypatch, config, kg, tmp_path):
        mcp_server = _get_isolated_mcp_server(monkeypatch, config, kg, tmp_path)

        for tool_name, entry in mcp_server.TOOLS.items():
            schema = entry["input_schema"]
            properties = set(schema.get("properties", {}).keys())
            required = set(schema.get("required", []))
            assert required <= properties, f"{tool_name} required keys not in properties"

    def test_write_tools_use_palace_write_lock(self, monkeypatch, config, kg, tmp_path):
        mcp_server = _get_isolated_mcp_server(monkeypatch, config, kg, tmp_path)
        lock_calls = []

        class _LockSpy:
            def __call__(self, *args, **kwargs):
                lock_calls.append(kwargs.get("purpose", ""))
                return self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        _install_readonly_backends(monkeypatch, mcp_server)
        monkeypatch.setattr(mcp_server, "_wal_log", lambda *args, **kwargs: None)
        spy = _LockSpy()
        monkeypatch.setattr(mcp_server, "palace_write_lock", spy)

        for name, entry in mcp_server.TOOLS.items():
            if name not in PALACE_WRITE_TOOLS:
                continue
            before = len(lock_calls)
            entry["handler"](**_required_tool_args(name, mcp_server))
            assert len(lock_calls) == before + 1

        assert set(lock_calls) == {
            "add_drawer",
            "delete_drawer",
            "update_drawer",
            "kg_add",
            "kg_invalidate",
            "diary_write",
        }

    def test_read_tools_do_not_use_palace_write_lock(self, monkeypatch, config, kg, tmp_path):
        mcp_server = _get_isolated_mcp_server(monkeypatch, config, kg, tmp_path)

        lock_calls = []

        class _LockSpy:
            def __call__(self, *args, **kwargs):
                lock_calls.append(kwargs.get("purpose", ""))
                return self

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        _install_readonly_backends(monkeypatch, mcp_server)
        monkeypatch.setattr(mcp_server, "palace_write_lock", _LockSpy())

        for name, entry in mcp_server.TOOLS.items():
            if name in PALACE_WRITE_TOOLS:
                continue
            result = entry["handler"](**_required_tool_args(name, mcp_server))
            assert result is not None
        assert not lock_calls


class TestReadToolsNoLock:
    @pytest.mark.parametrize(
        "tool_name,tool_args",
        [
            ("tool_status", {}),
            ("tool_list_wings", {}),
            ("tool_list_rooms", {}),
            ("tool_get_taxonomy", {}),
            ("tool_search", {"query": "test"}),
            ("tool_get_drawer", {"drawer_id": "nonexistent"}),
            ("tool_list_drawers", {}),
            ("tool_kg_query", {"entity": "Alice"}),
            ("tool_kg_timeline", {}),
            ("tool_kg_stats", {}),
            ("tool_diary_read", {"agent_name": "Nobody"}),
        ],
    )
    def test_read_tool_not_blocked_by_write_lock(
        self,
        monkeypatch,
        config,
        palace_path,
        kg,
        tmp_path,
        tool_name,
        tool_args,
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        _get_collection(palace_path, create=True)
        from mempalace import mcp_server

        ctx = _get_mp_context()
        ready = str(tmp_path / f"holder-ready-{tool_name}")
        release = str(tmp_path / f"holder-release-{tool_name}")
        holder = ctx.Process(
            target=_hold_palace_write_lock_process,
            args=(config.palace_path, os.environ["HOME"], ready, release),
        )
        holder.start()
        try:
            assert _wait_for_file(ready), "holder did not acquire lock"
            tool = getattr(mcp_server, tool_name)
            result = tool(**tool_args)
            assert "palace write lock timeout" not in json.dumps(result)
        finally:
            open(release, "w").close()
            holder.join(timeout=5)
        assert holder.exitcode == 0

    def test_check_duplicate_not_blocked_by_write_lock(
        self, monkeypatch, config, palace_path, seeded_collection, kg, tmp_path
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        ctx = _get_mp_context()
        ready = str(tmp_path / "holder-ready-check-duplicate")
        release = str(tmp_path / "holder-release-check-duplicate")
        holder = ctx.Process(
            target=_hold_palace_write_lock_process,
            args=(config.palace_path, os.environ["HOME"], ready, release),
        )
        holder.start()
        try:
            assert _wait_for_file(ready), "holder did not acquire lock"
            result = mcp_server.tool_check_duplicate("hello there", threshold=0.5)
            assert "palace write lock timeout" not in json.dumps(result)
        finally:
            open(release, "w").close()
            holder.join(timeout=5)
        assert holder.exitcode == 0


class TestWALFromRealWrites:
    def test_wal_captures_palace_path_from_real_add_drawer(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        _get_collection(palace_path, create=True)

        wal_file = tmp_path / "write_log_add.jsonl"
        monkeypatch.setattr(mcp_server, "_WAL_FILE", wal_file)

        result = mcp_server.tool_add_drawer(
            wing="unit", room="tools", content="Real WAL path check for add_drawer."
        )
        assert result["success"] is True

        entry = json.loads(wal_file.read_text().splitlines()[-1])
        assert entry["operation"] == "add_drawer"
        assert entry["palace_path"] == config.palace_path
        assert entry["params"]["drawer_id"] == result["drawer_id"]

    def test_wal_captures_palace_path_from_real_kg_add(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        wal_file = tmp_path / "write_log_kg_add.jsonl"
        monkeypatch.setattr(mcp_server, "_WAL_FILE", wal_file)

        result = mcp_server.tool_kg_add(subject="A", predicate="p", object="B")
        assert result["success"] is True

        entry = json.loads(wal_file.read_text().splitlines()[-1])
        assert entry["operation"] == "kg_add"
        assert entry["palace_path"] == config.palace_path
        assert entry["params"]["subject"] == "A"

    def test_wal_captures_palace_path_from_real_delete_drawer(
        self, monkeypatch, config, palace_path, seeded_collection, kg, tmp_path
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        wal_file = tmp_path / "write_log_delete.jsonl"
        monkeypatch.setattr(mcp_server, "_WAL_FILE", wal_file)

        result = mcp_server.tool_delete_drawer("drawer_proj_backend_aaa")
        assert result["success"] is True

        entry = json.loads(wal_file.read_text().splitlines()[-1])
        assert entry["operation"] == "delete_drawer"
        assert entry["palace_path"] == config.palace_path
        assert entry["params"]["drawer_id"] == "drawer_proj_backend_aaa"


# ── Write Tools ─────────────────────────────────────────────────────────


class TestWriteTools:
    def test_add_drawer(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_add_drawer

        result = tool_add_drawer(
            wing="test_wing",
            room="test_room",
            content="This is a test memory about Python decorators and metaclasses.",
        )
        assert result["success"] is True
        assert result["wing"] == "test_wing"
        assert result["room"] == "test_room"
        assert result["drawer_id"].startswith("drawer_test_wing_test_room_")

    def test_add_drawer_duplicate_detection(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_add_drawer

        content = "This is a unique test memory about Rust ownership and borrowing."
        result1 = tool_add_drawer(wing="w", room="r", content=content)
        assert result1["success"] is True

        result2 = tool_add_drawer(wing="w", room="r", content=content)
        assert result2["success"] is True
        assert result2["reason"] == "already_exists"

    def test_add_drawer_shared_header_no_collision(self, monkeypatch, config, palace_path, kg):
        """Documents sharing a >100-char header must get distinct IDs (full-content hash)."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_add_drawer

        header = "# ACME Corp Knowledge Base\n**Project:** Alpha | **Team:** Backend | **Status:** Active\n\n"
        doc1 = (
            header
            + "Decision: Use PostgreSQL for primary storage. Rationale: ACID compliance required."
        )
        doc2 = header + "Decision: Use Redis for session caching. Rationale: sub-ms latency needed."

        result1 = tool_add_drawer(wing="work", room="decisions", content=doc1)
        result2 = tool_add_drawer(wing="work", room="decisions", content=doc2)

        assert result1["success"] is True
        assert result2["success"] is True
        assert (
            result1["drawer_id"] != result2["drawer_id"]
        ), "Documents with shared header but different content must have distinct drawer IDs"

    def test_delete_drawer(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_delete_drawer

        result = tool_delete_drawer("drawer_proj_backend_aaa")
        assert result["success"] is True
        assert seeded_collection.count() == 3

    def test_delete_drawer_not_found(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_delete_drawer

        result = tool_delete_drawer("nonexistent_drawer")
        assert result["success"] is False

    def test_check_duplicate(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_check_duplicate

        # Exact match text from seeded_collection should be flagged
        result = tool_check_duplicate(
            "The authentication module uses JWT tokens for session management. "
            "Tokens expire after 24 hours. Refresh tokens are stored in HttpOnly cookies.",
            threshold=0.5,
        )
        assert result["is_duplicate"] is True

        # Unrelated content should not be flagged
        result = tool_check_duplicate(
            "Black holes emit Hawking radiation at the event horizon.",
            threshold=0.99,
        )
        assert result["is_duplicate"] is False

    def test_get_drawer(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_get_drawer

        result = tool_get_drawer("drawer_proj_backend_aaa")
        assert result["drawer_id"] == "drawer_proj_backend_aaa"
        assert result["wing"] == "project"
        assert result["room"] == "backend"
        assert "JWT tokens" in result["content"]

    def test_get_drawer_not_found(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_get_drawer

        result = tool_get_drawer("nonexistent_drawer")
        assert "error" in result

    def test_get_drawer_does_not_leak_absolute_source_file_path(
        self, monkeypatch, config, palace_path, collection, kg
    ):
        """tool_get_drawer must not expose the absolute filesystem path
        that the miners write into ``source_file``. Same threat class as
        the palace_path leak in mempalace_status: in nested-agent or
        multi-server MCP topologies the client is a separate trust
        domain, and the directory layout of the host has no documented
        client-side use. Basename is enough for citation."""
        _patch_mcp_server(monkeypatch, config, kg)

        secret_dir = "/private/home/alice/secret-research/2026"
        absolute_source = f"{secret_dir}/notes.md"
        collection.add(
            ids=["drawer_leak_probe"],
            documents=["verbatim drawer body for leak probe"],
            metadatas=[
                {
                    "wing": "research",
                    "room": "notes",
                    "source_file": absolute_source,
                    "chunk_index": 0,
                    "added_by": "miner",
                    "filed_at": "2026-05-03T00:00:00",
                }
            ],
        )

        from mempalace.mcp_server import tool_get_drawer

        result = tool_get_drawer("drawer_leak_probe")
        assert result["drawer_id"] == "drawer_leak_probe"
        assert result["metadata"]["source_file"] == "notes.md"
        # Defense-in-depth: no field anywhere in the response should
        # contain the absolute path or its parent directory.
        serialized = json.dumps(result)
        assert absolute_source not in serialized
        assert secret_dir not in serialized

    def test_list_drawers(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_list_drawers

        result = tool_list_drawers()
        assert result["count"] == 4
        assert len(result["drawers"]) == 4

    def test_list_drawers_with_wing_filter(
        self, monkeypatch, config, palace_path, seeded_collection, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_list_drawers

        result = tool_list_drawers(wing="project")
        assert result["count"] == 3
        assert all(d["wing"] == "project" for d in result["drawers"])

    def test_list_drawers_with_room_filter(
        self, monkeypatch, config, palace_path, seeded_collection, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_list_drawers

        result = tool_list_drawers(wing="project", room="backend")
        assert result["count"] == 2
        assert all(d["room"] == "backend" for d in result["drawers"])

    def test_list_drawers_pagination(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_list_drawers

        result = tool_list_drawers(limit=2, offset=0)
        assert result["count"] == 2
        assert result["limit"] == 2
        assert result["offset"] == 0

    def test_list_drawers_negative_offset_clamped(
        self, monkeypatch, config, palace_path, seeded_collection, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_list_drawers

        result = tool_list_drawers(offset=-5)
        assert result["offset"] == 0

    def test_update_drawer_content(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_update_drawer, tool_get_drawer

        result = tool_update_drawer(
            "drawer_proj_backend_aaa", content="Updated content about auth."
        )
        assert result["success"] is True

        fetched = tool_get_drawer("drawer_proj_backend_aaa")
        assert fetched["content"] == "Updated content about auth."

    def test_update_drawer_wing_and_room(
        self, monkeypatch, config, palace_path, seeded_collection, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_update_drawer

        result = tool_update_drawer("drawer_proj_backend_aaa", wing="new_wing", room="new_room")
        assert result["success"] is True
        assert result["wing"] == "new_wing"
        assert result["room"] == "new_room"

    def test_update_drawer_not_found(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_update_drawer

        result = tool_update_drawer("nonexistent_drawer", content="hello")
        assert result["success"] is False

    def test_update_drawer_noop(self, monkeypatch, config, palace_path, seeded_collection, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_update_drawer

        result = tool_update_drawer("drawer_proj_backend_aaa")
        assert result["success"] is True
        assert result.get("noop") is True

    def test_write_lock_timeout_returns_structured_error(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        """Direct write tool calls must return JSON-safe lock timeout errors."""
        from mempalace import mcp_server

        result = _run_tool_with_external_write_lock(
            monkeypatch,
            config,
            kg,
            tmp_path,
            mcp_server.tool_add_drawer,
            {"wing": "w", "room": "r", "content": "blocked write"},
        )
        assert result["success"] is False
        assert result["error"] == "palace write lock timeout"
        assert result["palace_path"] == config.palace_path

    def test_delete_drawer_lock_timeout_returns_structured_error(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        from mempalace import mcp_server

        result = _run_tool_with_external_write_lock(
            monkeypatch,
            config,
            kg,
            tmp_path,
            mcp_server.tool_delete_drawer,
            {"drawer_id": "any"},
        )
        assert result["success"] is False
        assert result["error"] == "palace write lock timeout"
        assert result["palace_path"] == config.palace_path

    def test_update_drawer_lock_timeout_returns_structured_error(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        from mempalace import mcp_server

        result = _run_tool_with_external_write_lock(
            monkeypatch,
            config,
            kg,
            tmp_path,
            mcp_server.tool_update_drawer,
            {"drawer_id": "any", "content": "lock test update"},
        )
        assert result["success"] is False
        assert result["error"] == "palace write lock timeout"
        assert result["palace_path"] == config.palace_path

    def test_kg_add_lock_timeout_returns_structured_error(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        from mempalace import mcp_server

        result = _run_tool_with_external_write_lock(
            monkeypatch,
            config,
            kg,
            tmp_path,
            mcp_server.tool_kg_add,
            {"subject": "S", "predicate": "P", "object": "O"},
        )
        assert result["success"] is False
        assert result["error"] == "palace write lock timeout"
        assert result["palace_path"] == config.palace_path

    def test_kg_invalidate_lock_timeout_returns_structured_error(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        from mempalace import mcp_server

        result = _run_tool_with_external_write_lock(
            monkeypatch,
            config,
            kg,
            tmp_path,
            mcp_server.tool_kg_invalidate,
            {"subject": "S", "predicate": "P", "object": "O"},
        )
        assert result["success"] is False
        assert result["error"] == "palace write lock timeout"
        assert result["palace_path"] == config.palace_path

    def test_diary_write_lock_timeout_returns_structured_error(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        from mempalace import mcp_server

        _client, _col = _get_collection(palace_path, create=True)
        del _client
        result = _run_tool_with_external_write_lock(
            monkeypatch,
            config,
            kg,
            tmp_path,
            mcp_server.tool_diary_write,
            {"agent_name": "A", "entry": "E"},
        )
        assert result["success"] is False
        assert result["error"] == "palace write lock timeout"
        assert result["palace_path"] == config.palace_path

    def test_handle_request_lock_timeout_is_tool_result(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        """JSON-RPC stays valid when a mutating tool cannot acquire the lock."""
        _patch_mcp_server(monkeypatch, config, kg)
        monkeypatch.setenv("MEMPALACE_MCP_WRITE_LOCK_TIMEOUT", "0.05")
        from mempalace.mcp_server import handle_request

        ctx = _get_mp_context()
        ready = str(tmp_path / "holder-ready-rpc")
        release = str(tmp_path / "holder-release-rpc")
        holder = ctx.Process(
            target=_hold_palace_write_lock_process,
            args=(config.palace_path, os.environ["HOME"], ready, release),
        )
        holder.start()
        try:
            assert _wait_for_file(ready), "holder did not acquire lock"
            resp = handle_request(
                {
                    "method": "tools/call",
                    "id": 99,
                    "params": {
                        "name": "mempalace_add_drawer",
                        "arguments": {"wing": "w", "room": "r", "content": "blocked write"},
                    },
                }
            )
            body = json.loads(resp["result"]["content"][0]["text"])
            assert body["success"] is False
            assert body["error"] == "palace write lock timeout"
        finally:
            open(release, "w").close()
            holder.join(timeout=5)
        assert holder.exitcode == 0

    def test_mcp_write_waits_for_cross_process_lock(
        self, monkeypatch, config, palace_path, kg, tmp_path
    ):
        """A real MCP write must wait behind a held palace write lock."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.palace import palace_write_lock

        ctx = _get_mp_context()
        started = str(tmp_path / "mcp-writer-started")
        result_queue = ctx.Queue()
        writer = ctx.Process(
            target=_mcp_add_drawer_process,
            args=(config.palace_path, os.environ["HOME"], started, result_queue),
        )

        with palace_write_lock(config.palace_path, blocking=True, timeout=1.0):
            writer.start()
            assert _wait_for_file(started, timeout=10.0), "writer did not reach tool call"
            with pytest.raises(queue.Empty):
                result_queue.get(timeout=0.3)

        writer.join(timeout=20)
        assert writer.exitcode == 0
        result = result_queue.get(timeout=1)
        assert result["success"] is True

        client, col = _get_collection(palace_path, create=False)
        stored = col.get(ids=[result["drawer_id"]])
        del client
        assert stored["ids"] == [result["drawer_id"]]


# ── KG Tools ────────────────────────────────────────────────────────────


class TestKGTools:
    def test_kg_add(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_add

        result = tool_kg_add(
            subject="Alice",
            predicate="likes",
            object="coffee",
            valid_from="2025-01-01",
        )
        assert result["success"] is True

    def test_kg_query(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_query

        result = tool_kg_query(entity="Max")
        assert result["count"] > 0

    def test_kg_invalidate(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_invalidate

        result = tool_kg_invalidate(
            subject="Max",
            predicate="does",
            object="chess",
            ended="2026-03-01",
        )
        assert result["success"] is True
        # Regression #1314: response must echo the actual ended date,
        # not silently drop it and return the literal string "today".
        assert result["ended"] == "2026-03-01"

    def test_kg_add_forwards_valid_to(self, monkeypatch, config, palace_path, kg):
        """Regression #1314 case 1: valid_to must round-trip through kg_add."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_add

        result = tool_kg_add(
            subject="_test_temporal",
            predicate="had_value",
            object="probe",
            valid_from="2026-01-01",
            valid_to="2026-04-28",
        )
        assert result["success"] is True

        facts = kg.query_entity("_test_temporal")
        assert len(facts) == 1
        assert facts[0]["valid_from"] == "2026-01-01"
        assert facts[0]["valid_to"] == "2026-04-28"
        # An already-ended fact must not be reported as still current.
        assert facts[0]["current"] is False

    def test_kg_add_forwards_source_provenance(self, monkeypatch, config, palace_path, kg):
        """Regression #1314 case 3: source_file / source_drawer_id reach storage."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace.mcp_server import tool_kg_add

        result = tool_kg_add(
            subject="operating-verb",
            predicate="candidate",
            object="husbandry",
            valid_from="2026-04-28",
            source_closet="closet-42",
            source_file="docs/decisions.md",
            source_drawer_id="drawer_abc123",
        )
        assert result["success"] is True

        triple_id = result["triple_id"]
        # Read raw row to verify all provenance columns persisted.
        with kg._lock:
            row = (
                kg._conn()
                .execute(
                    "SELECT source_closet, source_file, source_drawer_id FROM triples WHERE id = ?",
                    (triple_id,),
                )
                .fetchone()
            )
        assert row is not None
        assert row["source_closet"] == "closet-42"
        assert row["source_file"] == "docs/decisions.md"
        assert row["source_drawer_id"] == "drawer_abc123"

    def test_kg_invalidate_returns_actual_ended_date(
        self, monkeypatch, config, palace_path, seeded_kg
    ):
        """Regression #1314 case 2: response reports the resolved date, not 'today'."""
        from datetime import date as _date

        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_invalidate

        # Caller-supplied date round-trips into the response.
        explicit = tool_kg_invalidate(
            subject="Max",
            predicate="does",
            object="swimming",
            ended="2026-04-28",
        )
        assert explicit["ended"] == "2026-04-28"

        # Caller-omitted date resolves to today's ISO date — never the
        # literal string "today" the buggy implementation used to return.
        implicit = tool_kg_invalidate(
            subject="Max",
            predicate="loves",
            object="Chess",
        )
        assert implicit["ended"] != "today"
        assert implicit["ended"] == _date.today().isoformat()

    def test_kg_timeline(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_timeline

        result = tool_kg_timeline(entity="Alice")
        assert result["count"] > 0

    def test_kg_stats(self, monkeypatch, config, palace_path, seeded_kg):
        _patch_mcp_server(monkeypatch, config, seeded_kg)
        from mempalace.mcp_server import tool_kg_stats

        result = tool_kg_stats()
        assert result["entities"] >= 4


# ── Diary Tools ─────────────────────────────────────────────────────────


class TestDiaryTools:
    def test_diary_write_and_read(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_diary_write, tool_diary_read

        w = tool_diary_write(
            agent_name="TestAgent",
            entry="Today we discussed authentication patterns.",
            topic="architecture",
        )
        assert w["success"] is True
        # agent_name is normalized to lowercase on write (#1243).
        assert w["agent"] == "testagent"

        r = tool_diary_read(agent_name="TestAgent")
        assert r["total"] == 1
        assert r["entries"][0]["topic"] == "architecture"
        assert "authentication" in r["entries"][0]["content"]

    def test_diary_read_empty(self, monkeypatch, config, palace_path, kg):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_diary_read

        r = tool_diary_read(agent_name="Nobody")
        assert r["entries"] == []

    def test_diary_write_same_second_shared_prefix_no_collision(
        self, monkeypatch, config, palace_path, kg
    ):
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client

        from mempalace import mcp_server

        class FrozenDateTime:
            calls = [
                datetime(2026, 4, 13, 22, 15, 30, 123456),
                datetime(2026, 4, 13, 22, 15, 30, 123457),
            ]
            fallback = datetime(2026, 4, 13, 22, 15, 30, 123457)

            @classmethod
            def now(cls):
                if cls.calls:
                    return cls.calls.pop(0)
                return cls.fallback

        monkeypatch.setattr(mcp_server, "datetime", FrozenDateTime)

        from mempalace.mcp_server import tool_diary_read, tool_diary_write

        entry1 = "A" * 50 + " entry one"
        entry2 = "A" * 50 + " entry two"

        result1 = tool_diary_write(agent_name="TestAgent", entry=entry1, topic="status")
        result2 = tool_diary_write(agent_name="TestAgent", entry=entry2, topic="status")

        assert result1["success"] is True
        assert result2["success"] is True
        assert result1["entry_id"] != result2["entry_id"]

        read_result = tool_diary_read(agent_name="TestAgent")
        contents = [entry["content"] for entry in read_result["entries"]]
        assert read_result["total"] == 2
        assert entry1 in contents
        assert entry2 in contents

    def test_diary_read_empty_wing_spans_all_wings(self, monkeypatch, config, palace_path, kg):
        """diary_read(wing='') must return entries from every wing this agent
        wrote to. Hooks write to project-derived wings (#659); a reader that
        silos by default wing would never see those entries."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_diary_read, tool_diary_write

        w1 = tool_diary_write(
            agent_name="TestAgent",
            entry="default-wing entry",
            topic="general",
        )
        w2 = tool_diary_write(
            agent_name="TestAgent",
            entry="project-wing entry",
            topic="general",
            wing="wing_someproject",
        )
        assert w1["success"] and w2["success"]

        # Empty wing → return both entries
        r = tool_diary_read(agent_name="TestAgent", wing="")
        assert r["total"] == 2
        contents = {e["content"] for e in r["entries"]}
        assert "default-wing entry" in contents
        assert "project-wing entry" in contents

        # Explicit wing → return only that wing's entries
        r_scoped = tool_diary_read(agent_name="TestAgent", wing="wing_someproject")
        assert r_scoped["total"] == 1
        assert r_scoped["entries"][0]["content"] == "project-wing entry"

    def test_diary_read_case_insensitive_agent(self, monkeypatch, config, palace_path, kg):
        """Regression for #1243: diary_read must be case-insensitive over
        agent_name. Writing as "Claude" and reading as "claude" (or vice
        versa) must surface the same entries — sanitize_name preserved
        case, which silently dropped reads when the agent name's casing
        differed from the write."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace.mcp_server import tool_diary_read, tool_diary_write

        # Write as "Claude" → read as "claude" should match.
        w1 = tool_diary_write(
            agent_name="Claude",
            entry="entry written as Claude",
            topic="general",
        )
        assert w1["success"]

        r1 = tool_diary_read(agent_name="claude")
        assert "entries" in r1, r1
        contents1 = {e["content"] for e in r1["entries"]}
        assert "entry written as Claude" in contents1

        # Write as "CLAUDE" → read as "Claude" should also match the
        # same agent. After normalization both writes target the same
        # lowercase agent identity, so both entries are returned.
        w2 = tool_diary_write(
            agent_name="CLAUDE",
            entry="entry written as CLAUDE",
            topic="general",
        )
        assert w2["success"]

        r2 = tool_diary_read(agent_name="Claude")
        contents2 = {e["content"] for e in r2["entries"]}
        assert "entry written as Claude" in contents2
        assert "entry written as CLAUDE" in contents2

        # The stored agent metadata is the lowercase form, and the
        # default wing is derived from that lowercase form too.
        assert w1["agent"] == "claude"
        assert w2["agent"] == "claude"


# ── Cache Invalidation (inode/mtime) ──────────────────────────────────


class TestCacheInvalidation:
    """Tests for _get_collection inode/mtime cache invalidation logic."""

    def test_mtime_change_invalidates_cache(self, monkeypatch, config, palace_path, kg):
        """When mtime changes, the cached collection should be replaced."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        # Create a real collection so _get_collection succeeds
        _client, _col = _get_collection(palace_path, create=True)
        del _client

        # Prime the cache
        col1 = mcp_server._get_collection()
        assert col1 is not None

        # Simulate an external write changing the mtime
        old_mtime = mcp_server._palace_db_mtime
        monkeypatch.setattr(mcp_server, "_palace_db_mtime", old_mtime - 10.0)

        # _get_collection should detect the mtime drift and reconnect
        col2 = mcp_server._get_collection()
        assert col2 is not None

    def test_inode_change_invalidates_cache(self, monkeypatch, config, palace_path, kg):
        """When inode changes (file replaced), the cached collection should be replaced."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        _client, _col = _get_collection(palace_path, create=True)
        del _client

        # Prime the cache
        col1 = mcp_server._get_collection()
        assert col1 is not None

        # Simulate a rebuild that changes the inode
        monkeypatch.setattr(mcp_server, "_palace_db_inode", 99999)

        col2 = mcp_server._get_collection()
        assert col2 is not None

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="Windows holds chroma.sqlite3 open while the client is cached, blocking os.remove",
    )
    def test_missing_db_invalidates_cache(self, monkeypatch, config, palace_path, kg):
        """When chroma.sqlite3 disappears, a cached collection should be invalidated."""
        _patch_mcp_server(monkeypatch, config, kg)
        import os
        from mempalace import mcp_server

        _client, _col = _get_collection(palace_path, create=True)
        del _client

        # Prime the cache
        col1 = mcp_server._get_collection()
        assert col1 is not None
        assert mcp_server._collection_cache is not None

        # Delete the DB file to simulate a rebuild in progress
        db_file = os.path.join(palace_path, "chroma.sqlite3")
        if os.path.isfile(db_file):
            os.remove(db_file)

        # Cache should be invalidated; _get_collection returns None
        # because the backend can't open a missing DB without create=True
        mcp_server._get_collection()
        # The key assertion: the old cached collection was dropped
        assert mcp_server._palace_db_inode == 0
        assert mcp_server._palace_db_mtime == 0.0

    def test_reconnect_reports_failure_when_no_palace(self, monkeypatch, config, kg):
        """tool_reconnect should report failure when no collection is available."""
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        # Make _get_collection always return None
        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=False: None)

        result = mcp_server.tool_reconnect()
        assert result["success"] is False
        assert "No palace found" in result["message"]
        assert result["drawers"] == 0

    def test_reconnect_reports_success(self, monkeypatch, config, palace_path, kg):
        """tool_reconnect should report success with drawer count."""
        _patch_mcp_server(monkeypatch, config, kg)
        _client, _col = _get_collection(palace_path, create=True)
        del _client
        from mempalace import mcp_server

        result = mcp_server.tool_reconnect()
        assert result["success"] is True
        assert "Reconnected" in result["message"]
        assert isinstance(result["drawers"], int)

    def test_reconnect_closes_search_backend_cache(self, monkeypatch, config, palace_path, kg):
        """Reconnect must clear both MCP and searcher Chroma clients.

        ``tool_search`` routes through searcher.py, which owns a separate
        ChromaBackend cache from mcp_server._get_collection(). Clearing only
        the MCP cache leaves filtered search vulnerable to stale HNSW IDs after
        external writes.
        """
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        monkeypatch.setattr(mcp_server, "_get_collection", lambda create=False: _StubCollection())
        closed = []
        monkeypatch.setattr(mcp_server, "_close_search_backend_cache", lambda: closed.append(True))

        result = mcp_server.tool_reconnect()

        assert result["success"] is True
        assert closed == [True]

    def test_get_collection_create_true_avoids_get_or_create_on_reopen(
        self, monkeypatch, config, palace_path, kg
    ):
        """Regression for the MCP-server half of #1262.

        ChromaDB 1.5.x's Rust bindings SIGSEGV when
        ``client.get_or_create_collection`` is called with metadata that
        differs from the collection's stored metadata. The Stop hook
        path (``tool_diary_write`` -> ``_get_collection(create=True)``)
        was reaching that codepath on every session-end; #1262 fixed
        the equivalent crash class in ``ChromaBackend`` but left this
        site untouched. ``_get_collection(create=True)`` must call
        ``client.get_collection`` first and only fall back to
        ``client.create_collection`` when the collection does not yet
        exist on disk.
        """
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        col1 = mcp_server._get_collection(create=True)
        assert col1 is not None

        client = mcp_server._client_cache
        assert client is not None

        # Patch at the class level — chromadb's mtime-change detection
        # may rebuild the client between calls, so an instance-level
        # spy would not survive.
        client_cls = type(client)
        calls: list[tuple] = []

        def _spy(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError(
                "get_or_create_collection must not be called on reopen "
                "(SIGSEGV path on metadata mismatch)"
            )

        monkeypatch.setattr(client_cls, "get_or_create_collection", _spy)
        mcp_server._collection_cache = None

        col2 = mcp_server._get_collection(create=True)
        assert col2 is not None
        assert calls == [], f"get_or_create_collection was called: {calls}"

    def test_get_collection_passes_embedding_function(self, monkeypatch, config, palace_path, kg):
        """Regression for #1299.

        ``mcp_server._get_collection`` must pass ``embedding_function=`` into
        both ``client.get_collection`` and ``client.create_collection``,
        mirroring ``ChromaBackend.get_collection``. Without it, ChromaDB 1.x
        falls back to its built-in ``DefaultEmbeddingFunction`` (whose lazy
        ONNX provider selection has SIGSEGV'd on python 3.14 + Apple Silicon),
        and writers/readers can disagree with the miner about which EF is
        bound to the collection. The miner / Stop hook ingest path routes
        through ``ChromaBackend.get_collection`` which does this correctly;
        the MCP server must match.
        """
        _patch_mcp_server(monkeypatch, config, kg)
        from mempalace import mcp_server

        client = mcp_server._get_client()
        client_cls = type(client)
        captured: dict[str, list[dict]] = {"get": [], "create": []}
        real_get = client_cls.get_collection
        real_create = client_cls.create_collection

        def _spy_get(self, name, **kwargs):
            captured["get"].append(dict(kwargs))
            return real_get(self, name, **kwargs)

        def _spy_create(self, name, **kwargs):
            captured["create"].append(dict(kwargs))
            return real_create(self, name, **kwargs)

        monkeypatch.setattr(client_cls, "get_collection", _spy_get)
        monkeypatch.setattr(client_cls, "create_collection", _spy_create)
        mcp_server._collection_cache = None

        col = mcp_server._get_collection(create=True)
        assert col is not None

        all_calls = captured["get"] + captured["create"]
        assert all_calls, "expected get_collection or create_collection to be called"
        for kwargs in all_calls:
            assert (
                "embedding_function" in kwargs
            ), f"missing embedding_function= in chromadb call: {kwargs}"
            assert kwargs["embedding_function"] is not None

        # Same expectation on the create=False (cache-miss) reopen path.
        mcp_server._collection_cache = None
        captured["get"].clear()
        captured["create"].clear()
        col2 = mcp_server._get_collection()
        assert col2 is not None
        assert captured["get"], "expected get_collection on cache-miss reopen"
        for kwargs in captured["get"]:
            assert "embedding_function" in kwargs
            assert kwargs["embedding_function"] is not None
