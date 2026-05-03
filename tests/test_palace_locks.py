"""Tests for mine_palace_lock — the per-palace non-blocking mine guard.

Covers the fix for the runaway mine fan-out described alongside issues
#974 and #965: if N copies of `mempalace mine` are spawned concurrently
against the same palace, they must collapse to a single runner rather
than queue as waiters that will drive parallel HNSW inserts. Mines
against *different* palaces must still be free to run in parallel.
"""

from __future__ import annotations

import multiprocessing
import os
import sys
import unittest.mock
import time

import pytest

from mempalace.palace import (
    MineAlreadyRunning,
    PalaceWriteLockTimeout,
    mine_global_lock,
    mine_palace_lock,
    palace_write_lock,
)


def _get_mp_context():
    """Pick a start method that works on every CI runner.

    `fork` is cheaper (no re-import) but is unavailable on Windows, so we fall
    back to `spawn` there. `spawn` inherits ``os.environ`` (including the
    monkeypatched ``HOME``) and re-imports the ``mempalace`` package in the
    child, which is sufficient for the lock-file semantics exercised here.
    """
    start_method = "spawn" if os.name == "nt" else "fork"
    return multiprocessing.get_context(start_method)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hold_lock(palace_path: str, ready_flag: str, release_flag: str) -> int:
    """Acquire mine_palace_lock, signal readiness, wait for release flag.

    Returns 0 if we acquired the lock, 1 if MineAlreadyRunning was raised.
    Runs in a child process for true cross-process locking semantics.
    """
    try:
        with mine_palace_lock(palace_path):
            # Tell the parent we hold the lock
            open(ready_flag, "w").close()
            # Wait until parent tells us to release
            for _ in range(500):
                if os.path.exists(release_flag):
                    return 0
                time.sleep(0.01)
            return 0
    except MineAlreadyRunning:
        return 1


def _hold_write_lock(palace_path: str, ready_flag: str, release_flag: str) -> None:
    """Acquire palace_write_lock, signal readiness, wait for release flag."""
    with palace_write_lock(palace_path, blocking=True, timeout=5.0, purpose="test_holder"):
        open(ready_flag, "w").close()
        for _ in range(500):
            if os.path.exists(release_flag):
                return
            time.sleep(0.01)


def _blocking_write_lock_waiter(palace_path: str, started_flag: str, acquired_flag: str) -> None:
    """Block on palace_write_lock until the holder releases it."""
    open(started_flag, "w").close()
    with palace_write_lock(palace_path, blocking=True, timeout=5.0, purpose="test_waiter"):
        open(acquired_flag, "w").close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_acquire_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with mine_palace_lock(str(tmp_path / "palace")):
        pass  # should not raise


def test_lock_reusable_after_release(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    with mine_palace_lock(palace):
        pass
    # Re-acquire must succeed now that the previous holder released
    with mine_palace_lock(palace):
        pass


def test_palace_write_lock_single_acquire_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with palace_write_lock(str(tmp_path / "palace")) as resolved:
        assert resolved.endswith("palace")


def test_palace_write_lock_nonblocking_fails_when_held(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    with palace_write_lock(palace):
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(palace, blocking=False):
                pytest.fail("second non-blocking acquire should fail")


def test_palace_write_lock_timeout_when_held(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    with palace_write_lock(palace):
        start = time.monotonic()
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(palace, blocking=True, timeout=0.05):
                pytest.fail("blocking acquire should time out")
        assert time.monotonic() - start < 1.0


def test_palace_write_lock_timeout_zero_fails_immediately(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    with palace_write_lock(palace):
        start = time.monotonic()
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(palace, blocking=True, timeout=0):
                pytest.fail("blocking with timeout 0 should fail immediately")
        assert time.monotonic() - start < 0.5


def test_palace_write_lock_negative_timeout_treated_as_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    with palace_write_lock(palace):
        start = time.monotonic()
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(palace, blocking=True, timeout=-1):
                pytest.fail("negative timeout should be clamped to zero")
        assert time.monotonic() - start < 0.5


def test_palace_write_lock_uses_env_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MEMPALACE_MCP_WRITE_LOCK_TIMEOUT", "0.05")
    palace = str(tmp_path / "palace")
    with palace_write_lock(palace):
        start = time.monotonic()
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(palace, blocking=True):
                pytest.fail("blocking acquire should use env timeout")
        assert time.monotonic() - start < 1.0


def test_palace_write_lock_very_short_timeout_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    with palace_write_lock(palace):
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(palace, blocking=True, timeout=0.001):
                pytest.fail("very short timeout should still raise when lock is held")


def test_palace_write_lock_conflicts_with_mine_lock(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    with mine_palace_lock(palace):
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(palace, blocking=False):
                pytest.fail("write lock should share mine lock key")


def test_palace_write_lock_timeout_value_from_env_garbage(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MEMPALACE_MCP_WRITE_LOCK_TIMEOUT", "not_a_number")
    from mempalace.palace import _palace_lock_timeout

    assert _palace_lock_timeout() == 60.0


def test_palace_write_lock_timeout_value_from_env_empty_uses_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEMPALACE_MCP_WRITE_LOCK_TIMEOUT", raising=False)
    from mempalace.palace import _palace_lock_timeout

    assert _palace_lock_timeout() == 60.0


@pytest.mark.skipif(sys.platform == "win32", reason="mocks msvcrt on non-Windows")
def test_windows_nonblocking_acquire_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    fake_msvcrt = unittest.mock.MagicMock()
    fake_msvcrt.LK_NBLCK = 1
    fake_msvcrt.LK_UNLCK = 2
    fake_msvcrt.locking.return_value = None

    with unittest.mock.patch.dict("sys.modules", {"msvcrt": fake_msvcrt}):
        monkeypatch.setenv("HOME", str(tmp_path))
        with palace_write_lock(str(tmp_path / "palace"), blocking=False):
            pass

    assert fake_msvcrt.locking.called


@pytest.mark.skipif(sys.platform == "win32", reason="mocks msvcrt on non-Windows")
def test_windows_nonblocking_acquire_fails_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    fake_msvcrt = unittest.mock.MagicMock()
    fake_msvcrt.LK_NBLCK = 1
    fake_msvcrt.LK_UNLCK = 2
    fake_msvcrt.locking.side_effect = OSError("simulated lock failure")

    with unittest.mock.patch.dict("sys.modules", {"msvcrt": fake_msvcrt}):
        monkeypatch.setenv("HOME", str(tmp_path))
        with pytest.raises(PalaceWriteLockTimeout):
            with palace_write_lock(str(tmp_path / "palace"), blocking=False):
                pytest.fail("non-blocking acquire should raise")

    assert fake_msvcrt.locking.called


@pytest.mark.skipif(sys.platform == "win32", reason="mocks msvcrt on non-Windows")
def test_windows_blocking_acquire_retries_then_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    fake_msvcrt = unittest.mock.MagicMock()
    fake_msvcrt.LK_NBLCK = 1
    fake_msvcrt.LK_UNLCK = 2
    fake_msvcrt.locking.side_effect = [
        OSError("first failure"),
        OSError("second failure"),
        None,
        None,
    ]

    with unittest.mock.patch.dict("sys.modules", {"msvcrt": fake_msvcrt}):
        monkeypatch.setenv("HOME", str(tmp_path))
        with palace_write_lock(str(tmp_path / "palace"), blocking=True, timeout=1.0):
            pass

    lock_calls = [
        call for call in fake_msvcrt.locking.call_args_list if call.args[1] == fake_msvcrt.LK_NBLCK
    ]
    unlock_calls = [
        call for call in fake_msvcrt.locking.call_args_list if call.args[1] == fake_msvcrt.LK_UNLCK
    ]
    assert len(lock_calls) >= 2
    assert len(unlock_calls) >= 1


@pytest.mark.skipif(sys.platform == "win32", reason="mocks msvcrt on non-Windows")
def test_windows_release_calls_unlock(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    fake_msvcrt = unittest.mock.MagicMock()
    fake_msvcrt.LK_NBLCK = 1
    fake_msvcrt.LK_UNLCK = 2
    fake_msvcrt.locking.return_value = None

    with unittest.mock.patch.dict("sys.modules", {"msvcrt": fake_msvcrt}):
        monkeypatch.setenv("HOME", str(tmp_path))
        with palace_write_lock(str(tmp_path / "palace"), blocking=False):
            pass

    args = [call.args[1] for call in fake_msvcrt.locking.call_args_list]
    assert fake_msvcrt.LK_NBLCK in args
    assert fake_msvcrt.LK_UNLCK in args


def test_mine_palace_lock_still_raises_mine_already_running(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    with palace_write_lock(palace):
        with pytest.raises(MineAlreadyRunning):
            with mine_palace_lock(palace):
                pytest.fail("mine wrapper should preserve MineAlreadyRunning")


def test_same_palace_serializes_across_processes(tmp_path, monkeypatch):
    """Two processes contending for the same palace: second must be rejected."""
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    ready = str(tmp_path / "ready")
    release = str(tmp_path / "release")

    ctx = _get_mp_context()
    holder = ctx.Process(target=_hold_lock, args=(palace, ready, release))
    holder.start()
    try:
        # Wait for the holder to acquire
        for _ in range(500):
            if os.path.exists(ready):
                break
            time.sleep(0.01)
        assert os.path.exists(ready), "holder failed to acquire lock in time"

        # From the parent, we must not be able to acquire the same palace lock
        with pytest.raises(MineAlreadyRunning):
            with mine_palace_lock(palace):
                pytest.fail("second acquire of same palace should have raised")
    finally:
        open(release, "w").close()
        holder.join(timeout=5)
        assert holder.exitcode == 0


def test_palace_write_lock_blocking_waits_across_processes(tmp_path, monkeypatch):
    """Blocking mode must wait until the cross-process holder releases."""
    monkeypatch.setenv("HOME", str(tmp_path))
    palace = str(tmp_path / "palace")
    ready = str(tmp_path / "ready_holder")
    release = str(tmp_path / "release_holder")
    started = str(tmp_path / "started_waiter")
    acquired = str(tmp_path / "acquired_waiter")

    ctx = _get_mp_context()
    holder = ctx.Process(target=_hold_write_lock, args=(palace, ready, release))
    waiter = ctx.Process(target=_blocking_write_lock_waiter, args=(palace, started, acquired))
    holder.start()
    try:
        for _ in range(500):
            if os.path.exists(ready):
                break
            time.sleep(0.01)
        assert os.path.exists(ready), "holder failed to acquire lock in time"

        waiter.start()
        for _ in range(500):
            if os.path.exists(started):
                break
            time.sleep(0.01)
        assert os.path.exists(started), "waiter failed to start in time"
        time.sleep(0.2)
        assert not os.path.exists(acquired), "waiter acquired before holder released"
    finally:
        open(release, "w").close()
        holder.join(timeout=5)

    waiter.join(timeout=5)
    assert holder.exitcode == 0
    assert waiter.exitcode == 0
    assert os.path.exists(acquired), "waiter did not acquire after release"


def test_different_palaces_dont_conflict(tmp_path, monkeypatch):
    """Mines against different palaces must NOT block each other."""
    monkeypatch.setenv("HOME", str(tmp_path))
    palace_a = str(tmp_path / "palace_a")
    palace_b = str(tmp_path / "palace_b")
    ready = str(tmp_path / "ready_a")
    release = str(tmp_path / "release_a")

    ctx = _get_mp_context()
    holder = ctx.Process(target=_hold_lock, args=(palace_a, ready, release))
    holder.start()
    try:
        for _ in range(500):
            if os.path.exists(ready):
                break
            time.sleep(0.01)
        assert os.path.exists(ready), "holder failed to acquire lock in time"

        # Different palace — must succeed even while palace_a is held
        with mine_palace_lock(palace_b):
            pass  # no exception expected
    finally:
        open(release, "w").close()
        holder.join(timeout=5)


def test_palace_path_is_normalized(tmp_path, monkeypatch):
    """Relative and absolute forms of the same path must use the same lock."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    os.makedirs(tmp_path / "palace", exist_ok=True)
    absolute = str(tmp_path / "palace")
    relative = "palace"

    # Hold the lock with the absolute form; attempting to re-acquire with
    # the relative form (which resolves to the same absolute path) must fail.
    with mine_palace_lock(absolute):
        with pytest.raises(MineAlreadyRunning):
            with mine_palace_lock(relative):
                pytest.fail("normalized path collision should have raised")


def test_palace_path_tilde_and_expanded_home_conflict(tmp_path, monkeypatch):
    """Tilde and expanded-home forms of the same path must use one lock."""
    monkeypatch.setenv("HOME", str(tmp_path))
    os.makedirs(tmp_path / "palace", exist_ok=True)
    tilde_form = os.path.join("~", "palace")
    expanded = str(tmp_path / "palace")

    with mine_palace_lock(tilde_form):
        with pytest.raises(MineAlreadyRunning):
            with mine_palace_lock(expanded):
                pytest.fail("tilde path collision should have raised")


def test_mine_global_lock_is_alias_for_back_compat(tmp_path, monkeypatch):
    """Old callers of `mine_global_lock` should still work."""
    monkeypatch.setenv("HOME", str(tmp_path))
    assert mine_global_lock is mine_palace_lock
    with mine_global_lock(str(tmp_path / "palace")):
        pass  # the alias accepts the same palace_path argument
