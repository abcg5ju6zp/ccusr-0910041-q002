"""Tests for recoverable rename behaviour.

The rename of a file and its checkpoints must leave a *recoverable*
consistent state when something fails halfway:

- a checkpoint-store outage must not move the content while checkpoints are
  still attached to the old name;
- a failure while moving the content must roll the checkpoint moves back;
- if the rollback itself fails, repeating the rename must converge;
- an interrupted rename (content at the new path, checkpoints at the old)
- retried after the fact must be completed rather than rejected;
- rename events must only describe the final, committed state.
"""

import errno
import inspect
import os
import shutil

import pytest
from jupyter_core.utils import ensure_async
from tornado.web import HTTPError

from jupyter_server.services.contents.checkpoints import Checkpoints
from jupyter_server.services.contents.filecheckpoints import FileCheckpoints
from jupyter_server.services.contents.filemanager import (
    AsyncFileContentsManager,
    FileContentsManager,
)

from ...utils import expected_http_error

managers = [FileContentsManager, AsyncFileContentsManager]


@pytest.fixture(params=managers)
def cm(request, tmp_path):
    """项目内部接口说明。"""
    return request.param(root_dir=str(tmp_path))


def _is_async(cm):
    """项目内部接口说明。"""
    return inspect.iscoroutinefunction(cm.checkpoints.move_all_checkpoints)


async def _new_nb(cm, path):
    """项目内部接口说明。"""
    model = await ensure_async(cm.new_untitled(path="", type="notebook"))
    if model["path"] != path:
        await ensure_async(cm.rename(model["path"], path))
    return path


def _checkpoint_file(cm, path):
    """项目内部接口说明。"""
    return cm.checkpoints.checkpoint_path("checkpoint", path)


# ---------------------------------------------------------------------------
# Normal behaviour
# ---------------------------------------------------------------------------


async def test_rename_moves_checkpoint(cm):
    path = await _new_nb(cm, "nb.ipynb")
    await ensure_async(cm.create_checkpoint(path))
    assert os.path.isfile(_checkpoint_file(cm, path))

    await ensure_async(cm.rename(path, "renamed.ipynb"))

    assert not await ensure_async(cm.file_exists(path))
    assert await ensure_async(cm.file_exists("renamed.ipynb"))
    assert await ensure_async(cm.list_checkpoints(path)) == []
    listed = await ensure_async(cm.list_checkpoints("renamed.ipynb"))
    assert [c["id"] for c in listed] == ["checkpoint"]
    assert os.path.isfile(_checkpoint_file(cm, "renamed.ipynb"))
    assert not os.path.exists(_checkpoint_file(cm, path))


async def test_rename_without_checkpoint_still_works(cm):
    # A plain file (no checkpoints) renames just like a notebook.
    model = {"type": "file", "format": "text", "content": "hello"}
    await ensure_async(cm.save(model, "plain.txt"))
    await ensure_async(cm.rename("plain.txt", "plain2.txt"))
    got = await ensure_async(cm.get("plain2.txt", type="file"))
    assert got["content"] == "hello"


async def test_directory_rename_carries_checkpoint_tree(cm, tmp_path):
    # Checkpoints live *inside* the renamed directory, so they move with it.
    await ensure_async(cm.save({"type": "directory"}, "proj"))
    await ensure_async(
        cm.new_untitled(path="proj", type="notebook")
    )  # creates proj/Untitled.ipynb + checkpoint
    cp_dir = os.path.join(str(tmp_path), "proj", ".ipynb_checkpoints")
    assert os.path.isdir(cp_dir)
    assert os.listdir(cp_dir)  # a checkpoint file is present

    await ensure_async(cm.rename("proj", "project"))

    assert not os.path.exists(os.path.join(str(tmp_path), "proj"))
    new_cp_dir = os.path.join(str(tmp_path), "project", ".ipynb_checkpoints")
    assert os.path.isdir(new_cp_dir)
    assert os.listdir(new_cp_dir)
    listed = await ensure_async(cm.list_checkpoints("project/Untitled.ipynb"))
    assert [c["id"] for c in listed] == ["checkpoint"]


# ---------------------------------------------------------------------------
# Transient checkpoint-store failure
# ---------------------------------------------------------------------------


async def test_checkpoint_failure_leaves_content_in_place(cm):
    path = await _new_nb(cm, "nb.ipynb")
    cp = cm.checkpoints

    calls = {"n": 0}
    real = type(cp).rename_checkpoint

    def flaky_sync(self, checkpoint_id, old_path, new_path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EIO, "checkpoint store briefly unavailable")
        return real(self, checkpoint_id, old_path, new_path)

    async def flaky_async(self, checkpoint_id, old_path, new_path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EIO, "checkpoint store briefly unavailable")
        return await real(self, checkpoint_id, old_path, new_path)

    from unittest import mock

    with mock.patch.object(
        type(cp),
        "rename_checkpoint",
        flaky_async if _is_async(cm) else flaky_sync,
    ):
        with pytest.raises(OSError) as exc:
            await ensure_async(cm.rename(path, "renamed.ipynb"))
        assert exc.value.errno == errno.EIO

    # Nothing moved: content and checkpoint are still at the old path.
    assert await ensure_async(cm.file_exists(path))
    assert not await ensure_async(cm.file_exists("renamed.ipynb"))
    assert os.path.isfile(_checkpoint_file(cm, path))
    assert not os.path.exists(_checkpoint_file(cm, "renamed.ipynb"))

    # A retry (storage recovered) completes the rename cleanly.
    await ensure_async(cm.rename(path, "renamed.ipynb"))
    assert await ensure_async(cm.file_exists("renamed.ipynb"))
    listed = await ensure_async(cm.list_checkpoints("renamed.ipynb"))
    assert [c["id"] for c in listed] == ["checkpoint"]
    assert await ensure_async(cm.list_checkpoints(path)) == []


async def test_resume_interrupted_rename(cm, tmp_path):
    # Reproduce the exact post-failure state the platform found: the content
    # was moved but the checkpoint is still attached to the old name.
    path = await _new_nb(cm, "nb.ipynb")
    old_os = cm._get_os_path(path)
    new_os = cm._get_os_path("renamed.ipynb")

    shutil.move(old_os, new_os)
    assert os.path.isfile(new_os) and not os.path.exists(old_os)
    assert os.path.isfile(_checkpoint_file(cm, path))

    # Retrying the rename must finish it instead of raising a conflict.
    await ensure_async(cm.rename(path, "renamed.ipynb"))

    assert await ensure_async(cm.file_exists("renamed.ipynb"))
    listed = await ensure_async(cm.list_checkpoints("renamed.ipynb"))
    assert [c["id"] for c in listed] == ["checkpoint"]
    assert await ensure_async(cm.list_checkpoints(path)) == []


# ---------------------------------------------------------------------------
# Content-move failure after the checkpoints already moved
# ---------------------------------------------------------------------------


async def test_content_failure_rolls_checkpoints_back(cm):
    path = await _new_nb(cm, "nb.ipynb")

    real_rename_file = type(cm).rename_file
    calls = {"n": 0}

    def fail_once_sync(self, old_path, new_path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EIO, "content store hiccup")
        return real_rename_file(self, old_path, new_path)

    async def fail_once_async(self, old_path, new_path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EIO, "content store hiccup")
        return await real_rename_file(self, old_path, new_path)

    from unittest import mock

    with mock.patch.object(
        type(cm), "rename_file", fail_once_async if _is_async(cm) else fail_once_sync
    ):
        with pytest.raises(OSError) as exc:
            await ensure_async(cm.rename(path, "renamed.ipynb"))
        assert exc.value.errno == errno.EIO

    # The checkpoint move was rolled back: everything is back under the old
    # name and the content never moved.
    assert await ensure_async(cm.file_exists(path))
    assert not await ensure_async(cm.file_exists("renamed.ipynb"))
    assert os.path.isfile(_checkpoint_file(cm, path))
    assert not os.path.exists(_checkpoint_file(cm, "renamed.ipynb"))

    # Retry converges.
    await ensure_async(cm.rename(path, "renamed.ipynb"))
    assert await ensure_async(cm.file_exists("renamed.ipynb"))
    listed = await ensure_async(cm.list_checkpoints("renamed.ipynb"))
    assert [c["id"] for c in listed] == ["checkpoint"]


async def test_content_failure_with_failing_rollback_still_converges(cm):
    path = await _new_nb(cm, "nb.ipynb")
    cp = cm.checkpoints

    # Sequence of rename_checkpoint calls:
    # 1. forward move old -> new during the first rename (succeeds)
    # 2. rollback new -> old after the content move fails (fails)
    # 3. forward move on retry: the checkpoint is already at the new name,
    #    so the idempotent no-op leaves it there (succeeds)
    cp_calls = []
    real = type(cp).rename_checkpoint

    def sync_rename(self, checkpoint_id, old_path, new_path):
        cp_calls.append((old_path, new_path))
        if len(cp_calls) == 2:
            raise OSError(errno.EIO, "rollback store down")
        return real(self, checkpoint_id, old_path, new_path)

    async def async_rename(self, checkpoint_id, old_path, new_path):
        cp_calls.append((old_path, new_path))
        if len(cp_calls) == 2:
            raise OSError(errno.EIO, "rollback store down")
        return await real(self, checkpoint_id, old_path, new_path)

    real_rename_file = type(cm).rename_file
    file_calls = {"n": 0}

    def fail_file_sync(self, old_path, new_path):
        file_calls["n"] += 1
        if file_calls["n"] == 1:
            raise OSError(errno.EIO, "content store hiccup")
        return real_rename_file(self, old_path, new_path)

    async def fail_file_async(self, old_path, new_path):
        file_calls["n"] += 1
        if file_calls["n"] == 1:
            raise OSError(errno.EIO, "content store hiccup")
        return await real_rename_file(self, old_path, new_path)

    from unittest import mock

    with (
        mock.patch.object(
            type(cp), "rename_checkpoint", async_rename if _is_async(cm) else sync_rename
        ),
        mock.patch.object(
            type(cm), "rename_file", fail_file_async if _is_async(cm) else fail_file_sync
        ),
        pytest.raises(OSError),
    ):
        await ensure_async(cm.rename(path, "renamed.ipynb"))

    # Messy intermediate state: content at old path, checkpoint stranded at
    # the new name.  A plain retry must still converge.
    await ensure_async(cm.rename(path, "renamed.ipynb"))
    assert await ensure_async(cm.file_exists("renamed.ipynb"))
    listed = await ensure_async(cm.list_checkpoints("renamed.ipynb"))
    assert [c["id"] for c in listed] == ["checkpoint"]
    assert await ensure_async(cm.list_checkpoints(path)) == []


# ---------------------------------------------------------------------------
# Destination conflicts
# ---------------------------------------------------------------------------


async def test_rename_conflict_does_not_touch_checkpoints(cm):
    path = await _new_nb(cm, "nb.ipynb")
    other = await _new_nb(cm, "other.ipynb")

    with pytest.raises(HTTPError) as exc:
        await ensure_async(cm.rename(path, "other.ipynb"))
    assert expected_http_error(exc, 409)

    # Nothing moved and both checkpoint sets are intact.
    assert await ensure_async(cm.file_exists(path))
    assert await ensure_async(cm.file_exists(other))
    assert os.path.isfile(_checkpoint_file(cm, path))
    assert os.path.isfile(_checkpoint_file(cm, other))


async def test_rename_missing_source_is_404(cm):
    with pytest.raises(HTTPError) as exc:
        await ensure_async(cm.rename("nope.ipynb", "renamed.ipynb"))
    assert expected_http_error(exc, 404)


# ---------------------------------------------------------------------------
# Idempotent single-checkpoint moves
# ---------------------------------------------------------------------------


async def test_rename_checkpoint_is_idempotent(cm):
    path = await _new_nb(cm, "nb.ipynb")
    cp = cm.checkpoints
    old_cp = _checkpoint_file(cm, path)
    new_cp = _checkpoint_file(cm, "renamed.ipynb")

    await ensure_async(cp.rename_checkpoint("checkpoint", path, "renamed.ipynb"))
    assert os.path.isfile(new_cp) and not os.path.exists(old_cp)

    # Repeating the forward move is a no-op.
    await ensure_async(cp.rename_checkpoint("checkpoint", path, "renamed.ipynb"))
    assert os.path.isfile(new_cp)

    # Moving "back" when the old location is empty and the checkpoint lives
    # at the source of the reverse move is also a no-op (used by retries).
    await ensure_async(cp.rename_checkpoint("checkpoint", path, "renamed.ipynb"))
    assert os.path.isfile(new_cp) and not os.path.exists(old_cp)


# ---------------------------------------------------------------------------
# Multiple checkpoints: a failure midway rolls the earlier moves back
# ---------------------------------------------------------------------------


class _MultiCheckpoints(Checkpoints):
    """Checkpoint backend that stores several checkpoints per file."""

    supports_rename_rollback = True

    def __init__(self):
        super().__init__()
        self.store = {}
        self.calls = []

    def _add(self, path, *ids):
        self.store.setdefault(path, set()).update(ids)

    def list_checkpoints(self, path):
        return [{"id": i} for i in sorted(self.store.get(path, ()))]

    def rename_checkpoint(self, checkpoint_id, old_path, new_path):
        self.calls.append((checkpoint_id, old_path, new_path))
        if self.fail_on == (checkpoint_id, old_path, new_path):
            raise OSError(errno.EIO, "transient failure")
        self.store.setdefault(old_path, set()).discard(checkpoint_id)
        self.store.setdefault(new_path, set()).add(checkpoint_id)

    def create_checkpoint(self, contents_mgr, path):
        raise NotImplementedError

    def restore_checkpoint(self, contents_mgr, checkpoint_id, path):
        raise NotImplementedError

    def delete_checkpoint(self, checkpoint_id, path):
        self.store.get(path, set()).discard(checkpoint_id)

    fail_on = None


def test_move_all_checkpoints_rolls_back_partial_failure():
    cps = _MultiCheckpoints()
    cps._add("a.ipynb", "cp1", "cp2", "cp3")
    cps.fail_on = ("cp2", "a.ipynb", "b.ipynb")

    with pytest.raises(OSError):
        cps.move_all_checkpoints("a.ipynb", "b.ipynb")

    # cp1 moved before cp2 failed and must have been rolled back.
    assert cps.store.get("a.ipynb") == {"cp1", "cp2", "cp3"}
    assert not cps.store.get("b.ipynb")

    # Once the storage recovers the whole move succeeds.
    cps.fail_on = None
    moved, rollback = cps.move_all_checkpoints("a.ipynb", "b.ipynb")
    assert moved == ["cp1", "cp2", "cp3"]
    assert cps.store.get("b.ipynb") == {"cp1", "cp2", "cp3"}
    assert not cps.store.get("a.ipynb")
    assert rollback() == []
    assert cps.store.get("a.ipynb") == {"cp1", "cp2", "cp3"}
    assert not cps.store.get("b.ipynb")


# ---------------------------------------------------------------------------
# Backwards compatibility: a custom backend without rollback keeps the old
# content-first ordering.
# ---------------------------------------------------------------------------


class _LegacyCheckpoints(FileCheckpoints):
    """A backend that only implements the classic rename_all hook."""

    supports_rename_rollback = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.order = []

    def rename_all_checkpoints(self, old_path, new_path):
        self.order.append("checkpoints")
        super().rename_all_checkpoints(old_path, new_path)


async def test_legacy_backend_keeps_content_first_order(tmp_path):
    cps = _LegacyCheckpoints()
    cm = FileContentsManager(root_dir=str(tmp_path), checkpoints=cps)

    real = cm.rename_file

    def ordered(old_path, new_path):
        cps.order.append("content")
        return real(old_path, new_path)

    cm.rename_file = ordered
    path = cm.new_untitled(type="notebook")["path"]
    cm.rename(path, "renamed.ipynb")
    assert cps.order == ["content", "checkpoints"]


# ---------------------------------------------------------------------------
# Filesystems without atomic replace
# ---------------------------------------------------------------------------


async def test_rename_works_without_atomic_replace(cm, monkeypatch):
    import jupyter_server.services.contents.filecheckpoints as fc
    import jupyter_server.services.contents.filemanager as fm

    def no_replace(src, dst):
        raise OSError(errno.EINVAL, "atomic replace not supported")

    def no_move(src, dst, *args, **kwargs):
        raise OSError(errno.EINVAL, "atomic move not supported")

    monkeypatch.setattr(fc.os, "replace", no_replace)
    monkeypatch.setattr(fm.shutil, "move", no_move)

    path = await _new_nb(cm, "nb.ipynb")
    await ensure_async(cm.rename(path, "renamed.ipynb"))

    assert await ensure_async(cm.file_exists("renamed.ipynb"))
    listed = await ensure_async(cm.list_checkpoints("renamed.ipynb"))
    assert [c["id"] for c in listed] == ["checkpoint"]


async def test_failed_copy_fallback_cleans_partial_destination(cm, monkeypatch):
    import jupyter_server.services.contents.filemanager as fm

    path = await _new_nb(cm, "nb.ipynb")
    real_copy2 = fm.shutil.copy2
    real_move = fm.shutil.move
    copy_calls = {"n": 0}

    def no_move(src, dst, *args, **kwargs):
        raise OSError(errno.EINVAL, "atomic move not supported")

    def flaky_copy2(src, dst, *args, **kwargs):
        copy_calls["n"] += 1
        if copy_calls["n"] == 1:
            # Pretend the network died mid-copy, but leave a partial file.
            with open(dst, "wb") as f:
                f.write(b"partial")
            raise OSError(errno.EIO, "copy interrupted")
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr(fm.shutil, "move", no_move)
    monkeypatch.setattr(fm.shutil, "copy2", flaky_copy2)

    with pytest.raises(HTTPError) as exc:
        await ensure_async(cm.rename(path, "renamed.ipynb"))
    assert expected_http_error(exc, 500)

    # The partial destination was removed; source and checkpoint untouched.
    assert await ensure_async(cm.file_exists(path))
    assert not os.path.exists(cm._get_os_path("renamed.ipynb"))
    assert os.path.isfile(_checkpoint_file(cm, path))

    # Retry with the storage healthy converges (restore the normal move).
    monkeypatch.setattr(fm.shutil, "move", real_move)
    await ensure_async(cm.rename(path, "renamed.ipynb"))
    assert await ensure_async(cm.file_exists("renamed.ipynb"))


# ---------------------------------------------------------------------------
# Events reflect final facts only
# ---------------------------------------------------------------------------


async def test_event_emitted_only_on_success(cm, monkeypatch):
    path = await _new_nb(cm, "nb.ipynb")
    events = []
    monkeypatch.setattr(cm, "emit", lambda data: events.append(data))

    await ensure_async(cm.rename(path, "renamed.ipynb"))
    assert events == [{"action": "rename", "path": "renamed.ipynb", "source_path": "nb.ipynb"}]


async def test_no_event_when_checkpoint_move_fails(cm, monkeypatch):
    path = await _new_nb(cm, "nb.ipynb")
    cp = cm.checkpoints
    real = type(cp).rename_checkpoint
    events = []
    monkeypatch.setattr(cm, "emit", lambda data: events.append(data))

    def fail_sync(self, checkpoint_id, old_path, new_path):
        raise OSError(errno.EIO, "store down")

    async def fail_async(self, checkpoint_id, old_path, new_path):
        raise OSError(errno.EIO, "store down")

    from unittest import mock

    with (
        mock.patch.object(
            type(cp), "rename_checkpoint", fail_async if _is_async(cm) else fail_sync
        ),
        pytest.raises(OSError),
    ):
        await ensure_async(cm.rename(path, "renamed.ipynb"))

    assert events == []


async def test_failing_event_sink_does_not_fail_rename(cm, monkeypatch):
    path = await _new_nb(cm, "nb.ipynb")

    def boom(data):
        raise RuntimeError("event sink unavailable")

    monkeypatch.setattr(cm, "emit", boom)

    # The rename still completes; the event failure is logged, not raised.
    await ensure_async(cm.rename(path, "renamed.ipynb"))
    assert await ensure_async(cm.file_exists("renamed.ipynb"))
