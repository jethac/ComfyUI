import asyncio
import threading
import sys
import time
from pathlib import Path

import pytest
import yaml

from comfy import runner_identity


VALID_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture(autouse=True)
def git_root(monkeypatch):
    monkeypatch.setattr(runner_identity, "_is_git_root", lambda root, deadline: True)


def mock_observation(monkeypatch, output):
    monkeypatch.setattr(
        runner_identity,
        "_git_observation",
        lambda root, deadline: output,
    )


def test_clean_checkout_returns_exact_commit(monkeypatch):
    mock_observation(monkeypatch, f"# branch.oid {VALID_COMMIT}\n")

    assert runner_identity.get_runner_identity() == {
        "commit": VALID_COMMIT,
        "source_state": "clean",
    }


def test_tracked_change_and_untracked_file_are_dirty(monkeypatch):
    mock_observation(
        monkeypatch,
        f"# branch.oid {VALID_COMMIT}\n1 .M N... 100644 100644 100644 a b server.py\n? private.txt\n",
    )

    assert runner_identity.get_runner_identity()["source_state"] == "dirty"


def test_git_failure_fails_closed_without_error_details(monkeypatch):
    monkeypatch.setattr(runner_identity, "_is_git_root", lambda root, deadline: False)

    assert runner_identity.get_runner_identity() == {
        "commit": None,
        "source_state": "unknown",
    }


def test_source_archive_without_git_is_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr(runner_identity, "_source_root", lambda: tmp_path)
    monkeypatch.setattr(runner_identity, "_is_git_root", lambda root, deadline: False)

    assert runner_identity.get_runner_identity() == {
        "commit": None,
        "source_state": "unknown",
    }


def test_invalid_commit_and_status_failure_are_unknown(monkeypatch):
    mock_observation(monkeypatch, "# branch.oid not-a-commit\n")

    assert runner_identity.get_runner_identity() == {
        "commit": None,
        "source_state": "unknown",
    }


def test_submodule_status_is_dirty(monkeypatch):
    mock_observation(
        monkeypatch,
        f"# branch.oid {VALID_COMMIT}\n1 .M S.M. 160000 160000 160000 a b submodule\n",
    )

    assert runner_identity.get_runner_identity() == {
        "commit": VALID_COMMIT,
        "source_state": "dirty",
    }


def test_contract_has_no_path_or_runtime_metadata(monkeypatch):
    mock_observation(monkeypatch, f"# branch.oid {VALID_COMMIT}\n")

    result = runner_identity.get_runner_identity()
    assert set(result) == {"commit", "source_state"}
    assert not any(
        token in str(result).lower()
        for token in ("path", "remote", "branch", "user", "secret")
    )


def test_nested_source_tree_is_unknown(monkeypatch):
    monkeypatch.setattr(runner_identity, "_is_git_root", lambda root, deadline: False)

    assert runner_identity.get_runner_identity() == {
        "commit": None,
        "source_state": "unknown",
    }


def test_current_source_root_is_exact_git_root(monkeypatch):
    monkeypatch.undo()
    assert runner_identity._is_git_root(
        runner_identity._source_root(), time.monotonic() + 2
    )


def test_timeout_and_oversized_output_fail_closed(monkeypatch):
    monkeypatch.setattr(runner_identity, "_run_git_bounded", lambda *args: None)

    assert runner_identity.get_runner_identity() == {
        "commit": None,
        "source_state": "unknown",
    }


def test_real_helper_terminates_and_bounds_large_child_output(tmp_path):
    script = tmp_path / "writer.py"
    script.write_text(
        "import sys; sys.stdout.write('x' * 33554432); sys.stdout.flush()",
        encoding="utf-8",
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runner_identity, "_GIT_EXECUTABLE", sys.executable)
    start = time.monotonic()
    try:
        output = runner_identity._run_git_bounded([str(script)], Path.cwd(), start + 2)
    finally:
        monkeypatch.undo()

    assert output is None
    assert time.monotonic() - start < 2.5
    assert not any(
        thread.name == "comfy-git-reader" for thread in threading.enumerate()
    )


def test_real_helper_terminates_silent_hanging_child(tmp_path):
    script = tmp_path / "sleeper.py"
    script.write_text("import time; time.sleep(30)", encoding="utf-8")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runner_identity, "_GIT_EXECUTABLE", sys.executable)
    start = time.monotonic()
    try:
        output = runner_identity._run_git_bounded(
            [str(script)], Path.cwd(), start + 0.2
        )
    finally:
        monkeypatch.undo()

    assert output is None
    assert time.monotonic() - start < 1.0
    assert not any(
        thread.name == "comfy-git-reader" for thread in threading.enumerate()
    )


def test_reader_must_reach_eof_after_process_exit(tmp_path):
    script = tmp_path / "delayed.py"
    padding = "x" * (4096 - len(f"# branch.oid {VALID_COMMIT}\\n"))
    script.write_text(
        f"import sys,time; sys.stdout.write('# branch.oid {VALID_COMMIT}\\n{padding}'); sys.stdout.flush(); time.sleep(.2); sys.stdout.write('? dirty.py\\n'); sys.stdout.flush()",
        encoding="utf-8",
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runner_identity, "_GIT_EXECUTABLE", sys.executable)
    try:
        output = runner_identity._run_git_bounded(
            [str(script)], Path.cwd(), time.monotonic() + 2
        )
    finally:
        monkeypatch.undo()

    assert output is not None
    assert output.replace(b"\r\n", b"\n").endswith(b"? dirty.py\n")


def test_process_tree_cleanup_after_parent_exit(tmp_path):
    child = tmp_path / "child.py"
    child.write_text("import time; time.sleep(30)", encoding="utf-8")
    parent = tmp_path / "parent.py"
    parent.write_text(
        f"import subprocess,sys; subprocess.Popen([sys.executable, r'{child}']); sys.exit(0)",
        encoding="utf-8",
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(runner_identity, "_GIT_EXECUTABLE", sys.executable)
    start = time.monotonic()
    try:
        output = runner_identity._run_git_bounded(
            [str(parent)], Path.cwd(), start + 0.3
        )
    finally:
        monkeypatch.undo()

    assert output is None
    assert time.monotonic() - start < 1.5
    assert not any(
        thread.name == "comfy-git-reader" for thread in threading.enumerate()
    )


def test_async_cancellation_drains_worker(monkeypatch):
    finished = threading.Event()

    def slow_identity():
        time.sleep(0.05)
        finished.set()
        return {"commit": VALID_COMMIT, "source_state": "clean"}

    monkeypatch.setattr(runner_identity, "get_runner_identity", slow_identity)

    async def run():
        task = asyncio.create_task(runner_identity.get_runner_identity_async())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert not any(
            thread.name == "comfy-git-reader" for thread in threading.enumerate()
        )

    asyncio.run(run())


def test_async_cancellation_drains_real_process_tree(tmp_path, monkeypatch):
    child = tmp_path / "child.py"
    child.write_text("import time; time.sleep(30)", encoding="utf-8")
    parent = tmp_path / "parent.py"
    parent.write_text(
        f"import subprocess,sys; subprocess.Popen([sys.executable, r'{child}']); sys.exit(0)",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_identity, "_GIT_EXECUTABLE", sys.executable)

    def observe():
        return runner_identity._run_git_bounded(
            [str(parent)], Path.cwd(), time.monotonic() + 2
        )

    monkeypatch.setattr(runner_identity, "get_runner_identity", observe)

    async def run():
        task = asyncio.create_task(runner_identity.get_runner_identity_async())
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not any(
            thread.name == "comfy-git-reader" for thread in threading.enumerate()
        )

    asyncio.run(run())


def test_openapi_declares_identity_contract():
    schema = yaml.safe_load(Path("openapi.yaml").read_text(encoding="utf-8"))
    system = schema["components"]["schemas"]["SystemStatsResponse"]["properties"][
        "system"
    ]
    properties = system["properties"]
    assert properties["comfyui_commit"]["nullable"] is True
    assert properties["comfyui_commit"]["type"] == "string"
    assert properties["comfyui_source_state"]["enum"] == ["clean", "dirty", "unknown"]
    assert "comfyui_commit" in system["required"]
    assert "comfyui_source_state" in system["required"]


def test_system_stats_handler_places_identity_under_system():
    source = Path("server.py").read_text(encoding="utf-8")
    handler = source[
        source.index("async def system_stats") : source.index(
            '@routes.get("/features")'
        )
    ]
    assert "runner_identity = await get_runner_identity_async()" in handler
    assert '"comfyui_commit": runner_identity["commit"]' in handler
    assert '"comfyui_source_state": runner_identity["source_state"]' in handler
