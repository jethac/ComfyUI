"""Local identity for the ComfyUI core source that is running."""

import asyncio
import ctypes
import os
import queue
import re
import subprocess
import threading
import time
from pathlib import Path


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_MAX_OUTPUT = 64 * 1024
_GIT_TIMEOUT = 2.0
_GIT_EXECUTABLE = "git"


class _WindowsJob:
    def __init__(self, process: subprocess.Popen) -> None:
        self.handle = None
        if os.name != "nt":
            return
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            raise OSError("CreateJobObject failed")

        class IoCounters(ctypes.Structure):
            _fields_ = [("values", ctypes.c_uint64 * 6)]

        class BasicLimitInfo(ctypes.Structure):
            _fields_ = [
                ("per_process_user_time", ctypes.c_int64),
                ("per_job_user_time", ctypes.c_int64),
                ("limit_flags", ctypes.c_uint32),
                ("minimum_working_set_size", ctypes.c_size_t),
                ("maximum_working_set_size", ctypes.c_size_t),
                ("active_process_limit", ctypes.c_uint32),
                ("affinity", ctypes.c_size_t),
                ("priority_class", ctypes.c_uint32),
                ("scheduling_class", ctypes.c_uint32),
            ]

        class LimitInfo(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimitInfo),
                ("io", IoCounters),
                ("process_memory_limit", ctypes.c_size_t),
                ("job_memory_limit", ctypes.c_size_t),
                ("peak_process_memory_used", ctypes.c_size_t),
                ("peak_job_memory_used", ctypes.c_size_t),
            ]

        info = LimitInfo()
        info.basic.limit_flags = 0x2000
        if not kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            kernel32.CloseHandle(handle)
            raise OSError("SetInformationJobObject failed")
        if not kernel32.AssignProcessToJobObject(
            handle, ctypes.c_void_p(process._handle)
        ):
            kernel32.CloseHandle(handle)
            raise OSError("AssignProcessToJobObject failed")
        self.handle = handle

    def close(self) -> None:
        if self.handle:
            ctypes.windll.kernel32.CloseHandle(self.handle)
            self.handle = None


def _source_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if os.name == "nt" and process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            process.kill()
    else:
        try:
            os.killpg(process.pid, 9)
        except (OSError, ProcessLookupError):
            if process.poll() is None:
                process.kill()


def _run_git_bounded(arguments: list[str], root: Path, deadline: float) -> bytes | None:
    if time.monotonic() >= deadline:
        return None
    try:
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(
            [_GIT_EXECUTABLE, *arguments],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            start_new_session=(os.name != "nt"),
            creationflags=creationflags,
        )
    except OSError:
        return None
    try:
        process_job = _WindowsJob(process)
    except OSError:
        process.kill()
        process.wait()
        return None

    chunks: queue.Queue[tuple[str, bytes | None]] = queue.Queue(maxsize=2)
    stop_reader = threading.Event()

    def put_chunk(kind: str, chunk: bytes | None) -> bool:
        while not stop_reader.is_set():
            try:
                chunks.put((kind, chunk), timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def read_output() -> None:
        try:
            assert process.stdout is not None
            total = 0
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    put_chunk("eof", None)
                    return
                total += len(chunk)
                if not put_chunk("data", chunk):
                    return
                if total > _MAX_OUTPUT:
                    put_chunk("overflow", None)
                    return
        except (OSError, ValueError):
            put_chunk("error", None)

    reader = threading.Thread(target=read_output, name="comfy-git-reader")
    reader.start()
    output = bytearray()
    terminal = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                chunk = chunks.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                continue
            kind, chunk = chunk
            if kind != "data":
                terminal = kind
                break
            output.extend(chunk)
            if len(output) > _MAX_OUTPUT:
                break
        if terminal != "eof" or process.poll() is None:
            _terminate_process_tree(process)
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    finally:
        stop_reader.set()
        _terminate_process_tree(process)
        try:
            process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            return None
        process_job.close()
        if process.stdout is not None:
            process.stdout.close()
        reader.join(timeout=0.5)
        if reader.is_alive():
            return None
    if terminal != "eof" or process.returncode != 0 or len(output) > _MAX_OUTPUT:
        return None
    return bytes(output)


def _git_observation(root: Path, deadline: float) -> str | None:
    output = _run_git_bounded(
        [
            "status",
            "--porcelain=v2",
            "--branch",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ],
        root,
        deadline,
    )
    if output is None:
        return None
    try:
        return output.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


def _is_git_root(root: Path, deadline: float) -> bool:
    output = _run_git_bounded(["rev-parse", "--show-toplevel"], root, deadline)
    if output is None:
        return False
    try:
        return Path(output.decode("utf-8", errors="strict").strip()).resolve() == root
    except (OSError, UnicodeDecodeError):
        return False


def get_runner_identity() -> dict[str, str | None]:
    """Return the core commit and source state, without local details."""
    unknown = {"commit": None, "source_state": "unknown"}
    try:
        root = _source_root()
    except OSError:
        return unknown
    deadline = time.monotonic() + _GIT_TIMEOUT
    if not _is_git_root(root, deadline):
        return unknown
    observation = _git_observation(root, deadline)
    if observation is None:
        return unknown

    commit = None
    dirty = False
    for line in observation.splitlines():
        if line.startswith("# branch.oid "):
            commit = line.removeprefix("# branch.oid ").strip()
        elif line and not line.startswith("# "):
            dirty = True
    if commit is None or not _COMMIT_RE.fullmatch(commit):
        return unknown
    return {"commit": commit.lower(), "source_state": "dirty" if dirty else "clean"}


async def get_runner_identity_async() -> dict[str, str | None]:
    """Collect identity off the event loop and drain work before cancellation."""
    task = asyncio.create_task(asyncio.to_thread(get_runner_identity))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.shield(task)
        raise
