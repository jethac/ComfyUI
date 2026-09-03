# ComfyUI Runner Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add privacy-safe live source identity to `/system_stats` so Megumi can accept only an exact clean runner commit.

**Architecture:** Keep identity collection in a small pure helper module. It runs local Git commands against the running source tree, returns only the full commit and a closed set source state, and returns `unknown` when Git or source metadata is unavailable. The endpoint places the immutable identity under `system` without paths, remotes, branches, or process data from the helper.

**Tech Stack:** Python standard library, aiohttp JSON endpoint, pytest.

**Spec:** Approved Task 10 design: immutable local build/commit identity in system metadata.

## Authority Boundary

This contract identifies the exact Git worktree top level that contains the
running ComfyUI core source. It includes tracked files and all untracked files
reported by Git, including submodule changes. It does not claim that
`custom_nodes`, web extensions, or other runtime-loaded node packs are part of
the core source identity. Megumi binds those node packs with its separate
node-pack identity contract; accepting a clean core source state never accepts
unknown or dirty node-pack content.

## Global Constraints

- Return the exact 40-character Git commit when available.
- Return `clean`, `dirty`, or `unknown`; tracked and untracked files make a checkout dirty.
- Fail closed for missing Git, invalid output, command errors, archives, and unknown submodule state.
- Never expose local paths, remotes, branch labels, user names, environment data, or secrets.
- Support Git checkout, installed source archive, missing Git, dirty tracked files, untracked files, submodules, Windows, and Linux.
- Do not add dependencies or internet requests.

---

### Task 1: Add failing identity helper tests

**Files:**
- Create: `tests-unit/runner_identity_test.py`

**Interfaces:**
- Test `comfy.runner_identity.get_runner_identity()` returns `{"commit": str | None, "source_state": "clean" | "dirty" | "unknown"}`.

- [x] **Step 1: Write tests for valid clean and dirty checkout output, including untracked status.**
- [x] **Step 2: Write tests for missing Git, command failure, malformed commit, archive, and privacy/path output.**
- [x] **Step 3: Write tests for deterministic submodule status and Windows/Linux command construction without platform-specific paths in returned data.**
- [x] **Step 4: Run `python -m pytest tests-unit/runner_identity_test.py -q` and verify the new tests fail because the helper was absent.**

### Task 2: Implement the closed identity contract

**Files:**
- Create: `comfy/runner_identity.py`

**Interfaces:**
- Produces `get_runner_identity() -> dict[str, str | None]` with keys `commit` and `source_state`.

- [x] **Step 1: Implement local source-root discovery from the module location without returning the path.**
- [x] **Step 2: Run `git rev-parse --verify HEAD` and accept only a complete 40-hex commit.**
- [x] **Step 3: Run `git status --porcelain=v1 --untracked-files=all`; any output, including untracked files and submodule changes, is `dirty`.**
- [x] **Step 4: Convert missing Git, non-zero commands, malformed output, inaccessible source, and uncertain submodule state to `commit=None, source_state="unknown"` without exceptions or diagnostic data in the result.**
- [x] **Step 5: Run the focused tests and verify they pass.**

### Task 3: Add the identity to `/system_stats`

**Files:**
- Modify: `server.py:686-737`
- Test: `tests-unit/runner_identity_test.py`

**Interfaces:**
- `/system_stats` includes `system.comfyui_commit` and `system.comfyui_source_state` with the helper result.

- [x] **Step 1: Add endpoint schema assertions for exact keys and closed values.**
- [x] **Step 2: Call the helper when building system metadata and copy only its two fields.**
- [x] **Step 3: Assert endpoint metadata contains no path, remote, branch, username, environment, or secret fields from the identity helper.**
- [x] **Step 4: Run focused endpoint/schema tests.**

### Task 4: Verify all gates and prepare handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-09-02-comfyui-runner-identity.md`

- [x] **Step 1: Run `python -m pytest tests-unit/runner_identity_test.py -q`.**
- [x] **Step 2: Run the repository Python test gate selected by ComfyUI CI and record counts; full collection is blocked by the installed comfy_kitchen mismatch.**
- [x] **Step 3: Run formatting/lint checks for changed Python files.**
- [ ] **Step 4: Re-fetch `jethac/master`, verify the base and tree, inspect the final diff for privacy leaks, commit, push the branch, and open a non-draft PR only if the GitHub workflow is available.**
