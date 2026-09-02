# `/system_stats` runner identity

The `system.comfyui_commit` field is the full Git commit for the exact
top-level worktree that contains the running ComfyUI core source. The
`system.comfyui_source_state` field is `clean`, `dirty`, or `unknown`.

Tracked changes, untracked files, and submodule changes make the core source
state `dirty`. Missing Git, source archives, an inaccessible or nested source
root, malformed output, command failure, timeout, or oversized output make it
`unknown`. The endpoint does not return paths, remotes, branch names, user
names, environment values, or command diagnostics.

This is a core-source identity only. `custom_nodes`, web extensions, and other
runtime-loaded node packs are outside this authority boundary. Megumi must
validate them with its separate node-pack identity contract before accepting a
runner. A clean core source state is not evidence that those packs are clean.
