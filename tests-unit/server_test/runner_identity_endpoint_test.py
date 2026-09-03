import asyncio
import sys

import pytest

_original_argv = sys.argv
sys.argv = ["pytest", "--cpu"]
from comfy import options

options.enable_args_parsing()
import server

sys.argv = _original_argv


@pytest.fixture
def system_stats_app(monkeypatch, tmp_path):
    cpu = server.comfy.model_management.torch.device("cpu")
    monkeypatch.setattr(server.comfy.model_management, "get_torch_device", lambda: cpu)
    monkeypatch.setattr(
        server.comfy.model_management, "get_all_torch_devices", lambda: [cpu]
    )
    monkeypatch.setattr(
        server.comfy.model_management,
        "get_total_memory",
        lambda *args, **kwargs: (100, 100) if kwargs.get("torch_total_too") else 100,
    )
    monkeypatch.setattr(
        server.comfy.model_management,
        "get_free_memory",
        lambda *args, **kwargs: (50, 50) if kwargs.get("torch_free_too") else 50,
    )
    monkeypatch.setattr(
        server.comfy.model_management, "get_torch_device_name", lambda device: "cpu"
    )
    monkeypatch.setattr(
        server.FrontendManager, "get_required_frontend_version", lambda: None
    )
    monkeypatch.setattr(
        server.FrontendManager, "get_installed_templates_version", lambda: "0.3.0"
    )
    monkeypatch.setattr(
        server.FrontendManager, "get_required_templates_version", lambda: None
    )
    monkeypatch.setattr(
        server.FrontendManager, "get_comfy_package_versions", lambda: {}
    )
    monkeypatch.setattr(server.FrontendManager, "template_asset_handler", lambda: None)
    monkeypatch.setattr(server.FrontendManager, "embedded_docs_path", lambda: None)
    monkeypatch.setattr(
        server.FrontendManager, "init_frontend", lambda version: str(tmp_path)
    )
    prompt_server = server.PromptServer(asyncio.new_event_loop())
    prompt_server.add_routes()
    return prompt_server.app


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("commit", "source_state"),
    [
        ("0123456789abcdef0123456789abcdef01234567", "clean"),
        (None, "unknown"),
        (None, "dirty"),
    ],
)
async def test_system_stats_returns_runner_identity(
    aiohttp_client, system_stats_app, monkeypatch, commit, source_state
):
    async def identity():
        return {"commit": commit, "source_state": source_state}

    monkeypatch.setattr(server, "get_runner_identity_async", identity)
    client = await aiohttp_client(system_stats_app)

    response = await client.get("/system_stats")
    payload = await response.json()
    identity_payload = payload["system"]

    assert response.status == 200
    assert identity_payload["comfyui_commit"] == commit
    assert identity_payload["comfyui_source_state"] == source_state
    assert set(identity_payload) >= {"comfyui_commit", "comfyui_source_state"}
    assert not any(
        token in str(identity_payload).lower()
        for token in ("remote", "branch", "secret")
    )
