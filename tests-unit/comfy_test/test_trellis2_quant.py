"""TRELLIS2 / Pixal3D quantized-weight enablement regression tests.

Covers two things the native TRELLIS2 architecture needs so that quantized /
fp8 / GGUF (city96 ComfyUI-GGUF) weights load and run:

1. ``comfy.model_detection`` recognises TRELLIS2 (global) and Pixal3D (proj)
   checkpoints and reports the per-submodel projection config.
2. The Pixal3D projection-attention blocks cast ``proj_in`` to the *running
   compute dtype* rather than the stored weight dtype. Under quantized ops the
   stored ``weight.dtype`` is not a valid compute dtype (e.g. ``uint8`` for a
   GGML tensor, ``float8_e4m3fn`` for fp8 ops); casting the activation to it
   corrupts the tensor. This is the core-side enablement for GGUF loading.
"""

from __future__ import annotations

import torch
from torch import nn

from comfy.cli_args import args

if not torch.cuda.is_available():
    args.cpu = True

from comfy.model_detection import detect_unet_config  # noqa: E402
from comfy.ldm.trellis2.model import (  # noqa: E402
    ProjectAttentionDense,
    ProjectAttentionSparse,
)


# --------------------------------------------------------------------------- #
# model_detection
# --------------------------------------------------------------------------- #
def _trellis2_sd(proj: bool = False, tex: bool = False) -> dict:
    sd = {
        "img2shape.t_embedder.mlp.0.weight": torch.empty(8, 8),
        "structure_model.t_embedder.mlp.0.weight": torch.empty(8, 8),
    }
    if tex:
        sd["shape2txt.t_embedder.mlp.0.weight"] = torch.empty(8, 8)
    if proj:
        sd["img2shape.blocks.0.cross_attn.proj_linear.weight"] = torch.empty(16, 133)
        sd["structure_model.blocks.0.cross_attn.proj_linear.weight"] = torch.empty(16, 133)
    return sd


def test_detect_trellis2_global():
    cfg = detect_unet_config(_trellis2_sd(), "")
    assert cfg["image_model"] == "trellis2"
    assert cfg["init_txt_model"] is False
    assert cfg["txt_only"] is False
    # global attention: no proj config keys emitted
    assert "image_attn_mode_shape" not in cfg


def test_detect_trellis2_texture():
    cfg = detect_unet_config(_trellis2_sd(tex=True), "")
    assert cfg["image_model"] == "trellis2"
    assert cfg["init_txt_model"] is True
    assert cfg["txt_only"] is False


def test_detect_pixal3d_proj():
    cfg = detect_unet_config(_trellis2_sd(proj=True), "")
    assert cfg["image_model"] == "trellis2"
    assert cfg["image_attn_mode_shape"] == "proj"
    assert cfg["proj_in_channels_shape"] == 133
    assert cfg["image_attn_mode_structure"] == "proj"
    assert cfg["proj_in_channels_structure"] == 133


# --------------------------------------------------------------------------- #
# proj-attention quantized-weight tolerance
# --------------------------------------------------------------------------- #
class _FakeQuantLinear(nn.Module):
    """Stand-in for a GGUF/quantized Linear.

    ``.weight.dtype`` advertises a *storage* dtype (uint8) that is NOT a valid
    compute dtype, exactly like a city96 GGMLTensor or an fp8 weight. ``forward``
    dequantizes on the fly and asserts it received a floating-point activation.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        # Stored (quantized) weight advertises a non-compute dtype.
        self.weight = torch.zeros(out_features, in_features, dtype=torch.uint8, device=device)
        self._dequant = torch.randn(out_features, in_features, dtype=torch.float32, device=device) * 0.02
        self.bias = torch.zeros(out_features, dtype=torch.float32, device=device) if bias else None
        self.seen_input_dtype = None

    def forward(self, x):
        self.seen_input_dtype = x.dtype
        assert x.dtype.is_floating_point, (
            f"proj_in reached the quantized Linear as {x.dtype}; the block must "
            f"cast to a floating compute dtype, not the stored weight dtype"
        )
        w = self._dequant.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, w, b)


class _FakeQuantOps:
    Linear = _FakeQuantLinear


class _IdentityCrossDense(nn.Module):
    def forward(self, x, context, transformer_options=None):
        return x


class _FakeSparse:
    def __init__(self, feats):
        self.feats = feats

    def replace(self, feats):
        return _FakeSparse(feats)


class _IdentityCrossSparse(nn.Module):
    def forward(self, x, context, transformer_options=None):
        return x


def test_pre_fix_cast_was_broken():
    """Guard the premise: casting an activation to a uint8 weight dtype (the old
    ``proj_in.to(self.proj_linear.weight.dtype)``) is not a floating dtype."""
    fake = _FakeQuantLinear(4, 8)
    corrupted = torch.randn(2, 4).to(fake.weight.dtype)
    assert corrupted.dtype == torch.uint8
    assert not corrupted.dtype.is_floating_point


def test_project_attention_dense_quant_safe():
    compute_dtype = torch.bfloat16
    block = ProjectAttentionDense(
        _IdentityCrossDense(), channels=8, proj_in_channels=6, operations=_FakeQuantOps()
    )
    x = torch.randn(2, 3, 8, dtype=compute_dtype)
    # proj features arrive in float32 (as NAF/DINO feature maps do); the block
    # must down-cast them to the compute dtype, never to the uint8 weight dtype.
    proj_in = torch.randn(2, 3, 6, dtype=torch.float32)
    out = block(x, {"global": None, "proj": proj_in})
    assert out.shape == x.shape
    assert out.dtype == compute_dtype
    assert torch.isfinite(out).all()
    assert block.proj_linear.seen_input_dtype == compute_dtype


def test_project_attention_sparse_quant_safe():
    compute_dtype = torch.bfloat16
    block = ProjectAttentionSparse(
        _IdentityCrossSparse(), channels=8, proj_in_channels=6, operations=_FakeQuantOps()
    )
    x = _FakeSparse(torch.randn(5, 8, dtype=compute_dtype))
    proj_in = torch.randn(5, 6, dtype=torch.float32)
    out = block(x, {"global": None, "proj": proj_in})
    assert isinstance(out, _FakeSparse)
    assert out.feats.shape == x.feats.shape
    assert out.feats.dtype == compute_dtype
    assert torch.isfinite(out.feats).all()
    assert block.proj_linear.seen_input_dtype == compute_dtype


def test_project_attention_dense_tuple_proj():
    """proj context may arrive as a (semantic, color) tuple that is concatenated."""
    compute_dtype = torch.float32
    block = ProjectAttentionDense(
        _IdentityCrossDense(), channels=8, proj_in_channels=6, operations=_FakeQuantOps()
    )
    x = torch.randn(1, 4, 8, dtype=compute_dtype)
    proj_in = (torch.randn(1, 4, 2), torch.randn(1, 4, 4))
    out = block(x, {"global": None, "proj_semantic": proj_in[0], "proj_color": proj_in[1]})
    assert out.shape == x.shape
    assert block.proj_linear.seen_input_dtype == compute_dtype
