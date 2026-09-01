"""Regression tests for Pixal3D projection attention with quantized weights."""

from __future__ import annotations

import torch
from torch import nn

from comfy.ldm.trellis2.model import ProjectAttentionDense, ProjectAttentionSparse


class _FakeQuantLinear(nn.Module):
    """Stand-in for a GGUF/quantized Linear with non-compute storage dtype."""

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__()
        self.weight = torch.zeros(out_features, in_features, dtype=torch.uint8, device=device)
        self._dequant = torch.randn(out_features, in_features, dtype=torch.float32, device=device) * 0.02
        self.bias = torch.zeros(out_features, dtype=torch.float32, device=device) if bias else None
        self.seen_input_dtype = None

    def forward(self, x):
        self.seen_input_dtype = x.dtype
        assert x.dtype.is_floating_point, (
            f"proj_in reached the quantized Linear as {x.dtype}; the block must "
            "not cast the activation to the stored weight dtype"
        )
        weight = self._dequant.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(x, weight, bias)


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


def test_project_attention_dense_quant_safe():
    compute_dtype = torch.bfloat16
    block = ProjectAttentionDense(
        _IdentityCrossDense(), channels=8, proj_in_channels=6, operations=_FakeQuantOps()
    )
    x = torch.randn(2, 3, 8, dtype=compute_dtype)
    proj_in = torch.randn(2, 3, 6, dtype=torch.float32)

    out = block(x, {"global": None, "proj": proj_in})

    assert out.shape == x.shape
    assert out.dtype == compute_dtype
    assert torch.isfinite(out).all()
    assert block.proj_linear.seen_input_dtype.is_floating_point


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
    assert block.proj_linear.seen_input_dtype.is_floating_point


def test_project_attention_dense_tuple_proj():
    compute_dtype = torch.float32
    block = ProjectAttentionDense(
        _IdentityCrossDense(), channels=8, proj_in_channels=6, operations=_FakeQuantOps()
    )
    x = torch.randn(1, 4, 8, dtype=compute_dtype)
    proj_in = (torch.randn(1, 4, 2), torch.randn(1, 4, 4))

    out = block(x, {"global": None, "proj_semantic": proj_in[0], "proj_color": proj_in[1]})

    assert out.shape == x.shape
    assert block.proj_linear.seen_input_dtype == compute_dtype
