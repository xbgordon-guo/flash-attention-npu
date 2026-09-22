# Copyright (c) 2026. ATK custom executor for the Ascend 910 FA4 operator.
#
# 说明
# ----
# FA4 在本仓库中是 PyTorch 扩展（`flash_attn_npu_4`），不是 aclnn 自定义算子，
# 因此按 ATK《自定义执行方式》扩展：继承 `BaseApi` 并注册为 `fa4`。
#   - npu 后端：调用被测算子 flash_attn_func / flash_attn_varlen_func
#   - cpu 后端：用仓库内参考实现 tests/common/attention_ref.py 算 golden
#
# 覆盖三种模式（与 tests/test_flash_attn_npu_v4.py 对齐）：
#   mode 0 : dense BSND           q/k/v = (B, S, H, D)      -> flash_attn_func
#   mode 1 : dense varlen TND     q/k/v = (total, H, D)     -> flash_attn_varlen_func + cu_seqlens
#   mode 2 : paged KV cache TND   k/v = (nblocks, 128, H, D) -> flash_attn_varlen_func + page_table
#
# 入参（与 fa4_full.yaml 的 inputs 一致）：
#   q, k, v                              : tensor（shape 随 mode 变化）
#   mode/batch/seqlen_q/seqlen_k         : int
#   causal                               : bool
#   window_left/window_right             : int（-1 表示无限）
#   num_splits                           : int
#   is_varied                            : bool（varlen q/k 序列长度是否随机）
# nheads/nheads_k/head_dim 由 tensor shape 推导。
#
# 输出：out（fp16/bf16），softmax_lse（fp32）。反向用例只返回 out
#       （FA4 反向不支持 dlse，ATK 会对每个输出求导）。

import os
import sys

import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.common.attention_ref import ref_flash_attention  # noqa: E402
from tests.common.test_utils import (  # noqa: E402
    gather_paged_kv_batch,
    make_block_table,
    make_golden_attention_mask,
    make_padded_varlen_mask,
    make_varlen_seqlens,
    pad_packed_tensor,
)

BLOCK_SIZE = 128
VARLEN_SEED = 1234


def _as_bool(v) -> bool:
    if isinstance(v, torch.Tensor):
        return bool(v.reshape(-1)[0].item())
    return bool(v)


def _as_int(v) -> int:
    if isinstance(v, torch.Tensor):
        return int(v.reshape(-1)[0].item())
    return int(v)


def _seqs(batch, seqlen_q, seqlen_k, is_varied):
    if is_varied:
        return make_varlen_seqlens(batch, seqlen_q, seqlen_k, seed=VARLEN_SEED)
    return [seqlen_q] * batch, [seqlen_k] * batch


def _cum(seqs):
    out = [0]
    total = 0
    for s in seqs:
        total += s
        out.append(total)
    return out


def _masked_golden(q_pad, k_pad, v_pad, scale, atten_mask, dtype, q_valid=None):
    out_ref, lse_ref = ref_flash_attention(
        q_pad, k_pad, v_pad, scale, atten_mask, dtype, upcast=False
    )
    if atten_mask is not None:
        fully_masked = atten_mask.all(dim=-1)
        out_ref = out_ref.clone()
        if fully_masked.dim() == 1:
            out_ref[:, fully_masked] = 0
            lse_ref = lse_ref.clone()
            lse_ref[:, :, fully_masked] = torch.inf
        else:
            out_ref[fully_masked] = 0
            lse_ref = lse_ref.masked_fill(fully_masked[:, None, :], torch.inf)
    if q_valid is None:
        return out_ref, lse_ref
    return out_ref[q_valid], lse_ref.permute(0, 2, 1)[q_valid].transpose(0, 1)


@register("fa4")
class FA4Api(BaseApi):
    """FA4 执行器：npu 跑算子，cpu 算 golden（支持 BSND / TND / paged）。"""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        kw = input_data.kwargs or {}
        q, k, v = kw["q"], kw["k"], kw["v"]
        mode = _as_int(kw["mode"])
        batch = _as_int(kw["batch"])
        seqlen_q = _as_int(kw["seqlen_q"])
        seqlen_k = _as_int(kw["seqlen_k"])
        causal = _as_bool(kw["causal"])
        wl = _as_int(kw["window_left"])
        wr = _as_int(kw["window_right"])
        num_splits = _as_int(kw["num_splits"])
        is_varied = _as_bool(kw["is_varied"])
        is_bwd = bool(getattr(input_data, "require_grad", False))

        nheads = int(q.shape[-2])
        head_dim = int(q.shape[-1])
        scale = 1.0 / (head_dim ** 0.5)

        if is_bwd:
            q = q.requires_grad_(True)
            k = k.requires_grad_(True)
            v = v.requires_grad_(True)

        q_seqs, kv_seqs = _seqs(batch, seqlen_q, seqlen_k, is_varied)

        if self.device == "npu":
            out, lse = self._run_npu(q, k, v, mode, scale, causal, wl, wr, num_splits,
                                     batch, seqlen_q, seqlen_k, q_seqs, kv_seqs)
            return out if is_bwd else (out, lse)

        out_ref, lse_ref = self._run_golden(q, k, v, mode, scale, causal, wl, wr,
                                            batch, seqlen_q, seqlen_k, q_seqs, kv_seqs)
        return out_ref if is_bwd else (out_ref, lse_ref)

    @staticmethod
    def _run_npu(q, k, v, mode, scale, causal, wl, wr, num_splits,
                 batch, seqlen_q, seqlen_k, q_seqs, kv_seqs):
        from flash_attn_npu_4 import flash_attn_func, flash_attn_varlen_func

        if mode == 0:
            return flash_attn_func(
                q, k, v, softmax_scale=scale, causal=causal,
                window_size=(wl, wr), num_splits=num_splits, return_lse=True,
            )
        cu_q = torch.tensor(_cum(q_seqs), dtype=torch.int32, device=q.device)
        cu_k = None
        seqused_k = None
        page_table = None
        if mode == 1:
            cu_k = torch.tensor(_cum(kv_seqs), dtype=torch.int32, device=q.device)
        else:
            seqused_k = torch.tensor(kv_seqs, dtype=torch.int32, device=q.device)
            page_table = make_block_table(batch, seqlen_k, BLOCK_SIZE).to(q.device)
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=seqlen_q, max_seqlen_k=seqlen_k, seqused_k=seqused_k,
            page_table=page_table, softmax_scale=scale, causal=causal,
            window_size=(wl, wr), num_splits=num_splits, return_lse=True,
        )

    @staticmethod
    def _run_golden(q, k, v, mode, scale, causal, wl, wr,
                    batch, seqlen_q, seqlen_k, q_seqs, kv_seqs):
        if mode == 0:
            atten_mask, _, _ = make_golden_attention_mask(seqlen_q, seqlen_k, causal, wl, wr)
            return _masked_golden(q, k, v, scale, atten_mask, q.dtype)

        q_pad = pad_packed_tensor(q, q_seqs, seqlen_q)
        if mode == 1:
            k_pad = pad_packed_tensor(k, kv_seqs, seqlen_k)
            v_pad = pad_packed_tensor(v, kv_seqs, seqlen_k)
        else:
            block_table = make_block_table(batch, seqlen_k, BLOCK_SIZE)
            k_pad, v_pad = gather_paged_kv_batch(k, v, block_table, seqlen_k, BLOCK_SIZE)
        q_valid, _, atten_mask = make_padded_varlen_mask(
            q_seqs, kv_seqs, seqlen_q, seqlen_k, causal, wl, wr
        )
        return _masked_golden(q_pad, k_pad, v_pad, scale, atten_mask, q.dtype, q_valid)
