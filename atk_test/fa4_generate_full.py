# Copyright (c) 2026. ATK full-coverage case generator for the Ascend 910 FA4 operator.
#
# 说明
# ----
# 直接 import 仓库自带测试 tests/test_flash_attn_npu_v4.py 的四张 shape 表，
# 归一化成 ATK 用例，保证覆盖与原测试一致：
#   test_cases(216) + hd_cases(82) + flash_attn_func_cases(10) + head_size_v_cases(108) = 416
#
# 归一化后每条：(dtype, mode, batch, nheads, nheads_k, seqlen_q, seqlen_k,
#                head_dim, head_dim_v, causal, wl, wr, num_splits, is_varied)
#   mode: 0=dense BSND, 1=dense varlen TND, 2=paged KV TND
#
# 注册两个生成器：
#   fa4_full : 全部 416 条（前向）
#   fa4_bwd  : 反向可测子集（mode=1 且 num_splits<=1，对齐原测试 bwd_supported）
#
# 生成： atk case -f fa4_full.yaml -p fa4_generate_full.py
#        atk case -f fa4_bwd.yaml  -p fa4_generate_full.py

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from atk.case_generator.generator.base_generator import CaseGenerator  # noqa: E402
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY  # noqa: E402
from atk.configs.case_config import CaseConfig  # noqa: E402
from tests.common.test_utils import make_varlen_seqlens  # noqa: E402
from tests.test_flash_attn_npu_v4 import (  # noqa: E402
    flash_attn_func_cases,
    hd_cases,
    head_size_v_cases,
    test_cases,
)

BLOCK_SIZE = 128
VARLEN_SEED = 1234
_DTYPE = {torch.float16: "fp16", torch.bfloat16: "bf16"}


def _mode(layout, cache_mode):
    if layout == "BSND":
        return 0
    return 2 if cache_mode == 1 else 1


def _build_specs():
    specs = []
    for dt, b, h, hk, sq, sk, d, cm, _bs, ic, layout, iv, wl, wr, ns in test_cases:
        specs.append((_DTYPE[dt], _mode(layout, cm), b, h, hk, sq, sk, d, d,
                      ic, wl, wr, ns, bool(iv)))
    for dt, b, h, hk, sq, sk, d, cm, _bs, ic, layout, ns, wl, wr, _sc in hd_cases:
        specs.append((_DTYPE[dt], _mode(layout, cm), b, h, hk, sq, sk, d, d,
                      ic, wl, wr, ns, layout == "TND"))
    for dt, b, h, hk, sq, sk, d, ic, win in flash_attn_func_cases:
        specs.append((_DTYPE[dt], 0, b, h, hk, sq, sk, d, d,
                      ic, win[0], win[1], 0, False))
    for dt, b, h, hk, sq, sk, dqk, dv, cm, _bs, ic, layout, iv, wl, wr, ns in head_size_v_cases:
        specs.append((_DTYPE[dt], _mode(layout, cm), b, h, hk, sq, sk, dqk, dv,
                      ic, wl, wr, ns, bool(iv)))
    return specs


_SPECS = _build_specs()
# 原测试 bwd_supported: layout=TND 且 cache_mode=0 且 num_splits<=1（910 无 SWA 限制）
_BWD_SPECS = [s for s in _SPECS if s[1] == 1 and s[12] <= 1]


def _seqs(b, sq, sk, is_varied):
    if is_varied:
        return make_varlen_seqlens(b, sq, sk, seed=VARLEN_SEED)
    return [sq] * b, [sk] * b


def _shapes(mode, b, h, hk, sq, sk, d, dv, is_varied):
    q_seqs, kv_seqs = _seqs(b, sq, sk, is_varied)
    if mode == 0:
        return ([b, sq, h, d], [b, sk, hk, d], [b, sk, hk, dv])
    total_q = sum(q_seqs)
    if mode == 1:
        total_k = sum(kv_seqs)
        return ([total_q, h, d], [total_k, hk, d], [total_k, hk, dv])
    nblocks = b * ((sk + BLOCK_SIZE - 1) // BLOCK_SIZE)
    return ([total_q, h, d], [nblocks, BLOCK_SIZE, hk, d], [nblocks, BLOCK_SIZE, hk, dv])


def _apply(case_config, specs, counter, deterministic=None):
    inputs = case_config.inputs
    if inputs and isinstance(inputs[0], list):
        inputs = inputs[0]
    if len(inputs) < 12:
        return case_config
    if inputs[0].shape and 0 in inputs[0].shape:
        return case_config

    idx = counter["n"] % len(specs)
    counter["n"] += 1
    dtype, mode, b, h, hk, sq, sk, d, dv, causal, wl, wr, ns, is_varied = specs[idx]
    q_shape, k_shape, v_shape = _shapes(mode, b, h, hk, sq, sk, d, dv, is_varied)

    inputs[0].dtype = dtype
    inputs[0].shape = q_shape
    inputs[0].range_values = [-1, 1]
    inputs[1].dtype = dtype
    inputs[1].shape = k_shape
    inputs[1].range_values = [-1, 1]
    inputs[2].dtype = dtype
    inputs[2].shape = v_shape
    inputs[2].range_values = [-1, 1]

    inputs[3].range_values = mode
    inputs[4].range_values = b
    inputs[5].range_values = sq
    inputs[6].range_values = sk
    inputs[7].range_values = causal
    inputs[8].range_values = wl
    inputs[9].range_values = wr
    inputs[10].range_values = ns
    inputs[11].range_values = is_varied
    if deterministic is not None and len(inputs) >= 13:
        inputs[12].range_values = deterministic
    return case_config


@GENERATOR_REGISTRY.register("fa4_full")
class FA4FullGenerator(CaseGenerator):
    """全部 416 条（前向）。"""

    _counter = {"n": 0}

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        return _apply(case_config, _SPECS, self._counter)


@GENERATOR_REGISTRY.register("fa4_bwd")
class FA4BwdGenerator(CaseGenerator):
    """反向可测子集（mode=1 且 num_splits<=1）。"""

    _counter = {"n": 0}

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        return _apply(case_config, _BWD_SPECS, self._counter)


@GENERATOR_REGISTRY.register("fa4_bwd_det")
class FA4BwdDetGenerator(CaseGenerator):
    """反向确定性子集（同 fa4_bwd，deterministic=True）。"""

    _counter = {"n": 0}

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        return _apply(case_config, _BWD_SPECS, self._counter, deterministic=True)
