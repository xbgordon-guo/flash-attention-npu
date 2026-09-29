#!/usr/bin/env python3
# Copyright (c) 2026. FA4 前向 batch 不变性测试（逐位一致）。
#
# 对 ATK 生成的前向用例逐条：
#   1. 单独跑样本 s（B=1）-> ref
#   2. 把 s 放进 B=N 的 batch（重复 N 份 / s + 干扰样本），比对 s 的输出与 ref 是否逐位一致
#
# 仅测试，不修改算子源码。
# 用法： python3 fa4_batch_consistency.py [--limit K] [--n 2,4,8] [--pos 0,-1]

import argparse
import json
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch_npu  # noqa: F401,E402
from flash_attn_npu_4 import flash_attn_func, flash_attn_varlen_func  # noqa: E402
from tests.common.test_utils import make_block_table  # noqa: E402

BLOCK_SIZE = 128
DEV = "npu"
_DT = {"fp16": torch.float16, "bf16": torch.bfloat16}
_TARGET_SEED = 1000
_DISTRACTOR_BASE = 5000


def _inp(case, name, key):
    for i in case["inputs"]:
        if i["name"] == name:
            return i.get(key)
    return None


def _attrs(case):
    return {
        "dt": _DT[case["inputs"][0]["dtype"]],
        "mode": int(_inp(case, "mode", "range_values")),
        "sq": int(_inp(case, "seqlen_q", "range_values")),
        "sk": int(_inp(case, "seqlen_k", "range_values")),
        "causal": bool(_inp(case, "causal", "range_values")),
        "wl": int(_inp(case, "window_left", "range_values")),
        "wr": int(_inp(case, "window_right", "range_values")),
        "ns": int(_inp(case, "num_splits", "range_values")),
        "h": int(_inp(case, "q", "shape")[-2]),
        "d": int(_inp(case, "q", "shape")[-1]),
        "hk": int(_inp(case, "k", "shape")[-2]),
        "dv": int(_inp(case, "v", "shape")[-1]),
    }


def _sample(shape, dt, seed):
    torch.manual_seed(seed)
    return torch.randn(*shape, dtype=dt, device=DEV)


def _sample_shapes(a):
    if a["mode"] == 2:
        nblk = (a["sk"] + BLOCK_SIZE - 1) // BLOCK_SIZE
        ksh = (nblk, BLOCK_SIZE, a["hk"], a["d"])
        vsh = (nblk, BLOCK_SIZE, a["hk"], a["dv"])
    else:
        ksh = (a["sk"], a["hk"], a["d"])
        vsh = (a["sk"], a["hk"], a["dv"])
    return (a["sq"], a["h"], a["d"]), ksh, vsh


def _pack(parts, mode):
    return torch.stack(parts, dim=0) if mode == 0 else torch.cat(parts, dim=0)


def _take(out, mode, pos, sq):
    return out[pos] if mode == 0 else out[pos * sq:(pos + 1) * sq]


def _run(a, n, q, k, v):
    scale = 1.0 / (a["d"] ** 0.5)
    if a["mode"] == 0:
        out, _ = flash_attn_func(
            q, k, v, softmax_scale=scale, causal=a["causal"],
            window_size=(a["wl"], a["wr"]), num_splits=a["ns"], return_lse=True,
        )
        return out
    cu_q = torch.tensor([i * a["sq"] for i in range(n + 1)], dtype=torch.int32, device=DEV)
    if a["mode"] == 1:
        cu_k = torch.tensor([i * a["sk"] for i in range(n + 1)], dtype=torch.int32, device=DEV)
        out, _ = flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=a["sq"], max_seqlen_k=a["sk"], softmax_scale=scale,
            causal=a["causal"], window_size=(a["wl"], a["wr"]),
            num_splits=a["ns"], return_lse=True,
        )
        return out
    seqused_k = torch.tensor([a["sk"]] * n, dtype=torch.int32, device=DEV)
    page_table = make_block_table(n, a["sk"], BLOCK_SIZE).to(DEV)
    out, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=None,
        max_seqlen_q=a["sq"], max_seqlen_k=a["sk"], seqused_k=seqused_k,
        page_table=page_table, softmax_scale=scale, causal=a["causal"],
        window_size=(a["wl"], a["wr"]), num_splits=a["ns"], return_lse=True,
    )
    return out


def _cmp(tag, got, ref):
    eq = bool(torch.equal(got, ref))
    md = float((got.float() - ref.float()).abs().max().item()) if got.numel() else 0.0
    return {"tag": tag, "equal": eq, "max_diff": md}


def check_case(case, ns_list, positions):
    a = _attrs(case)
    qsh, ksh, vsh = _sample_shapes(a)
    q1 = _sample(qsh, a["dt"], _TARGET_SEED)
    k1 = _sample(ksh, a["dt"], _TARGET_SEED + 1)
    v1 = _sample(vsh, a["dt"], _TARGET_SEED + 2)
    ref = _take(_run(a, 1, _pack([q1], a["mode"]), _pack([k1], a["mode"]),
                     _pack([v1], a["mode"])), a["mode"], 0, a["sq"]).clone()

    results = []
    for n in ns_list:
        od = _run(a, n, _pack([q1] * n, a["mode"]), _pack([k1] * n, a["mode"]),
                  _pack([v1] * n, a["mode"]))
        for p in range(n):
            results.append(_cmp(f"dup N={n} p={p}", _take(od, a["mode"], p, a["sq"]), ref))
        for p in positions:
            pp = p if p >= 0 else n + p
            if pp < 0 or pp >= n:
                continue
            qs, ks, vs = [], [], []
            for i in range(n):
                if i == pp:
                    qs.append(q1); ks.append(k1); vs.append(v1)
                else:
                    b = _DISTRACTOR_BASE + 10 * i
                    qs.append(_sample(qsh, a["dt"], b))
                    ks.append(_sample(ksh, a["dt"], b + 1))
                    vs.append(_sample(vsh, a["dt"], b + 2))
            om = _run(a, n, _pack(qs, a["mode"]), _pack(ks, a["mode"]), _pack(vs, a["mode"]))
            results.append(_cmp(f"mix N={n} p={pp}", _take(om, a["mode"], pp, a["sq"]), ref))
    return a, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="result/fa4_full/json/all_fa4_full.json")
    ap.add_argument("--n", default="2,4,8")
    ap.add_argument("--pos", default="0,-1")
    ap.add_argument("--modes", default="0,1,2")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    ns_list = [int(x) for x in args.n.split(",")]
    positions = [int(x) for x in args.pos.split(",")]
    modes = {int(x) for x in args.modes.split(",")}

    cases = json.load(open(args.cases))
    end = len(cases) if args.limit <= 0 else min(len(cases), args.start + args.limit)
    total = bad = skipped = 0
    fails = []
    for idx in range(args.start, end):
        case = cases[idx]
        a = _attrs(case)
        if a["mode"] not in modes:
            skipped += 1
            continue
        total += 1
        try:
            a, results = check_case(case, ns_list, positions)
        except Exception as e:
            if "out of memory" in str(e).lower():
                total -= 1
                skipped += 1
                print(f"[SKIP] case {idx} OOM")
                torch.npu.empty_cache()
                continue
            raise
        finally:
            torch.npu.empty_cache()
        bad_res = [r for r in results if not r["equal"]]
        if bad_res:
            bad += 1
            fails.append((idx, a, bad_res))
            worst = max(r["max_diff"] for r in bad_res)
            print(f"[FAIL] case {idx} mode={a['mode']} dt={a['dt']} "
                  f"Sq={a['sq']} Sk={a['sk']} H={a['h']} Hk={a['hk']} D={a['d']} Dv={a['dv']} "
                  f"causal={int(a['causal'])} ns={a['ns']} -> {len(bad_res)}/{len(results)} 不一致, "
                  f"max_diff={worst:.3e}")
            for r in bad_res[:6]:
                print(f"        {r['tag']}: max_diff={r['max_diff']:.3e}")
        else:
            print(f"[ ok ] case {idx} mode={a['mode']} dt={a['dt']} Sq={a['sq']} Sk={a['sk']} "
                  f"D={a['d']} Dv={a['dv']} ns={a['ns']} ({len(results)} checks)")
        sys.stdout.flush()

    print()
    print(f"总计: {total} 条, 逐位一致 {total - bad}, 不一致 {bad}, 跳过(mode过滤) {skipped}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump([{"case": i, "attrs": {k: str(v) for k, v in a.items()},
                        "fails": f} for i, a, f in fails], f, indent=2)
        print(f"明细已写入 {args.out}")


if __name__ == "__main__":
    main()
