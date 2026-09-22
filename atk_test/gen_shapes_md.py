# Copyright (c) 2026. 由 atk case 生成的 JSON 汇总出 SHAPES.md 用例清单。
#
# 用法: python3 gen_shapes_md.py

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FWD = os.path.join(HERE, "result/fa4_full/json/all_fa4_full.json")
BWD = os.path.join(HERE, "result/fa4_bwd/json/all_fa4_bwd.json")
OUT = os.path.join(HERE, "SHAPES.md")

MODE_NAME = {0: "dense BSND", 1: "dense varlen TND", 2: "paged KV TND"}
HDR = ("| id | dtype | B | H | Hk | Sq | Sk | D[/Dv] | causal | window(l,r) | ns | varied |\n"
       "|---:|:--|---:|---:|---:|---:|---:|:--:|:--:|:--:|---:|:--:|")


def rec(c):
    i = c["inputs"]
    return dict(id=c["id"], dtype=i[0]["dtype"], mode=i[3]["range_values"],
                B=i[4]["range_values"], H=i[0]["shape"][-2], Hk=i[1]["shape"][-2],
                Sq=i[5]["range_values"], Sk=i[6]["range_values"],
                D=i[0]["shape"][-1], Dv=i[2]["shape"][-1],
                causal=i[7]["range_values"], wl=i[8]["range_values"],
                wr=i[9]["range_values"], ns=i[10]["range_values"], var=i[11]["range_values"])


def row(r):
    dv = f"{r['D']}" if r["D"] == r["Dv"] else f"{r['D']}/{r['Dv']}"
    return (f"| {r['id']} | {r['dtype']} | {r['B']} | {r['H']} | {r['Hk']} | {r['Sq']} | {r['Sk']} | "
            f"{dv} | {'T' if r['causal'] else 'F'} | ({r['wl']},{r['wr']}) | {r['ns']} | "
            f"{'T' if r['var'] else 'F'} |")


def mask_cat(r):
    swa = r["wl"] != -1 or r["wr"] != -1
    if r["causal"] and swa:
        return "causal+SWA"
    if r["causal"]:
        return "causal"
    return "SWA/local" if swa else "no-mask"


def head_cat(r):
    if r["H"] == r["Hk"]:
        return "MHA"
    return "MQA" if r["Hk"] == 1 else "GQA"


def main():
    R = [rec(c) for c in json.load(open(FWD))]
    Rb = [rec(c) for c in json.load(open(BWD))]
    out = ["# FA4 ATK 用例 Shape 清单\n",
           "> 由 `gen_shapes_md.py` 从 `atk case` 生成的 JSON 汇总，shape 源自 "
           "`tests/test_flash_attn_npu_v4.py`。",
           "> `D[/Dv]`：qk head_dim；分离时显示 `D/Dv`。TND 的 Sq/Sk 为最大值，"
           "实际长度由 `make_varlen_seqlens(seed=1234)` 生成。\n",
           "## 一、汇总\n", "| 项 | 值 |", "|---|---|",
           f"| 前向总条数 | {len(R)} |",
           "| ├ mode | " + "、".join(f"{MODE_NAME[k]} {v}" for k, v in sorted(Counter(r['mode'] for r in R).items())) + " |",
           "| ├ dtype | " + "、".join(f"{k} {v}" for k, v in Counter(r['dtype'] for r in R).items()) + " |",
           "| ├ head 结构 | " + "、".join(f"{k} {v}" for k, v in Counter(head_cat(r) for r in R).items()) + " |",
           "| ├ mask | " + "、".join(f"{k} {v}" for k, v in Counter(mask_cat(r) for r in R).items()) + " |",
           "| ├ head_dim 分离 | " + str(sum(1 for r in R if r['D'] != r['Dv'])) + " |",
           f"| ├ head_dim 范围 | {min(r['D'] for r in R)} ~ {max(r['D'] for r in R)} |",
           f"| └ seqlen_k 范围 | {min(r['Sk'] for r in R)} ~ {max(r['Sk'] for r in R)} |",
           f"| 反向总条数 | {len(Rb)}（mode=dense varlen TND，ns≤1） |", ""]
    out.append("## 二、前向用例\n")
    for m in (0, 1, 2):
        sub = [r for r in R if r["mode"] == m]
        out.append(f"### mode {m} — {MODE_NAME[m]}（{len(sub)} 条）\n")
        out.append(HDR)
        out += [row(r) for r in sub]
        out.append("")
    out.append("## 三、反向用例\n")
    out.append(HDR)
    out += [row(r) for r in Rb]
    out.append("")
    open(OUT, "w").write("\n".join(out))
    print(f"written {OUT}: {len(out)} lines; forward={len(R)} backward={len(Rb)}")


if __name__ == "__main__":
    main()
