# Ascend 910 FA4 算子 ATK 测试

基于 [ATK（Ascend Test Kit）](https://gitcode.com/AscendTest/ATK) 对昇腾 910 的
FA4 算子（`flash_attn_npu_4`）做端到端精度测试，覆盖与仓库自带
`tests/test_flash_attn_npu_v4.py` **完全一致**的 shape（head_dim 1~576）。

> **本目录只做 ATK 测试，不修改算子源码（`csrc/`）。**

---

## 1. 背景：为什么用“自定义执行方式”

FA4 在本仓库中是 PyTorch 扩展，不是 aclnn 自定义算子，因此 ATK 默认的
`AclnnBaseApi` 路径不适用。按 ATK《自定义执行方式》扩展：

- `fa4_api.py`（注册名 `fa4`）：`npu` 后端调 `flash_attn_func` / `flash_attn_varlen_func`；
  `cpu` 后端用仓库参考实现 `tests/common/attention_ref.py` 算 golden。
- `fa4_generate_full.py`（注册名 `fa4_full` / `fa4_bwd`）：按序号指定 shape/dtype/attr。

---

## 2. 环境准备

ATK 未随 CANN 发布，需从 [官方 Releases](https://gitcode.com/AscendTest/ATK/releases)
下载对应 Python 版本的 whl：

```bash
curl -L -o atk-26.7.8-cp312-cp312-linux_aarch64.whl \
  https://gitcode.com/AscendTest/ATK/releases/download/v26.7.8/atk-26.7.8-cp312-cp312-linux_aarch64.whl
pip install ./atk-26.7.8-cp312-cp312-linux_aarch64.whl
atk --help
```

> 注意：
> - 最新 tag `v26.9.7` 只提供 cp39/cp311，无 cp312，故选 `v26.7.8`（含 cp312）。
> - 安装依赖务必锁 `numpy==1.26.4`：`matplotlib` 会把 numpy 升到 2.x，
>   破坏 `torch_npu` / `triton-ascend`。

其余前置：CANN 已 source、`torch_npu` 与 `flash_attn_npu_4` 已安装、NPU Health=OK。

---

## 3. 目录结构

```
atk_test/
├── README.md              本文档
├── SHAPES.md              用例 shape 完整清单（476 前向 + 96 反向）
├── gen_shapes_md.py       由 JSON 生成 SHAPES.md
├── fa4_api.py             自定义执行方式（npu 算子 / cpu golden，支持 BSND/TND/paged）
├── fa4_full.yaml          前向用例设计（476 条）
├── fa4_bwd.yaml           反向用例设计（96 条）
├── fa4_bwd_det.yaml       反向确定性用例设计（96 条，deterministic=True）
├── fa4_generate_full.py   自定义参数约束（import 原测试 shape 表）
├── node.yaml              节点配置（npu 待测 + cpu 标杆）
├── nodes_dc.yaml          确定性测试节点配置（accuracy_dc + 确定性 run mode）
├── run.sh                 一键脚本
├── result/                atk case 生成物（运行后产生）
└── atk_output/            ATK 结果与报告（运行后产生）
```

---

## 4. 使用方式

```bash
export PATH=/usr/local/python3.12.13/bin:$PATH
cd atk_test

./run.sh gen       # 生成前向 476 + 反向 96
./run.sh smoke     # 前向冒烟（前 10 条）
./run.sh forward   # 前向全量
./run.sh bwd       # 反向全量
./run.sh dc_fwd    # 前向确定性（accuracy_dc）
./run.sh dc_bwd    # 反向确定性（accuracy_dc）
```

等价手工两步：

```bash
atk case -f fa4_full.yaml -p fa4_generate_full.py
atk task -c result/fa4_full/json/all_fa4_full.json -n node.yaml -p fa4_api.py --task accuracy
```

---

## 5. 用例设计

### 5.1 覆盖范围（与原测试一致）

`fa4_generate_full.py` **直接 import** `tests/test_flash_attn_npu_v4.py` 的四张
shape 表并归一化，保证覆盖一致：

| 来源 | 条数 | 说明 |
|---|---|---|
| `test_cases` | 216 | dense BSND / dense varlen TND / paged KV TND |
| `hd_cases` | 82 | head_dim ≤ 256 覆盖 |
| `flash_attn_func_cases` | 10 | dense BSND |
| `head_size_v_cases` | 168 | qk/v head_dim 分离（含 (576,512)） |
| **合计** | **476** | dtype: fp16 191 / bf16 285；head_dim 1~576 |

> `fa4_full.yaml` 的 `dtype_numbers` 必须等于上表合计（当前 476）；若原测试
> shape 表变更，需同步更新该值。

三种执行模式：

| mode | 布局 | q/k/v shape | 调用 |
|---|---|---|---|
| 0 | dense BSND | `(B, S, H, D)` | `flash_attn_func` |
| 1 | dense varlen TND | `(total, H, D)` | `flash_attn_varlen_func` + cu_seqlens |
| 2 | paged KV TND | k/v `(nblocks, 128, H, D)` | `flash_attn_varlen_func` + page_table |

反向子集 `fa4_bwd`（96 条）= mode 1 且 `num_splits ≤ 1`，对齐原测试的
`bwd_supported` 条件。

### 5.2 输入参数

| 输入 | 类型 | 说明 |
|---|---|---|
| `q` / `k` / `v` | tensor | shape 随 mode 变化 |
| `mode` / `batch` / `seqlen_q` / `seqlen_k` | attr(int) | 结构参数 |
| `causal` | attr(bool) | 因果掩码 |
| `window_left` / `window_right` | attr(int) | -1 表示无限窗口 |
| `num_splits` | attr(int) | KV 切分 |
| `is_varied` | attr(bool) | varlen 序列长度是否随机 |

varlen 的 `cu_seqlens` 由插件用 `make_varlen_seqlens(seed=1234)` 复算
（ATK 随机生成的 cumsum 不合法）。

### 5.3 精度标准：双标杆（cv_fused_double_benchmark）

ATK 对每个输出做三路比对：

| 角色 | 后端 | 输入精度 | 用途 |
|---|---|---|---|
| golden | CPU 第 1 遍 | ATK 升精度（fp16/bf16 → fp32） | 高精度真值 |
| benchmark | CPU 第 2 遍 | 保持原精度 | CPU 自身误差水平 |
| npu | NPU | 原精度 | 待测对象 |

判据是**误差比值**：`最大相对误差比例 = Actual/Benchmark`（阈值 10）、
`均方根误差比例`（阈值 2）。前向与反向套件均使用该标准。

**关键约束**：golden 与 benchmark 两遍 CPU 结果必须不同，否则分母为 0。
故插件 CPU 参考**跟随传入 dtype**（`ref_flash_attention(..., upcast=False)`），
**不得硬 cast 到 fp32**（见《ATK_TEST_GUIDE》第三章 §4）。

---

## 6. 测试结果（本机 8×910B3）

### 6.1 前向：476/476 Pass ✅

```
|  名称 | 总用例数 | 执行成功用例个数 | 执行失败用例个数 | 通过用例个数 | 通过率 | 精度是否达标 |
| cpu_0 |   476    |       476        |        0         |     476      | 100.0  |     Pass     |
Total Task: 476, success 476, failed 0
Summary info: acc_pass_result:Pass
```

含 **60 条 head_dim=576 / head_dim_v=512** 用例（全部通过）。
`D≥513` 曾因前向 L1 预算无符号下溢而崩溃，现算子侧已用 `kvStackCap` 修复；
本套件据此把 head_dim 覆盖扩到 576 并实测通过。

### 6.2 反向：88/96（8 条失败 = 仓库既有问题）

```
| cpu_0 |    96    |        96        |        0         |      88      |  91.7  |    Failed    |
Total Task: 96, success 96, failed 0
```

8 条失败全部是 **`Dv > D` 的 qk/v head_dim 分离 TND 反向**
（`D=64, Dv=128`，MHA/MQA × fp16/bf16 × causal/非causal），失败输出为 dK/dV。
用仓库自带测试独立复现，结论一致（`Dv ≤ D` 全部通过，`Dv > D` 失败）。
详见 §8。

---

## 7. 已知边界与发现（本 ATK 套件实测）

- **前向 head_dim 有效上限**：算子声明 1~576，但 `D≥513` 会因
  `mha_fwd_kvcache.cpp` 的 `L1_MAX_SIZE - embedV*MAX_KV_STACK_LEN*sizeof` 无符号
  下溢而崩溃。算子侧已用 `kvStackCap` 修复，修复后 1~576 全部通过。
- **反向 head_dim 上限**：1~256（`flash_api.cpp` 校验），无缺口。
- **反向 `Dv > D` 偏差**：见 §8。

---

## 8. 反向 `Dv > D` 问题（既有，非 ATK 引入）

| (dqk, dv) | 关系 | 仓库自带测试 |
|---|---|---|
| (192,128) | Dv < D | PASS |
| (128,64) | Dv < D | PASS |
| **(64,128)** | **Dv > D** | **FAIL**（dQ/dK 严重偏离，NPU ~100+ vs 参考 ~0.003） |

ATK 忠实复现该问题，非测试框架或本端口引入。

---

## 9. 备注：ATK 安装对系统环境的影响

ATK 及依赖装入 `/usr/local/python3.12.13`；安装过程曾把 `numpy` 升至 2.5.3
（破坏 `torch_npu`），已回退至 `numpy==1.26.4`。

---

## 10. 确定性测试（determinism）

用 ATK 原生 `accuracy_dc` 任务：**克隆输入重复执行 N 次并逐位比对**
（日志 `start run accuracy dc compare in N times`），节点配置见 `nodes_dc.yaml`
（`task: ['accuracy_dc']` + `run_modes: ['ascend_use_deterministic_algorithms']`）。

```bash
./run.sh dc_fwd                    # 前向 476 条，dc_loop=50
./run.sh dc_bwd                    # 反向 96 条，dc_loop=50
DC_LOOP_NUMS=200 ./run.sh dc_bwd   # 覆盖循环次数
```

汇总字段 `is_acc_dc_pass`，报告列"确定性计算是否达标"。

### 10.1 结果

| 套件 | 用例数 | 确定性达标 | 结论 |
|---|---|---|---|
| 前向 | 476 | 476 (100%) | 确定 |
| 反向（默认 `deterministic=False`） | 96 | 49 (51.0%) | **47 条非确定** |
| 反向（`deterministic=True`） | 96 | **96 (100%)** | 确定 |

### 10.2 根因：反向 `deterministic` 开关（默认 false）

```
flash_attn_varlen_func(..., deterministic: bool = False)          # 默认 False
  -> flash_api.cpp:            fagInfo.isDeterministic = deterministic
  -> bwd_dispatch_common.hpp:  BWD_BOOL_SWITCH(deterministic, IsDtm, ...)  # 模板 IS_DTM
  -> fag_kernel.cpp:           if constexpr (IS_DTM == ENABLE)  两条内核路径
       默认(False): ComputeMMDqkv -> mmad_fag_dqkv.hpp SetAtomicType<float>()  原子累加 -> 非确定
       True       : DTMComputeMMDqkv + EpilogueFAGDtmAdd + SyncAll()           确定性归约
```

`fag_tiling.h` 默认 `isDeterministic = false`。

### 10.3 独立逐位复现（20 轮，同输入）

| 用例 | out | dQ | dK | dV |
|---|---|---|---|---|
| D=32 bf16 | 确定 | 确定 | 非确定 (11/19 轮) | 确定 |
| D=64 fp16 causal | 确定 | 确定 | 非确定 (19/19 轮) | 确定 |
| Sq=16 Sk=4096 | 确定 | 非确定 | 非确定 | 确定 |

非确定项固定为 `dK`（主）/`dQ`（次）；`out` 与 `dV` 恒确定。失败与
`num_splits ≥ 1` 强相关（11/12），但 `num_splits = 0` 也有 36/84 失败。

### 10.4 `deterministic=True` 可完全消除（ATK 全量验证）

确定性用例拆分为**独立设计文件** `fa4_bwd_det.yaml`（与 `fa4_bwd.yaml` 同 shape，
仅多 `deterministic` 入参并置 True），**不影响精度套件**：

| 设计文件 | 用途 | deterministic |
|---|---|---|
| `fa4_bwd.yaml` | 精度（算子默认路径） | 无此入参（默认 False） |
| `fa4_bwd_det.yaml` | 确定性 | True |

```bash
atk case -f fa4_bwd_det.yaml -p fa4_generate_full.py
./run.sh dc_bwd        # 96/96 达标
```

实测：反向 96 条从 49/96 提升到 **96/96**（`is_acc_dc_pass:Pass`）。

> `run_modes: ascend_use_deterministic_algorithms` 对本算子**无效**——算子不读
> `torch.use_deterministic_algorithms`，只读自身 `deterministic` 入参。
