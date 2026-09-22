#!/usr/bin/env bash
# =============================================================================
# Ascend 910 FA4 算子 ATK 测试一键脚本
#
# 用法：
#   ./run.sh gen      # 生成用例 JSON（前向 416 + 反向 96）
#   ./run.sh smoke    # 前向冒烟（前 10 条）
#   ./run.sh forward  # 前向全量（416 条）
#   ./run.sh bwd      # 反向全量（96 条）
#   ./run.sh          # 默认 = smoke
#
# 依赖：ATK 已安装，CANN 环境已 source，torch_npu + flash_attn_npu_4 已安装，
#       NPU 设备 Health=OK。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY_BIN="${ATK_PY_BIN:-/usr/local/python3.12.13/bin}"
export PATH="${PY_BIN}:${PATH}"

GENERATOR="fa4_generate_full.py"
PLUGIN="fa4_api.py"
NODES="node.yaml"
FULL_JSON="result/fa4_full/json/all_fa4_full.json"
BWD_JSON="result/fa4_bwd/json/all_fa4_bwd.json"

gen() {
    echo "[INFO] 生成前向用例（416）"
    atk case -f fa4_full.yaml -p "$GENERATOR"
    echo "[INFO] 生成反向用例（96）"
    atk case -f fa4_bwd.yaml -p "$GENERATOR"
}

smoke() {
    [[ -f "$FULL_JSON" ]] || gen
    echo "[INFO] 前向冒烟（前 10 条）"
    atk task -c "$FULL_JSON" -n "$NODES" -p "$PLUGIN" --task accuracy -e 10
}

forward() {
    [[ -f "$FULL_JSON" ]] || gen
    echo "[INFO] 前向全量（416 条）"
    atk task -c "$FULL_JSON" -n "$NODES" -p "$PLUGIN" --task accuracy
}

bwd() {
    [[ -f "$BWD_JSON" ]] || gen
    echo "[INFO] 反向全量（96 条）"
    atk task -c "$BWD_JSON" -n "$NODES" -p "$PLUGIN" --task accuracy
}

case "${1:-smoke}" in
    gen)     gen ;;
    smoke)   smoke ;;
    forward) forward ;;
    bwd)     bwd ;;
    *) echo "Usage: $0 [gen|smoke|forward|bwd]"; exit 1 ;;
esac
