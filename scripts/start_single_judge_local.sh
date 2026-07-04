#!/usr/bin/env bash
set -euo pipefail

cd /root/ICL/LWY-bench
mkdir -p logs

run_one() {
  local name="$1"
  local pred_dir="$2"
  echo "=== ${name} start $(date -Is) ==="
  python3 scripts/judge_lwy_single.py \
    --pred-dir "${pred_dir}" \
    --out-dir "${pred_dir}/deepseek_judge_single" \
    --judge-model deepseek-v4-flash \
    --workers 4 \
    --max-tokens 1200 \
    --retries 3
  echo "=== ${name} done $(date -Is) ==="
}

run_one qwen2_5_1_5b results/raw_outputs/qwen2_5_1_5b_generated_v1
run_one symbol_lora_qwen1_5b results/raw_outputs/symbol_lora_qwen1_5b_generated_v1
run_one metaicl_lora_qwen1_5b results/raw_outputs/metaicl_lora_qwen1_5b_generated_v1
