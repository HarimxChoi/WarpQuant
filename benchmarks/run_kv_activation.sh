#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${MODEL:?set MODEL to the WarpQuant checkpoint directory}"
python="${PYTHON:-python}"
results="${RESULTS:-$repo/results/kv-activation}"
reference_logits="$results/reference-logits"
export PYTHONPATH="$repo/benchmarks${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$results"

"$python" "$repo/benchmarks/evaluate_kv_activation.py" \
  --model "$model" \
  --output "$results/weight-only.json" \
  --attention-mode bf16 \
  --max-tokens 4096 \
  --sequence-length 512 \
  --save-logits "$reference_logits"

"$python" "$repo/benchmarks/evaluate_kv_activation.py" \
  --model "$model" \
  --output "$results/k4v4r128.json" \
  --attention-mode turboquant \
  --key-bits 4 \
  --value-bits 4 \
  --residual-length 128 \
  --max-tokens 4096 \
  --sequence-length 512 \
  --reference-logits "$reference_logits"

"$python" "$repo/benchmarks/evaluate_kv_activation.py" \
  --model "$model" \
  --output "$results/a8.json" \
  --attention-mode bf16 \
  --activation-bits 8 \
  --max-tokens 4096 \
  --sequence-length 512 \
  --reference-logits "$reference_logits"

"$python" "$repo/benchmarks/evaluate_kv_activation.py" \
  --model "$model" \
  --output "$results/k4v4r128-a8.json" \
  --attention-mode turboquant \
  --key-bits 4 \
  --value-bits 4 \
  --residual-length 128 \
  --activation-bits 8 \
  --max-tokens 4096 \
  --sequence-length 512 \
  --reference-logits "$reference_logits"
