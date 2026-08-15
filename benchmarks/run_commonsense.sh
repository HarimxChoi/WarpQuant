#!/usr/bin/env bash
set -euo pipefail

workspace="${WARPQUANT_WORKSPACE:-/workspace/warpquant-qwen38-27b-20260815}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="$workspace/llama.cpp/build/bin"
datasets="$workspace/datasets/commonsense"
results="$workspace/results/commonsense"
python="${PYTHON:-python}"
export LD_LIBRARY_PATH="$bin:${LD_LIBRARY_PATH:-}"

mkdir -p "$datasets" "$results"

download() {
  local url="$1"
  local output="$2"
  if [[ ! -s "$output" ]]; then
    rm -f "$output.tmp"
    curl --fail --location --retry 5 --output "$output.tmp" "$url"
    mv "$output.tmp" "$output"
  fi
}

download \
  "https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/resolve/37884b81b4957f1950a53b6ff48d77c8dd5e430c/hellaswag-validation.bin" \
  "$datasets/hellaswag-validation.bin"
download \
  "https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/resolve/93f7fb0697ca6cd2c7e5727dcba3b7200fa4014c/winogrande-debiased-eval.csv" \
  "$datasets/winogrande-debiased-eval.csv"
download \
  "https://huggingface.co/datasets/ybisk/piqa/resolve/078a131412f46a38025a762322c174a8bae2610c/plain_text/piqa-validation.parquet" \
  "$datasets/piqa-validation.parquet"

"$python" "$repo/benchmarks/build_piqa_multiple_choice.py" \
  --input "$datasets/piqa-validation.parquet" \
  --output "$datasets/piqa-validation.bin" \
  --manifest "$datasets/piqa-validation.json"

"$python" - "$datasets/hellaswag-validation.bin" <<'PY'
import struct
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("rb") as handle:
    tasks = struct.unpack("<I", handle.read(4))[0]
assert tasks == 10042, tasks
PY
test "$(awk 'NF { count += 1 } END { print count }' "$datasets/winogrande-debiased-eval.csv")" = "1267"
"$python" - "$datasets/piqa-validation.json" <<'PY'
import json
import sys

manifest = json.load(open(sys.argv[1], encoding="utf-8"))
assert manifest["tasks"] == 1838
PY

sha256sum \
  "$datasets/hellaswag-validation.bin" \
  "$datasets/winogrande-debiased-eval.csv" \
  "$datasets/piqa-validation.parquet" \
  "$datasets/piqa-validation.bin" \
  > "$results/dataset-sha256.txt"

run_eval() {
  local output="$1"
  local terminal_pattern="$2"
  shift 2
  if [[ "$(grep -c "$terminal_pattern" "$output" 2>/dev/null || true)" != "1" ]]; then
    rm -f "$output.tmp"
    "$@" > "$output.tmp" 2>&1
    test "$(grep -c "$terminal_pattern" "$output.tmp")" = "1"
    mv "$output.tmp" "$output"
  fi
}

conditions=(bf16 q4_k_m iq3_s warpquant)
models=(
  "$workspace/gguf/source-f16.gguf"
  "$workspace/gguf/source-q4_k_m.gguf"
  "$workspace/gguf/source-iq3_s.gguf"
  "$workspace/gguf/warpquant-r16e4h4-f16.gguf"
)

for index in "${!conditions[@]}"; do
  condition="${conditions[$index]}"
  model="${models[$index]}"
  test -s "$model"
  sha256sum "$model" > "$results/$condition-model-sha256.txt"

  run_eval "$results/$condition-hellaswag.log" 'Final result:' \
    "$bin/llama-perplexity" \
    --model "$model" \
    --binary-file "$datasets/hellaswag-validation.bin" \
    --multiple-choice --multiple-choice-tasks 1000 --parallel 8 \
    --gpu-layers 999 --ctx-size 512 --batch-size 512 --ubatch-size 512

  run_eval "$results/$condition-winogrande.log" 'Final Winogrande score(1000 tasks):' \
    "$bin/llama-perplexity" \
    --model "$model" \
    --file "$datasets/winogrande-debiased-eval.csv" \
    --winogrande --winogrande-tasks 1000 --parallel 8 \
    --gpu-layers 999 --ctx-size 512 --batch-size 512 --ubatch-size 512

  run_eval "$results/$condition-piqa.log" 'Final result:' \
    "$bin/llama-perplexity" \
    --model "$model" \
    --binary-file "$datasets/piqa-validation.bin" \
    --multiple-choice --multiple-choice-tasks 1000 --parallel 8 \
    --gpu-layers 999 --ctx-size 512 --batch-size 512 --ubatch-size 512
done

"$python" "$repo/benchmarks/summarize_commonsense.py" --results "$results"

