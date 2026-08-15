#!/usr/bin/env bash
set -euo pipefail

condition="$1"
model="$2"
workspace="${WARPQUANT_WORKSPACE:-/workspace/warpquant-qwen38-27b-20260815}"
bin="$workspace/llama.cpp/build/bin"
datasets="$workspace/datasets"
results="$workspace/results/llama"
export LD_LIBRARY_PATH="$bin:${LD_LIBRARY_PATH:-}"

mkdir -p "$results"
exec 9> "$results/$condition-eval.lock"
flock 9
test -s "$model"

input_manifest="$results/$condition-input-sha256.txt"
input_tmp="$input_manifest.tmp.$$"
sha256sum \
  "$model" \
  "$bin/llama-perplexity" \
  "$datasets/wiki.test.raw" \
  "$datasets/arc-challenge-validation.bin" \
  "$datasets/mmlu-test.bin" \
  > "$input_tmp"
if [[ ! -s "$input_manifest" ]] || ! cmp -s "$input_tmp" "$input_manifest"; then
  rm -f \
    "$results/$condition-wiki.log" \
    "$results/$condition-arc.log" \
    "$results/$condition-mmlu.log"
fi

run_quality() {
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

run_quality "$results/$condition-wiki.log" 'Final estimate: PPL' \
  "$bin/llama-perplexity" \
  --model "$model" \
  --file "$datasets/wiki.test.raw" \
  --gpu-layers 999 \
  --ctx-size 512 \
  --batch-size 512 \
  --ubatch-size 512

run_quality "$results/$condition-arc.log" 'Final result:' \
  "$bin/llama-perplexity" \
  --model "$model" \
  --binary-file "$datasets/arc-challenge-validation.bin" \
  --multiple-choice \
  --multiple-choice-tasks 0 \
  --parallel 8 \
  --gpu-layers 999 \
  --ctx-size 512 \
  --batch-size 512 \
  --ubatch-size 512

run_quality "$results/$condition-mmlu.log" 'Final result:' \
  "$bin/llama-perplexity" \
  --model "$model" \
  --binary-file "$datasets/mmlu-test.bin" \
  --multiple-choice \
  --multiple-choice-tasks 0 \
  --parallel 8 \
  --gpu-layers 999 \
  --ctx-size 512 \
  --batch-size 512 \
  --ubatch-size 512

mv "$input_tmp" "$input_manifest"

