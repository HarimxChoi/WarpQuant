#!/usr/bin/env bash
set -euo pipefail

workspace="${WARPQUANT_WORKSPACE:-/workspace/warpquant-qwen38-27b-20260815}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bin="$workspace/llama.cpp/build/bin"
python="${PYTHON:-python}"
lm_eval="${LM_EVAL:-lm_eval}"
results="$workspace/results/gsm8k-500"
datasets="$workspace/datasets/gsm8k"
port=18081
export LD_LIBRARY_PATH="$bin:${LD_LIBRARY_PATH:-}"
export HF_HOME="$workspace/hf-cache"

mkdir -p "$results" "$datasets"
test -x "$bin/llama-server"
command -v "$lm_eval" >/dev/null

conditions=(bf16 q4_k_m iq3_s warpquant)
models=(
  "$workspace/gguf/source-f16.gguf"
  "$workspace/gguf/source-q4_k_m.gguf"
  "$workspace/gguf/source-iq3_s.gguf"
  "$workspace/gguf/warpquant-r16e4h4-f16.gguf"
)

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
  "https://huggingface.co/datasets/openai/gsm8k/resolve/e53f048856ff4f594e959d75785d2c2d37b678ee/main/train-00000-of-00001.parquet?download=true" \
  "$datasets/train.parquet"
download \
  "https://huggingface.co/datasets/openai/gsm8k/resolve/e53f048856ff4f594e959d75785d2c2d37b678ee/main/test-00000-of-00001.parquet?download=true" \
  "$datasets/test.parquet"
echo "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d  $datasets/train.parquet" | sha256sum -c -
echo "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59  $datasets/test.parquet" | sha256sum -c -

task_yaml="$repo/benchmarks/tasks/gsm8k_500_local.yaml"
cp "$task_yaml" "$results/gsm8k-task.yaml"
"$python" -m pip freeze > "$results/python-freeze.txt"

server_pid=""
stop_server() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid"
    wait "$server_pid" || true
  fi
  server_pid=""
}
trap stop_server EXIT INT TERM

result_ready() {
  local directory="$1"
  "$python" - "$directory" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
results = []
for path in root.rglob("results_*.json"):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        continue
    if "gsm8k_500_local" in payload.get("results", {}):
        results.append(path)
samples = list(root.rglob("samples_gsm8k_500_local_*.jsonl"))
assert len(results) == 1
assert len(samples) == 1
records = [json.loads(line) for line in samples[0].open(encoding="utf-8") if line.strip()]
expected_docs = set(range(500))
docs_by_filter = {
    name: {record["doc_id"] for record in records if record["filter"] == name}
    for name in ("strict-match", "flexible-extract")
}
assert docs_by_filter == {
    "strict-match": expected_docs,
    "flexible-extract": expected_docs,
}
PY
}

for index in "${!conditions[@]}"; do
  condition="${conditions[$index]}"
  model="${models[$index]}"
  output="$results/$condition"
  test -s "$model"
  mkdir -p "$output"

  if result_ready "$output" 2>/dev/null; then
    continue
  fi
  find "$output" -type f \
    \( -name 'results_*.json' -o -name 'samples_*.jsonl' \) -delete
  sha256sum "$model" > "$output/model-sha256.txt"

  "$bin/llama-server" \
    --model "$model" \
    --alias qwen38-27b \
    --host 127.0.0.1 \
    --port "$port" \
    --gpu-layers 999 \
    --ctx-size 32768 \
    --parallel 8 \
    --batch-size 2048 \
    --ubatch-size 512 \
    > "$output/server.log" 2>&1 &
  server_pid=$!

  ready=0
  for _ in $(seq 1 180); do
    if curl --silent --fail "http://127.0.0.1:$port/health" >/dev/null; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  test "$ready" = 1

  HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 "$lm_eval" run \
    --model local-completions \
    --model_args \
      "model=qwen38-27b,base_url=http://127.0.0.1:$port/v1/completions,tokenizer=$workspace/source-bf16,tokenizer_backend=huggingface,tokenized_requests=False,num_concurrent=8,max_retries=3,max_gen_toks=256,max_length=4096,timeout=600" \
    --tasks gsm8k_500_local \
    --include_path "$repo/benchmarks/tasks" \
    --num_fewshot 5 \
    --limit 500 \
    --batch_size 1 \
    --seed 0,1234,1234,1234 \
    --cache_requests true \
    --output_path "$output" \
    --log_samples \
    > "$output/harness.log" 2>&1

  result_ready "$output"
  stop_server
done

"$python" "$repo/benchmarks/summarize_gsm8k.py" \
  --results "$results" \
  --conditions "${conditions[@]}" \
  --expected-samples 500

