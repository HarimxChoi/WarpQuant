# Benchmark reproduction

This directory contains only the evaluation code behind the tables in the main
README. Model production, remote-instance setup, upload automation, and private
experiment orchestration are intentionally excluded.

## Environment

- NVIDIA H100 80GB
- `llama.cpp` commit `2606220d9f2705dab8260633e9f85ce5b081319e`
- `lm-evaluation-harness==0.4.12`
- Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
- Text denominator: `26,895,998,464` parameters, excluding vision and MTP

```bash
pip install -r requirements.txt
```

The scripts use `WARPQUANT_WORKSPACE` for model, dataset, and result paths. The
default reproduces the original experiment layout:

```text
/workspace/warpquant-qwen38-27b-20260815/
├── gguf/
│   ├── source-f16.gguf
│   ├── source-q4_k_m.gguf
│   ├── source-iq3_s.gguf
│   └── warpquant-r16e4h4-f16.gguf
├── source-bf16/
├── llama.cpp/build/bin/
├── datasets/
└── results/
```

## WT2, ARC-Challenge, and MMLU

`run_text_metrics.sh` evaluates one model at a time with `llama-perplexity`.
The recorded datasets are:

| Dataset | Examples | SHA-256 |
|---|---:|---|
| WikiText-2 test | full test text | `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08` |
| ARC-Challenge validation | 299 | `19af3dbe27d9bb301a939cebbda1b0e4c6b3c43a4a724ae99fed27fba88edc22` |
| MMLU test | 13,943 | `88bbb0b99f0ac75dc63b82017d301be2e6b1b922b6a6eb508df41e5ac4396b5c` |

```bash
bash benchmarks/run_text_metrics.sh bf16 "$WARPQUANT_WORKSPACE/gguf/source-f16.gguf"
bash benchmarks/run_text_metrics.sh q4_k_m "$WARPQUANT_WORKSPACE/gguf/source-q4_k_m.gguf"
bash benchmarks/run_text_metrics.sh iq3_s "$WARPQUANT_WORKSPACE/gguf/source-iq3_s.gguf"
bash benchmarks/run_text_metrics.sh warpquant "$WARPQUANT_WORKSPACE/gguf/warpquant-r16e4h4-f16.gguf"
```

## Commonsense reasoning

`run_commonsense.sh` evaluates the same fixed first 1,000 examples from
HellaSwag, WinoGrande, and PIQA for each format. The reported score is their
unweighted macro average.

```bash
bash benchmarks/run_commonsense.sh
```

## GSM8K-500

`run_gsm8k_500.sh` runs the same first 500 GSM8K test examples with five-shot
prompts and deterministic decoding. Each document is validated under both the
strict-match and flexible-extract filters before the four-model summary is
written.

```bash
PYTHON=python LM_EVAL=lm_eval bash benchmarks/run_gsm8k_500.sh
```

## KV cache and activation ablation

`run_kv_activation.sh` evaluates weight-only, K4/V4/R128, dynamic A8, and their
combination on 4,088 WikiText-2 validation tokens. It records PPL, paired KL,
top-1 agreement, and context-dependent KV compression.

```bash
MODEL=/path/to/r16e4h4-text-only bash benchmarks/run_kv_activation.sh
```

The corresponding quantizer unit tests are under `benchmarks/tests/`:

```bash
PYTHONPATH=benchmarks pytest -q benchmarks/tests
```
