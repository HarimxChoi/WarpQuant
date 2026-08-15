import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoTokenizer,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from tqi3.activation_quant import LinearInputQuantizer
from tqi3.cache_quant import (
    CacheQuantConfig,
    TurboQuantAttention,
    packed_cache_bytes_per_token,
    resident_cache_bytes,
)


def load_model(path: Path, attention_mode: str, quantizer: TurboQuantAttention | None):
    config = AutoConfig.from_pretrained(path)
    text_config = getattr(config, "text_config", config)
    if quantizer is not None:
        ALL_ATTENTION_FUNCTIONS.register("warpquant_tq", quantizer)
    attention = "warpquant_tq" if attention_mode == "turboquant" else "eager"
    if text_config.model_type == "qwen3_5_text":
        model = AutoModelForImageTextToText.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=attention,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=attention,
        )
    model.eval()
    return model, text_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-mode", choices=("bf16", "turboquant"), default="bf16")
    parser.add_argument("--key-bits", type=int, default=3)
    parser.add_argument("--value-bits", type=int, default=4)
    parser.add_argument("--value-group-size", type=int, default=32)
    parser.add_argument("--residual-length", type=int, default=128)
    parser.add_argument("--activation-bits", type=int, choices=(4, 8))
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--save-logits", type=Path)
    parser.add_argument("--reference-logits", type=Path)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.config, split=args.split)
    documents = [text for text in dataset["text"] if text.strip()]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokens = tokenizer("\n\n".join(documents), return_tensors="pt", add_special_tokens=False).input_ids[0]
    tokens = tokens[: args.max_tokens]
    quant_config = CacheQuantConfig(
        key_bits=args.key_bits,
        value_bits=args.value_bits,
        value_group_size=args.value_group_size,
        residual_length=args.residual_length,
    )
    quantizer = TurboQuantAttention(quant_config) if args.attention_mode == "turboquant" else None
    model, config = load_model(args.model, args.attention_mode, quantizer)
    activation_quantizer = None
    if args.activation_bits is not None:
        activation_quantizer = LinearInputQuantizer(args.activation_bits)
        activation_quantizer.install(model)
    if args.save_logits is not None:
        args.save_logits.mkdir(parents=True, exist_ok=True)

    total_nll = 0.0
    total_tokens = 0
    total_kl = 0.0
    total_agreement = 0
    max_logit_error = 0.0
    mean_logit_error = 0.0
    logit_values = 0
    with torch.inference_mode():
        for index, start in enumerate(range(0, tokens.numel() - 1, args.sequence_length)):
            chunk = tokens[start : start + args.sequence_length].unsqueeze(0).to(model.device)
            if chunk.shape[1] < 2:
                continue
            logits = model(input_ids=chunk, use_cache=False).logits[:, :-1].float()
            labels = chunk[:, 1:]
            total_nll += float(
                F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), reduction="sum")
            )
            total_tokens += labels.numel()
            if args.save_logits is not None:
                np.save(args.save_logits / f"{index:03d}.npy", logits.cpu().half().numpy())
            if args.reference_logits is not None:
                reference = torch.from_numpy(np.load(args.reference_logits / f"{index:03d}.npy")).to(
                    logits.device
                ).float()
                reference_log = F.log_softmax(reference, dim=-1)
                candidate_log = F.log_softmax(logits, dim=-1)
                total_kl += float(torch.sum(torch.exp(reference_log) * (reference_log - candidate_log)))
                total_agreement += int(torch.sum(torch.argmax(reference, dim=-1) == torch.argmax(logits, dim=-1)))
                difference = torch.abs(reference - logits)
                max_logit_error = max(max_logit_error, float(difference.max()))
                mean_logit_error += float(difference.sum())
                logit_values += difference.numel()
            print(f"tokens={total_tokens}", flush=True)

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    layer_types = getattr(config, "layer_types", ["full_attention"] * config.num_hidden_layers)
    attention_layers = sum(layer == "full_attention" for layer in layer_types)
    packed_per_head = packed_cache_bytes_per_token(
        head_dim,
        args.key_bits,
        args.value_bits,
        args.value_group_size,
    )
    bf16_per_token = attention_layers * config.num_key_value_heads * head_dim * 2 * 2
    packed_per_token = attention_layers * config.num_key_value_heads * packed_per_head
    bf16_context_bytes = args.sequence_length * bf16_per_token
    resident_bytes = resident_cache_bytes(
        args.sequence_length,
        args.residual_length,
        bf16_per_token,
        packed_per_token,
    )
    metrics = {
        "model": str(args.model),
        "attention_mode": args.attention_mode,
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "fingerprint": dataset._fingerprint,
        "tokens": total_tokens,
        "sequence_length": args.sequence_length,
        "nll": total_nll / total_tokens,
        "ppl": math.exp(total_nll / total_tokens),
        "cache": {
            "attention_layers": attention_layers,
            "key_bits": args.key_bits,
            "value_bits": args.value_bits,
            "value_group_size": args.value_group_size,
            "residual_length": args.residual_length,
            "bf16_bytes_per_token": bf16_per_token,
            "packed_bytes_per_old_token": packed_per_token,
            "asymptotic_compression": bf16_per_token / packed_per_token,
            "bf16_context_bytes": bf16_context_bytes,
            "resident_context_bytes": resident_bytes,
            "context_compression": bf16_context_bytes / resident_bytes,
            "fake_quant_only": True,
        },
    }
    if args.reference_logits is not None:
        metrics["paired_kl"] = total_kl / total_tokens
        metrics["top1_agreement"] = total_agreement / total_tokens
        metrics["max_abs_logit_error"] = max_logit_error
        metrics["mean_abs_logit_error"] = mean_logit_error / logit_values
    if activation_quantizer is not None:
        metrics["activation"] = {
            "bits": args.activation_bits,
            "modules": activation_quantizer.stats.modules,
            "calls": activation_quantizer.stats.calls,
            "values": activation_quantizer.stats.values,
            "scope": "decoder linear inputs",
            "integer_kernel": False,
        }
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


