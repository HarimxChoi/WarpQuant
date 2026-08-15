#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


MODELS = ["bf16", "q4_k_m", "iq3_s", "warpquant"]


def score(path: Path, pattern: str) -> float:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError(f"expected one terminal score in {path}, found {len(matches)}")
    return float(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for model in MODELS:
        hellaswag = score(args.results / f"{model}-hellaswag.log", r"Final result:\s+([0-9.]+)\s+\+/-")
        winogrande = score(
            args.results / f"{model}-winogrande.log",
            r"Final Winogrande score\(1000 tasks\):\s+([0-9.]+)\s+\+/-",
        )
        piqa = score(args.results / f"{model}-piqa.log", r"Final result:\s+([0-9.]+)\s+\+/-")
        rows.append(
            {
                "format": model,
                "hellaswag_1000": hellaswag,
                "winogrande_1000": winogrande,
                "piqa_1000": piqa,
                "commonsense_macro_average": (hellaswag + winogrande + piqa) / 3,
            }
        )

    output = {
        "metric": "Commonsense Reasoning Average",
        "aggregation": "unweighted arithmetic mean of HellaSwag, WinoGrande, and PIQA accuracy",
        "sampling": "1000 deterministic llama.cpp tasks per dataset",
        "rows": rows,
    }
    (args.results / "commonsense-summary.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "| Format | HellaSwag | WinoGrande | PIQA | Commonsense Avg. |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['format']} | {row['hellaswag_1000']:.2f} | "
            f"{row['winogrande_1000']:.2f} | {row['piqa_1000']:.2f} | "
            f"{row['commonsense_macro_average']:.2f} |"
        )
    (args.results / "commonsense-summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


