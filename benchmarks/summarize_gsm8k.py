import argparse
import hashlib
import json
from pathlib import Path

TASK = "gsm8k_500_local"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_file(root: Path) -> Path:
    matches = []
    for path in root.rglob("results_*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if TASK in payload.get("results", {}):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"expected one result JSON in {root}, found {len(matches)}")
    return matches[0]


def sample_file(root: Path) -> Path:
    matches = list(root.rglob(f"samples_{TASK}_*.jsonl"))
    if len(matches) != 1:
        raise ValueError(f"expected one sample JSONL in {root}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    parser.add_argument("--expected-samples", type=int, default=500)
    args = parser.parse_args()

    rows = []
    for condition in args.conditions:
        root = args.results / condition
        result_path = result_file(root)
        samples_path = sample_file(root)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = result["results"][TASK]
        records = [
            json.loads(line)
            for line in samples_path.open(encoding="utf-8")
            if line.strip()
        ]
        expected_docs = set(range(args.expected_samples))
        docs_by_filter = {
            name: {record["doc_id"] for record in records if record["filter"] == name}
            for name in ("strict-match", "flexible-extract")
        }
        if docs_by_filter != {
            "strict-match": expected_docs,
            "flexible-extract": expected_docs,
        }:
            counts = {name: len(doc_ids) for name, doc_ids in docs_by_filter.items()}
            raise ValueError(
                f"{condition}: expected both filters over doc_id 0..{args.expected_samples - 1}, "
                f"found {counts}"
            )
        sample_count = len(expected_docs)
        rows.append(
            {
                "condition": condition,
                "samples": sample_count,
                "num_fewshot": 5,
                "strict_match_percent": 100 * metrics["exact_match,strict-match"],
                "strict_match_stderr_percent": 100 * metrics["exact_match_stderr,strict-match"],
                "flexible_extract_percent": 100 * metrics["exact_match,flexible-extract"],
                "flexible_extract_stderr_percent": 100 * metrics["exact_match_stderr,flexible-extract"],
                "result": str(result_path),
                "result_sha256": sha256(result_path),
                "samples_file": str(samples_path),
                "samples_sha256": sha256(samples_path),
            }
        )

    payload = {
        "task": "GSM8K first-500 quick screen",
        "harness": "EleutherAI lm-evaluation-harness 0.4.12",
        "seed": [0, 1234, 1234, 1234],
        "rows": rows,
    }
    (args.results / "gsm8k-500-summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = [
        "| Format | GSM8K strict | GSM8K flexible | n |",
        "|---|---:|---:|---:|",
        *[
            f"| {row['condition']} | {row['strict_match_percent']:.2f} | "
            f"{row['flexible_extract_percent']:.2f} | {row['samples']} |"
            for row in rows
        ],
    ]
    (args.results / "gsm8k-500-summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


