#!/usr/bin/env python3
import argparse
import hashlib
import json
import struct
from pathlib import Path

import pyarrow.parquet as pq


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def pack_answers(answers: list[str], label: int) -> bytes:
    labels = [int(index == label) for index in range(len(answers))]
    return (
        struct.pack("<I", len(answers))
        + b"".join(pack_string(answer) for answer in answers)
        + struct.pack(f"<{len(labels)}i", *labels)
    )


def pack_task(goal: str, answers: list[str], label: int) -> bytes:
    return pack_string(goal) + pack_answers(answers, label) + struct.pack("<I", 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    table = pq.read_table(args.input, columns=["goal", "sol1", "sol2", "label"])
    rows = table.to_pylist()
    if len(rows) != 1838:
        raise ValueError(f"expected 1838 PIQA validation rows, found {len(rows)}")

    tasks = []
    for index, row in enumerate(rows):
        goal = row["goal"].strip()
        answers = [row["sol1"].strip(), row["sol2"].strip()]
        label = int(row["label"])
        if not goal or any(not answer for answer in answers) or label not in (0, 1):
            raise ValueError(f"invalid PIQA row {index}")
        tasks.append(pack_task(goal, answers, label))

    header_size = 4 + 4 * len(tasks)
    positions = []
    offset = header_size
    for task in tasks:
        positions.append(offset)
        offset += len(task)

    payload = struct.pack("<I", len(tasks))
    payload += struct.pack(f"<{len(positions)}I", *positions)
    payload += b"".join(tasks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)

    manifest = {
        "dataset": "ybisk/piqa",
        "revision": "078a131412f46a38025a762322c174a8bae2610c",
        "split": "validation",
        "tasks": len(tasks),
        "choices_per_task": 2,
        "scoring": "length-normalized continuation log-likelihood",
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(payload).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


