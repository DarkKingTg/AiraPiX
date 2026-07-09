from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from airapix.training.dataset import iter_jsonl


CATEGORIES = ("text", "chat", "reasoning", "code", "tool_use")


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Qwen teacher JSONL into processed training shards.")
    parser.add_argument("--base-dir", default="dataset_builder/data/processed")
    parser.add_argument("--teacher-jsonl", required=True)
    parser.add_argument("--output-dir", default="dataset_builder/data/processed_teacher")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    teacher_path = Path(args.teacher_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ["train.jsonl", "val.jsonl"]:
        copy_if_exists(base_dir / name, output_dir / name)
    for split in ["train", "val"]:
        for category in CATEGORIES:
            copy_if_exists(base_dir / f"{split}_{category}.jsonl", output_dir / f"{split}_{category}.jsonl")

    counts = Counter()
    for row in iter_jsonl(teacher_path):
        category = row.get("category", "chat")
        if category not in CATEGORIES:
            category = "chat"
            row["category"] = category
        append_jsonl(output_dir / "train.jsonl", row)
        append_jsonl(output_dir / f"train_{category}.jsonl", row)
        counts[category] += 1

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base_dir.resolve()),
        "teacher_jsonl": str(teacher_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "teacher_counts": dict(counts),
    }
    (output_dir / "teacher_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
