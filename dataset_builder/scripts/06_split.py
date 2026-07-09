from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from common import CATEGORIES, iter_jsonl, load_config, pipeline_path


def open_outputs(processed_dir: Path, category_shards: bool):
    outputs = {
        "train": (processed_dir / "train.jsonl").open("w", encoding="utf-8", newline="\n"),
        "val": (processed_dir / "val.jsonl").open("w", encoding="utf-8", newline="\n"),
    }
    if category_shards:
        for split in ("train", "val"):
            for category in CATEGORIES:
                outputs[f"{split}_{category}"] = (
                    processed_dir / f"{split}_{category}.jsonl"
                ).open("w", encoding="utf-8", newline="\n")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_builder/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pipe = config["pipeline"]
    random.seed(int(pipe.get("seed", 42)))

    interim_dir = pipeline_path(config, "interim_dir")
    processed_dir = pipeline_path(config, "processed_dir")
    manifest_dir = pipeline_path(config, "manifest_dir")
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    in_path = interim_dir / "05_deduped.jsonl"
    train_fraction = float(pipe.get("train_fraction", 0.98))
    category_shards = bool(pipe.get("output_category_shards", True))
    outputs = open_outputs(processed_dir, category_shards)
    counts = defaultdict(Counter)
    ratio_sum = defaultdict(float)

    try:
        for example in iter_jsonl(in_path):
            split = "train" if random.random() < train_fraction else "val"
            line = json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n"
            outputs[split].write(line)
            category = example.get("category", "unknown")
            counts[split][category] += 1
            ratio_sum[category] += float(example.get("meta", {}).get("gzip_ratio", 0.0))
            if category_shards and category in CATEGORIES:
                outputs[f"{split}_{category}"].write(line)
    finally:
        for f in outputs.values():
            f.close()

    category_totals = Counter()
    for split_counts in counts.values():
        category_totals.update(split_counts)
    avg_gzip = {
        category: round(ratio_sum[category] / max(1, total), 4)
        for category, total in category_totals.items()
    }
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": config["_config_path"],
        "train_fraction": train_fraction,
        "counts": {split: dict(counter) for split, counter in counts.items()},
        "category_totals": dict(category_totals),
        "avg_gzip_ratio_by_category": avg_gzip,
        "outputs": {
            "train": str(processed_dir / "train.jsonl"),
            "val": str(processed_dir / "val.jsonl"),
        },
    }
    (manifest_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
