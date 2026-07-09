from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from common import enabled_datasets, ensure_dirs, load_config, pipeline_path


def download_one(spec: dict, raw_dir: Path) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Run `pip install -r requirements.txt`.") from exc

    name = spec["name"]
    out_path = raw_dir / f"{name}.jsonl"
    kwargs = {
        "path": spec["hf_name"],
        "split": spec.get("split", "train"),
        "streaming": bool(spec.get("streaming", True)),
    }
    if spec.get("hf_config"):
        kwargs["name"] = spec["hf_config"]

    dataset = load_dataset(**kwargs)
    sample_limit = spec.get("sample_limit")
    rows = dataset if sample_limit is None else islice(dataset, int(sample_limit))

    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            row = dict(row)
            row["_source_name"] = name
            row["_category"] = spec["category"]
            row["_format"] = spec["format"]
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1

    return {"name": name, "raw_path": str(out_path), "records": count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_builder/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = pipeline_path(config, "raw_dir")
    manifest_dir = pipeline_path(config, "manifest_dir")
    ensure_dirs(raw_dir, manifest_dir)

    manifest = {"stage": "download", "datasets": []}
    for spec in enabled_datasets(config):
        print(f"[download] {spec['name']} from {spec['hf_name']}")
        manifest["datasets"].append(download_one(spec, raw_dir))

    (manifest_dir / "01_download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
