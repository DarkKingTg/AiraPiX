from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from common import enabled_datasets, ensure_dirs, load_config, pipeline_path


def download_one(spec: dict, raw_dir: Path) -> dict:
    import time
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Run `pip install -r requirements.txt`.") from exc

    name = spec["name"]
    out_path = raw_dir / f"{name}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open("r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())
        print(f"[download] {name} already exists with {existing_count} rows. Skipping download.")
        return {"name": name, "raw_path": str(out_path), "records": existing_count}

    kwargs = {
        "path": spec["hf_name"],
        "split": spec.get("split", "train"),
        "streaming": bool(spec.get("streaming", True)),
        "trust_remote_code": True,
    }
    if spec.get("hf_config"):
        kwargs["name"] = spec["hf_config"]

    max_retries = 3
    dataset = None
    try:
        dataset = load_dataset(**kwargs)
    except Exception as err:
        if kwargs.get("streaming"):
            print(f"  [streaming fallback] Streaming failed ({err}). Trying streaming=False...")
            kwargs["streaming"] = False
            dataset = load_dataset(**kwargs)
        else:
            raise err

    sample_limit = spec.get("sample_limit")
    
    count = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        rows = dataset if sample_limit is None else islice(dataset, int(sample_limit))
        for attempt in range(1, max_retries + 1):
            try:
                for row in rows:
                    row_dict = dict(row)
                    row_dict["_source_name"] = name
                    row_dict["_category"] = spec["category"]
                    row_dict["_format"] = spec["format"]
                    f.write(json.dumps(row_dict, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
                break
            except Exception as err:
                if attempt == max_retries:
                    raise err
                print(f"  [retry {attempt}/{max_retries}] Error streaming rows ({err}). Retrying in {attempt * 3}s...")
                time.sleep(attempt * 3)

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
        try:
            res = download_one(spec, raw_dir)
            manifest["datasets"].append(res)
        except Exception as err:
            print(f"[download] WARNING: Skipping dataset '{spec['name']}' due to error: {err}", flush=True)

    (manifest_dir / "01_download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
