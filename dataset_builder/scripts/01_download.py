from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
import os
import socket
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
socket.setdefaulttimeout(45.0)

sys.path.append(str(Path(__file__).resolve().parent))
from common import enabled_datasets, ensure_dirs, load_config, pipeline_path


def download_one(spec: dict, raw_dir: Path) -> dict:
    import time
    import gc
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets. Run `pip install -r requirements.txt`.") from exc

    name = spec["name"]
    sample_limit = spec.get("sample_limit")
    limit = int(sample_limit) if sample_limit is not None else None

    out_path = raw_dir / f"{name}.jsonl"
    if out_path.exists() and out_path.stat().st_size > 0:
        with out_path.open("r", encoding="utf-8") as f:
            existing_count = sum(1 for line in f if line.strip())
        if limit is None or existing_count >= limit:
            print(f"[download] {name} already exists with {existing_count} rows. Skipping download.")
            return {"name": name, "raw_path": str(out_path), "records": existing_count}
        else:
            print(f"[download] {name} currently has {existing_count:,} rows, but config requests {limit:,}. Re-downloading fresh sample...")

    kwargs = {
        "path": spec["hf_name"],
        "split": spec.get("split", "train"),
        "streaming": bool(spec.get("streaming", True)),
    }
    if spec.get("hf_config"):
        kwargs["name"] = spec["hf_config"]
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    if spec.get("trust_remote_code"):
        kwargs["trust_remote_code"] = True

    dataset = None
    try:
        dataset = load_dataset(**kwargs)
    except Exception as err:
        # If loading the default branch fails on streaming (e.g. monolithic json), try HF parquet mirror
        if kwargs.get("streaming") and "revision" not in kwargs:
            print(f"  [fallback] Primary stream failed ({err}). Trying HF Parquet mirror...")
            fallback_kw = dict(kwargs)
            fallback_kw["revision"] = "refs/convert/parquet"
            try:
                dataset = load_dataset(**fallback_kw)
                kwargs = fallback_kw
            except Exception:
                raise err
        else:
            raise err

    count = 0
    max_retries = 3
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        iterator = iter(dataset)
        consecutive_errors = 0
        while limit is None or count < limit:
            try:
                row = next(iterator)
                consecutive_errors = 0
            except StopIteration:
                break
            except Exception as stream_err:
                consecutive_errors += 1
                if consecutive_errors > max_retries:
                    print(f"  [stream stop] Exceeded retries on '{name}' ({stream_err}). Stopping at {count} rows.")
                    break
                print(f"  [retry {consecutive_errors}/{max_retries}] Stream pause ({stream_err}). Retrying in 2s...")
                time.sleep(2)
                try:
                    dataset = load_dataset(**kwargs)
                    iterator = iter(dataset)
                    row = next(iterator)
                except Exception:
                    continue

            row_dict = dict(row)
            row_dict["_source_name"] = name
            row_dict["_category"] = spec["category"]
            row_dict["_format"] = spec["format"]
            f.write(json.dumps(row_dict, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            if count % 10000 == 0:
                f.flush()
                target_str = f" / {limit:,}" if limit else ""
                print(f"  [stream {name}] {count:,}{target_str} rows...", flush=True)

    del dataset
    gc.collect()
    try:
        import pyarrow
        pyarrow.default_memory_pool().release_unused()
    except Exception:
        pass

    print(f"[download] {name} finished with {count:,} rows.", flush=True)
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
