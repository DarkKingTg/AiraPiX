from __future__ import annotations

import argparse
import fnmatch
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Missing dependency: pyyaml. Run `pip install -r requirements.txt`.") from exc


TOKENIZER_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.py",
    "tokenizer*",
    "vocab.*",
    "merges.txt",
    "generation_config.json",
    "chat_template.jinja",
    "README.md",
    "LICENSE*",
]

DEFAULT_IGNORE_PATTERNS = [
    "*.h5",
    "*.msgpack",
    "*.onnx",
    "*.tflite",
    "*.gguf",
]


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def selected_models(manifest: dict[str, Any], names: list[str], include_disabled: bool) -> list[dict[str, Any]]:
    models = manifest.get("models", [])
    if names == ["all"]:
        selected = models
    else:
        wanted = set(names)
        selected = [model for model in models if model["name"] in wanted or model["repo_id"] in wanted]
    selected = [model for model in selected if include_disabled or model.get("enabled", True)]
    if not selected:
        raise SystemExit(f"No teacher models matched names={names!r}.")
    return selected


def match_any(path: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def estimate_size_gb(info, allow_patterns: list[str] | None, ignore_patterns: list[str] | None) -> float:
    total = 0
    for sibling in info.siblings:
        filename = sibling.rfilename
        if ignore_patterns and match_any(filename, ignore_patterns):
            continue
        if allow_patterns and not match_any(filename, allow_patterns):
            continue
        total += sibling.size or 0
    return total / (1024**3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Aira teacher sub-models from Hugging Face.")
    parser.add_argument("--manifest", default="training/teacher_models.yaml")
    parser.add_argument("--names", nargs="+", default=["all"], help="Model names or repo IDs from the manifest.")
    parser.add_argument("--mode", choices=["metadata", "tokenizer", "full"], default="full")
    parser.add_argument("--cache-dir", default=None, help="Override manifest cache_dir.")
    parser.add_argument("--local-dir", default=None, help="Optional local snapshot directory root.")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub. Run `pip install -r requirements.txt`.") from exc

    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    cache_dir = Path(args.cache_dir or manifest.get("cache_dir") or "models/teachers/hf_cache")
    local_root_raw = args.local_dir if args.local_dir is not None else manifest.get("local_dir")
    local_root = Path(local_root_raw) if local_root_raw else None
    cache_dir.mkdir(parents=True, exist_ok=True)
    if local_root is not None:
        local_root.mkdir(parents=True, exist_ok=True)

    allow_patterns = TOKENIZER_PATTERNS if args.mode == "tokenizer" else None
    ignore_patterns = DEFAULT_IGNORE_PATTERNS
    api = HfApi(token=args.hf_token)
    selected = selected_models(manifest, args.names, args.include_disabled)

    print(
        json.dumps(
            {
                "manifest": str(manifest_path.resolve()),
                "mode": args.mode,
                "cache_dir": str(cache_dir.resolve()),
                "local_root": str(local_root.resolve()) if local_root is not None else None,
                "models": [model["name"] for model in selected],
            },
            indent=2,
        ),
        flush=True,
    )

    results = []
    for model in selected:
        name = model["name"]
        repo_id = model["repo_id"]
        print(f"[teacher-download] inspecting {name} ({repo_id})", flush=True)
        info = api.model_info(repo_id, files_metadata=True)
        estimated_gb = estimate_size_gb(info, allow_patterns, ignore_patterns)
        row = {
            "name": name,
            "repo_id": repo_id,
            "role": model.get("role"),
            "revision": info.sha,
            "mode": args.mode,
            "estimated_download_gb": round(estimated_gb, 3),
            "download_path": None,
            "status": "metadata",
        }
        print(json.dumps(row, indent=2), flush=True)
        if args.mode != "metadata":
            started = time.time()
            local_dir = str(local_root / name) if local_root is not None else None
            print(
                f"[teacher-download] downloading {name}; estimated={estimated_gb:.2f} GB; "
                f"cache={cache_dir}; local_dir={local_dir or '-'}",
                flush=True,
            )
            path = snapshot_download(
                repo_id=repo_id,
                revision=info.sha,
                cache_dir=str(cache_dir),
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                token=args.hf_token,
                max_workers=args.max_workers,
            )
            elapsed = time.time() - started
            row["download_path"] = str(Path(path).resolve())
            row["status"] = "downloaded"
            row["elapsed_min"] = round(elapsed / 60, 2)
            print(
                f"[teacher-download] done {name}; elapsed={elapsed / 60:.1f} min; path={path}",
                flush=True,
            )
        results.append(row)

    out_dir = cache_dir / "_airapix_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"teacher_download_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "results": results}, indent=2), encoding="utf-8")
    print(f"[teacher-download] manifest written: {out_path}", flush=True)


if __name__ == "__main__":
    main()
