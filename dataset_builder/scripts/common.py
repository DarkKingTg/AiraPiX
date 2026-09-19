from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - setup guidance handles this.
    raise SystemExit("Missing dependency: pyyaml. Run `pip install -r requirements.txt`.") from exc


CATEGORIES = ("text", "chat", "reasoning", "code", "tool_use")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path.resolve())
    config["_base_dir"] = str(config_path.resolve().parent)
    return config


def base_dir(config: dict[str, Any]) -> Path:
    return Path(config["_base_dir"])


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir(config) / path


def pipeline_path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config, config["pipeline"][key])


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists() or path_obj.stat().st_size == 0:
        return
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_uuid(source: str, payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{source}\n{raw}".encode("utf-8", errors="ignore")).hexdigest()
    return str(uuid.UUID(digest[:32]))


def estimate_tokens(text: str) -> int:
    # English byte-level BPE commonly lands near 3-5 chars/token. This estimate
    # is used only for filtering/manifests before the real tokenizer exists.
    return max(1, math.ceil(len(text) / 4))


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False, sort_keys=True)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def combined_text(example: dict[str, Any]) -> str:
    parts = [example.get("input", ""), example.get("output", "")]
    return "\n".join(str(part) for part in parts if part)


def gzip_ratio(text: str) -> float:
    raw = text.encode("utf-8", errors="ignore")
    if not raw:
        return 1.0
    compressed = gzip.compress(raw, compresslevel=6)
    return len(compressed) / max(1, len(raw))


def compression_quality_score(
    ratio: float,
    preferred_min: float,
    preferred_max: float,
    hard_min: float,
    hard_max: float,
) -> float:
    if ratio < hard_min or ratio > hard_max:
        return 0.0
    if preferred_min <= ratio <= preferred_max:
        return 1.0
    if ratio < preferred_min:
        span = max(1e-6, preferred_min - hard_min)
        return max(0.0, (ratio - hard_min) / span)
    span = max(1e-6, hard_max - preferred_max)
    return max(0.0, (hard_max - ratio) / span)


def text_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.casefold()).strip()
    return hashlib.sha1(normalized.encode("utf-8", errors="ignore")).hexdigest()


def get_field(row: dict[str, Any], field: str | None, default: str = "") -> Any:
    if not field:
        return default
    current: Any = row
    for part in field.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def enabled_datasets(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in config.get("datasets", []) if item.get("enabled", True)]
