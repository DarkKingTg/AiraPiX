from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from common import iter_jsonl, load_config, normalize_space, pipeline_path

try:
    import ftfy
except ImportError:  # pragma: no cover - optional but recommended.
    ftfy = None


HTML_TAG_RE = re.compile(r"<[^>]+>")
MARKDOWN_NOISE_RE = re.compile(r"!\[[^\]]*]\([^)]*\)|\[(.*?)\]\([^)]*\)")


def clean_text(text: str) -> str:
    if ftfy is not None:
        text = ftfy.fix_text(text)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = MARKDOWN_NOISE_RE.sub(r"\1", text)
    text = re.sub(r"`{3,}", "```", text)
    return normalize_space(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_builder/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    interim_dir = pipeline_path(config, "interim_dir")
    in_path = interim_dir / "02_normalized.jsonl"
    out_path = interim_dir / "03_cleaned.jsonl"

    kept = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for example in iter_jsonl(in_path):
            example["input"] = clean_text(example.get("input", ""))
            example["output"] = clean_text(example.get("output", ""))
            if not example["output"]:
                continue
            out.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
            kept += 1

    print(f"[clean] wrote {kept} examples to {out_path}")


if __name__ == "__main__":
    main()
