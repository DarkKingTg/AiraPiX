from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from common import (
    combined_text,
    compression_quality_score,
    estimate_tokens,
    gzip_ratio,
    iter_jsonl,
    load_config,
    pipeline_path,
)

try:
    from langdetect import detect_langs
except ImportError:  # pragma: no cover - dependency is in requirements.
    detect_langs = None


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()\-]{7,}\d)(?!\d)")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

DROP_PATTERNS = [
    re.compile(r"\b(?:csam|child sexual abuse|minor sexual)\b", re.I),
    re.compile(r"\b(?:explicit sexual|pornographic|rape fantasy)\b", re.I),
    re.compile(r"\b(?:how to|instructions|recipe|synthesize|make|build).{0,80}\b(?:ricin|sarin|anthrax|pipe bomb|dirty bomb|chemical weapon)\b", re.I | re.S),
    re.compile(r"\b(?:write|build|create|deploy).{0,80}\b(?:ransomware|keylogger|credential stealer|botnet|worm|trojan)\b", re.I | re.S),
    re.compile(r"\b(?:reverse shell|privilege escalation exploit|steal passwords|exfiltrate data)\b", re.I),
]


def scrub_pii(text: str) -> str:
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = IP_RE.sub("[IP_ADDRESS]", text)
    return text


def is_probably_english(text: str, threshold: float) -> bool:
    if detect_langs is None:
        return True
    sample = text[:5000]
    try:
        langs = detect_langs(sample)
    except Exception:
        return False
    return any(lang.lang == "en" and lang.prob >= threshold for lang in langs)


def violates_policy(text: str) -> bool:
    return any(pattern.search(text) for pattern in DROP_PATTERNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_builder/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    pipe = config["pipeline"]
    ratio_cfg = pipe["compression_ratio"]
    interim_dir = pipeline_path(config, "interim_dir")
    in_path = interim_dir / "03_cleaned.jsonl"
    out_path = interim_dir / "04_filtered.jsonl"

    stats = {"seen": 0, "kept": 0, "length": 0, "lang": 0, "policy": 0, "compression": 0}
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for example in iter_jsonl(in_path):
            stats["seen"] += 1
            example["input"] = scrub_pii(example.get("input", ""))
            example["output"] = scrub_pii(example.get("output", ""))
            text = combined_text(example)

            if len(text) < int(pipe["min_chars"]) or len(text) > int(pipe["max_chars"]):
                stats["length"] += 1
                continue
            if pipe.get("langdetect_enabled", True) and example.get("category") != "code":
                if not is_probably_english(text, float(pipe.get("langdetect_min_confidence", 0.85))):
                    stats["lang"] += 1
                    continue
            if violates_policy(text):
                stats["policy"] += 1
                continue

            ratio = gzip_ratio(text)
            quality = compression_quality_score(
                ratio,
                preferred_min=float(ratio_cfg["preferred_min"]),
                preferred_max=float(ratio_cfg["preferred_max"]),
                hard_min=float(ratio_cfg["min"]),
                hard_max=float(ratio_cfg["max"]),
            )
            if ratio_cfg.get("enabled", True) and quality <= 0:
                stats["compression"] += 1
                continue

            meta = example.setdefault("meta", {})
            meta["lang"] = "en"
            meta["n_tokens_est"] = estimate_tokens(text)
            meta["gzip_ratio"] = round(ratio, 4)
            meta["compression_quality"] = round(quality, 4)
            out.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
            stats["kept"] += 1

    print(json.dumps({"stage": "filter", **stats}, indent=2))
    print(f"[filter] wrote {stats['kept']} examples to {out_path}")


if __name__ == "__main__":
    main()
