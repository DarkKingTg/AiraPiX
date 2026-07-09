from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from common import combined_text, iter_jsonl, load_config, pipeline_path, text_hash

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:  # pragma: no cover - exact dedup fallback works.
    MinHash = None
    MinHashLSH = None


def shingles(text: str, size: int) -> list[str]:
    tokens = re.findall(r"\w+", text.casefold())
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def make_minhash(text: str, num_perm: int, shingle_size: int):
    mh = MinHash(num_perm=num_perm)
    for shingle in shingles(text, shingle_size):
        mh.update(shingle.encode("utf-8", errors="ignore"))
    return mh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_builder/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dedup_cfg = config["pipeline"]["dedup"]
    interim_dir = pipeline_path(config, "interim_dir")
    in_path = interim_dir / "04_filtered.jsonl"
    out_path = interim_dir / "05_deduped.jsonl"

    use_minhash = dedup_cfg.get("method") == "minhash" and MinHash is not None
    lsh = None
    if use_minhash:
        lsh = MinHashLSH(
            threshold=float(dedup_cfg.get("threshold", 0.86)),
            num_perm=int(dedup_cfg.get("num_perm", 64)),
        )

    seen_hashes: set[str] = set()
    stats = {"seen": 0, "kept": 0, "exact_duplicates": 0, "near_duplicates": 0}

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for example in iter_jsonl(in_path):
            stats["seen"] += 1
            text = combined_text(example)
            digest = text_hash(text)
            if digest in seen_hashes:
                stats["exact_duplicates"] += 1
                continue
            seen_hashes.add(digest)

            if use_minhash and lsh is not None:
                mh = make_minhash(
                    text,
                    num_perm=int(dedup_cfg.get("num_perm", 64)),
                    shingle_size=int(dedup_cfg.get("shingle_size", 5)),
                )
                matches = lsh.query(mh)
                if matches:
                    stats["near_duplicates"] += 1
                    continue
                lsh.insert(example["id"], mh)

            out.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
            stats["kept"] += 1

    print(json.dumps({"stage": "dedup", "minhash": use_minhash, **stats}, indent=2))
    print(f"[dedup] wrote {stats['kept']} examples to {out_path}")


if __name__ == "__main__":
    main()
