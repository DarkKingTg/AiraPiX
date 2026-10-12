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


def make_minhash(text: str, num_perm: int, shingle_size: int):
    raw_words = text.lower().split()
    if not raw_words:
        return MinHash(num_perm=num_perm)
    if len(raw_words) < shingle_size:
        shingles = [b" ".join(w.encode("utf-8", errors="ignore") for w in raw_words)]
    else:
        encoded_words = [w.encode("utf-8", errors="ignore") for w in raw_words]
        shingles = [
            b" ".join(encoded_words[i : i + shingle_size])
            for i in range(len(encoded_words) - shingle_size + 1)
        ]
    mh = MinHash(num_perm=num_perm)
    if shingles:
        mh.update_batch(shingles)
    return mh


def main() -> None:
    import time
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

    num_perm = int(dedup_cfg.get("num_perm", 64))
    shingle_size = int(dedup_cfg.get("shingle_size", 5))

    start_time = time.time()
    last_log_time = start_time

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
                mh = make_minhash(text, num_perm=num_perm, shingle_size=shingle_size)
                matches = lsh.query(mh)
                if matches:
                    stats["near_duplicates"] += 1
                    continue
                lsh.insert(example["id"], mh)

            out.write(json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n")
            stats["kept"] += 1

            if stats["seen"] % 10000 == 0:
                now = time.time()
                rate = stats["seen"] / max(0.01, now - start_time)
                print(f"[dedup] processed {stats['seen']:,} | kept: {stats['kept']:,} | near_dup: {stats['near_duplicates']:,} ({rate:.0f} docs/sec)", flush=True)
                out.flush()

    total_time = max(0.01, time.time() - start_time)
    print(json.dumps({"stage": "dedup", "minhash": use_minhash, "elapsed_sec": round(total_time, 2), "docs_per_sec": round(stats["seen"] / total_time, 1), **stats}, indent=2))
    print(f"[dedup] wrote {stats['kept']} examples to {out_path} in {total_time:.1f}s")


if __name__ == "__main__":
    main()
