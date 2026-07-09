from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def iter_text(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            parts = [row.get("input", ""), row.get("output", "")]
            text = "\n\n".join(part for part in parts if part)
            if text:
                yield text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Processed train.jsonl file.")
    parser.add_argument("--output", required=True, help="Path to tokenizer.json.")
    parser.add_argument("--vocab-size", type=int, default=12000)
    args = parser.parse_args()

    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    except ImportError as exc:
        raise SystemExit("Missing dependency: tokenizers. Run `pip install -r requirements.txt`.") from exc

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=SPECIAL_TOKENS)
    tokenizer.train_from_iterator(iter_text(Path(args.input)), trainer=trainer)

    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B <eos>",
        special_tokens=[("<bos>", bos_id), ("<eos>", eos_id)],
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    print(f"[tokenizer] wrote {out_path}")


if __name__ == "__main__":
    main()
