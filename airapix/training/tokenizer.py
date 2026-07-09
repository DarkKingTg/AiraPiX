from __future__ import annotations

from pathlib import Path


class TokenizerWrapper:
    def __init__(self, tokenizer_path: str | Path) -> None:
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit("Missing dependency: tokenizers. Run `pip install -r requirements.txt`.") from exc
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.pad_token_id = self.tokenizer.token_to_id("<pad>") or 0
        self.bos_token_id = self.tokenizer.token_to_id("<bos>") or 1
        self.eos_token_id = self.tokenizer.token_to_id("<eos>") or 2

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
