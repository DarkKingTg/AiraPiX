import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from airapix.training.dataset import render_example, encode_for_lm, MixtureJsonlDataset, JsonlLMDataset


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        return [ord(c) % 500 + 10 for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr((i - 10) % 500) for i in ids if i not in (0, 1, 2))


def test_render_example_chat():
    row = {"category": "chat", "input": "Hello", "output": "Hi there!"}
    prompt, response = render_example(row)
    assert prompt == "User:\nHello\n\nAssistant:\n"
    assert response == "Hi there!"


def test_render_example_text():
    row = {"category": "text", "input": "", "output": "Some plain text standard paragraph."}
    prompt, response = render_example(row)
    assert prompt == ""
    assert response == "Some plain text standard paragraph."


def test_prompt_loss_masking():
    tokenizer = DummyTokenizer()
    example = ("User:\nHello\n\nAssistant:\n", "Hi there!")
    context_len = 32

    input_ids, labels = encode_for_lm(tokenizer, example, context_len=context_len, mask_prompt=True)
    prompt_ids = tokenizer.encode(example[0])
    num_prompt = len(prompt_ids)

    # Check prompt position labels are masked with -100
    assert torch.all(labels[:num_prompt] == -100)
    # Check response position labels equal input_ids
    assert torch.all(labels[num_prompt:len(tokenizer.encode(example[0] + example[1]))] == input_ids[num_prompt:len(tokenizer.encode(example[0] + example[1]))])
    # Check padding position labels are -100
    assert torch.all(labels[len(tokenizer.encode(example[0] + example[1])):] == -100)


def test_mixture_dataset_empty_shard_filtering():
    tokenizer = DummyTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        valid_shard = tmp_path / "valid.jsonl"
        empty_shard = tmp_path / "empty.jsonl"

        valid_shard.write_text('{"category": "chat", "input": "Q", "output": "A"}\n', encoding="utf-8")
        empty_shard.write_text('', encoding="utf-8")

        category_paths = {
            "chat": valid_shard,
            "tool_use": empty_shard,
        }
        weights = {"chat": 0.5, "tool_use": 0.5}

        ds = MixtureJsonlDataset(
            category_paths=category_paths,
            weights=weights,
            tokenizer=tokenizer,
            context_len=32,
            mask_prompt=True,
        )

        # Empty shard should be safely filtered out
        assert "tool_use" not in ds.category_paths
        assert "chat" in ds.category_paths

        # Iteration should produce items from valid shard without crashing
        sample = next(iter(ds))
        assert "input_ids" in sample
        assert "labels" in sample


if __name__ == "__main__":
    test_render_example_chat()
    test_render_example_text()
    test_prompt_loss_masking()
    test_mixture_dataset_empty_shard_filtering()
    print("ALL TESTS PASSED SUCCESSFULLY!")
