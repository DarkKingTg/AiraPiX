# Colab Runner

Upload or clone this repository into Colab, then run:

```bash
python colab/run_colab_training.py --max-steps 1000 --log-every 5
```

For a quick test:

```bash
python colab/run_colab_training.py --max-steps 5 --log-every 1
```

To also generate Qwen2.5-7B teacher data:

```bash
python colab/run_colab_training.py --distill --distill-limit 1000 --max-steps 1000 --log-every 5
```

Remove `--distill-limit 1000` to generate teacher data for all selected rows.
That can take a long time and may exceed free Colab session limits.

Useful options:

- `--mount-drive`: mount Google Drive and copy outputs to
  `/content/drive/MyDrive/AiraPix`.
- `--skip-dataset`: reuse existing processed JSONL files.
- `--skip-tokenizer`: reuse existing `tokenizer.json`.
- `--skip-probe`: skip VRAM probing.
- `--skip-train`: only build/download data and tokenizer.
- `--distill-inputs train_chat.jsonl train_reasoning.jsonl train_tool_use.jsonl`:
  choose which shards to feed to Qwen.
- `--hf-token <token>`: pass a Hugging Face token for higher rate limits or
  gated models.

The runner prints commands, dataset counts, runtime/GPU information, training
loss, validation loss, learning rate, token throughput, ETA, and GPU memory to
the Colab CLI output.
