# AiraPix Setup Status

Updated: 2026-07-09

## Environment

- Virtual environment: `.venv`
- PyTorch: `2.11.0+cu128`
- CUDA runtime: `12.8`
- GPU verified: `NVIDIA GeForce RTX 2050`
- CUDA tensor test: passed
- Package health: `pip check` passed

## Dataset

The current trainable v0 dataset was built with:

```powershell
.\.venv\Scripts\python.exe dataset_builder\run_all.py --config dataset_builder\config.trainable_v0.yaml
```

Outputs:

- `dataset_builder/data/processed/train.jsonl`
- `dataset_builder/data/processed/val.jsonl`
- category shards for text, chat, reasoning, code, and tool-use
- `dataset_builder/data/manifests/dataset_manifest.json`

Current cleaned total: `53,193` examples.

## Tokenizer

The tokenizer was trained with:

```powershell
.\.venv\Scripts\python.exe -m airapix.model.tokenizer.train_tokenizer `
  --input dataset_builder\data\processed\train.jsonl `
  --output airapix\model\tokenizer\tokenizer.json `
  --vocab-size 12000
```

Verified vocab size: `12,000`.

## Training

The main training config is now:

```text
training/config.yaml
```

It uses:

- preset: `125m`
- measured params: about `101M`
- context: `1024`
- micro-batch: `4`
- effective batch: `64`
- precision: `auto` (`bf16` on the RTX 2050 in the smoke test)
- optimizer: Muon + AdamW

Start training:

```powershell
.\.venv\Scripts\python.exe -m airapix.training.train --config training\config.yaml
```

The verified one-step GPU smoke test used:

```powershell
.\.venv\Scripts\python.exe -m airapix.training.train --config training\config.rtx2050_125m_smoke.yaml
```

Smoke checkpoint:

```text
runs/rtx2050_125m_smoke/checkpoints/final.pt
```

## Qwen Distillation

The distillation script and dependencies are installed, but Qwen2.5-7B should
normally be run on Colab or a larger machine. The RTX 2050 has only 4 GB VRAM.

Teacher sub-model manifest:

```text
training/teacher_models.yaml
```

Download/check teacher models:

```powershell
.\.venv\Scripts\python.exe -m airapix.training.download_teacher_models --manifest training\teacher_models.yaml --mode metadata
.\.venv\Scripts\python.exe -m airapix.training.download_teacher_models --manifest training\teacher_models.yaml --mode full
```

Script:

```powershell
.\.venv\Scripts\python.exe -m airapix.training.distill_from_qwen `
  --config training\config.yaml `
  --input dataset_builder\data\processed\train_chat.jsonl `
  --output dataset_builder\data\processed\distilled_qwen25_7b.jsonl
```
