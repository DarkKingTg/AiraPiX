# AiraPix Phase 1-3 Bootstrap

AiraPix is a small, local-first AI companion project. This repository now starts
the first three phases:

- Phase 1: dataset builder for open Hugging Face datasets.
- Phase 2: raw PyTorch small language model with MLA-style attention and a
  pure-PyTorch SSM fallback block.
- Phase 3: pretraining loop with VRAM probing, RegMix-inspired data mixing,
  compression-ratio scoring, Qwen2.5-7B sequence distillation, and a
  Muon-plus-AdamW hybrid optimizer.

The defaults are conservative for a 4 GB VRAM target:

- Context length: `1024`
- Vocab size: `12000`
- Attention: MLA-style compressed KV attention on every fourth layer
- Other layers: pure-PyTorch diagonal SSM mixer
- Distillation: offline sequence-level generation from
  `Qwen/Qwen2.5-7B-Instruct`
- Optimizer: Muon for 2D hidden-layer matrices, AdamW for embeddings, norms,
  biases, and output heads

## Setup

PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py` is not installed, install Python 3.11 or 3.12 from python.org and make
sure it is added to PATH.

## Smoke Test

```powershell
python scripts\smoke_test.py
```

This only builds a tiny model and checks a forward/backward/optimizer step. It
does not download datasets.

## Phase 1 Dataset Builder

Edit `dataset_builder/config.yaml`, then run:

```powershell
python dataset_builder\run_all.py --config dataset_builder\config.yaml
```

The final training files are written under:

```text
dataset_builder/data/processed/
```

## Phase 2 Tokenizer

After Phase 1 has produced `train.jsonl`:

```powershell
python -m airapix.model.tokenizer.train_tokenizer `
  --input dataset_builder\data\processed\train.jsonl `
  --output airapix\model\tokenizer\tokenizer.json `
  --vocab-size 12000
```

## Phase 3 Training

Probe VRAM first:

```powershell
python -m airapix.training.probe_vram --config training\config.yaml
```

Then train:

```powershell
python -m airapix.training.train --config training\config.yaml
```

## Google Colab

For Colab, use the single runner:

```bash
python colab/run_colab_training.py --max-steps 1000 --log-every 5
```

To generate Qwen2.5-7B teacher data first:

```bash
python colab/run_colab_training.py --distill --distill-limit 1000 --max-steps 1000 --log-every 5
```

See `colab/README.md` for options. The Colab runner prints runtime details,
dataset counts, teacher-generation progress, loss, learning rate, token
throughput, ETA, checkpoint events, and GPU memory directly in the CLI.

Optional offline distillation from Qwen2.5-7B:

```powershell
python -m airapix.training.distill_from_qwen `
  --config training\config.yaml `
  --input dataset_builder\data\processed\train_chat.jsonl `
  --output dataset_builder\data\processed\distilled_qwen25_7b.jsonl
```

Running the 7B teacher locally is not realistic on a 4 GB GPU unless you use CPU
offload or quantization. The script is designed for Colab or any machine with
enough memory.

## Research Notes

The research assumptions and current model references used for this bootstrap
are recorded in `docs/research_notes.md`.
