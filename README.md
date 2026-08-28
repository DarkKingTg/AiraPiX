# AiraPiX 🚀

> **AiraAI**: A High-Efficiency, Scalable Hybrid Language Model Architecture (1.5B – 8B parameters) designed for high performance on Frontier Cloud GPUs and low-resource Edge Deployment (4GB VRAM / 2GB RAM).

---

## 🌟 Key Highlights

- **Multi-Scale Model Presets**: Support for `1.5B`, `3B`, `7B`, and **`8B`** parameter models (`d_model=4096`, `n_layers=32`, `n_heads=32`).
- **Hybrid MLA + Diagonal SSM Architecture**:
  - **MLA (Multi-Head Latent Attention)**: Compressed Key-Value latent projections with decoupled Rotary Position Embeddings (RoPE) every 4th layer.
  - **Diagonal SSM**: Linear-complexity state-space mixer for long-context sequence modeling on intermediate layers.
- **CPU-to-CUDA 4-Bit Block Streaming**: Zero-spike 4-bit NF4 quantization (`--load-in-4bit --use-qlora`) streams layer blocks from CPU RAM to CUDA GPU, allowing an **8B parameter model to train on a single 15GB Tesla T4 GPU** (peak initialization VRAM < 4.0 GB).
- **Gold-Standard Pure Dataset Pipeline**: Built-in automated ingestion, deduplication, and quality filtering for:
  - **FineWeb-Edu**: High-quality educational text & reasoning
  - **UltraChat 200k**: Multi-turn dialogue synthesis
  - **Orca Math 200k**: Chain-of-Thought mathematical reasoning
  - **CodeFeedback & CodeAlpaca**: Multi-language programming & code instructions
  - **Hermes & Glaive Function Calling**: Structured JSON tool use and agentic workflows
- **Frontier Model Distillation**: Distill reasoning capabilities from frontier teacher models (`Qwen2.5-72B-Instruct`, `DeepSeek-R1-Distill-Qwen-32B`, `Llama-3.3-70B-Instruct`).
- **Muon + AdamW Hybrid Optimizer**: Muon optimizer for 2D weight matrices combined with AdamW for 1D vectors, embeddings, and normalization layers.
- **Edge Quantization Exporter**: Export trained 4-bit checkpoints for edge execution (`scripts/export_quantized.py`).

---

## 🛠️ Installation & Setup

### Local Setup (Windows / Linux)

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate environment (Linux / macOS)
source .venv/bin/activate

# Upgrade pip & install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## ⚡ Quick Start: Google Colab Training (8B Model on T4 GPU)

Train an **8B parameter model** using QLoRA in Google Colab (Tesla T4 15GB VRAM):

```bash
# 1. Clone repository & navigate to directory
!git clone https://github.com/DarkKingTg/AiraPiX.git
%cd AiraPiX
!git checkout codex/aira-training-setup

# 2. Run automated Colab training pipeline
!python colab/run_colab_training.py \
  --preset 8b \
  --load-in-4bit \
  --use-qlora \
  --base-training-config training/config.colab_8b_qlora.yaml \
  --micro-batch-size 2 \
  --effective-batch-size 32 \
  --skip-probe \
  --mount-drive
```

---

## 📊 Dataset Pipeline

Run the end-to-end dataset builder (download, normalize, clean, filter, MinHash dedup, split):

```bash
python dataset_builder/run_all.py --config dataset_builder/config.trainable_v0.yaml
```

The processed data will be saved under:
```text
dataset_builder/data/processed/
├── train.jsonl
├── val.jsonl
├── train_text.jsonl
├── train_chat.jsonl
├── train_reasoning.jsonl
├── train_code.jsonl
└── train_tool_use.jsonl
```

---

## 🔤 Tokenizer Training

Train a customized Byte-Pair Encoding (BPE) tokenizer:

```bash
python -m airapix.model.tokenizer.train_tokenizer \
  --input dataset_builder/data/processed/train.jsonl \
  --output airapix/model/tokenizer/tokenizer.json \
  --vocab-size 12000
```

---

## 🏋️ Training & Distillation

### 1. Direct Training

Train locally or on a server:

```bash
python -m airapix.training.train \
  --config training/config.colab_8b_qlora.yaml \
  --preset 8b \
  --load-in-4bit \
  --use-qlora
```

### 2. Frontier Teacher Model Distillation

Download teacher model weights (`Qwen2.5-72B`, `DeepSeek-R1-32B`, `Llama-3.3-70B`) and run offline sequence distillation:

```bash
# Check/download teacher models
python colab/run_colab_training.py --download-teachers --skip-dataset --skip-train

# Run offline teacher distillation
python -m airapix.training.distill_from_qwen \
  --config training/config.yaml \
  --input dataset_builder/data/processed/train_chat.jsonl \
  --output dataset_builder/data/processed/distilled_teacher.jsonl
```

---

## 💬 Inference & Response Generation

Generate responses using a trained checkpoint or model architecture:

```bash
python airapix/inference/generate.py \
  --preset 8b \
  --prompt "Explain quantum entanglement in simple terms." \
  --max-tokens 256 \
  --temperature 0.7
```

---

## 📦 Quantization & Edge Export

Export trained weights into 4-bit NF4/INT4 for low-resource deployment (4GB VRAM / 2GB RAM):

```bash
python scripts/export_quantized.py \
  --checkpoint path/to/model.pt \
  --output path/to/airapix_8b_4bit.safetensors
```

---

## 📄 License & Research Notes

Architectural research notes and benchmark comparisons are documented under `docs/research_notes.md`.
