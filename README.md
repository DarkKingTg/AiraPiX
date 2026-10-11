# AiraPiX

> **AiraAI**: An Autonomous Cognitive Brain & Hybrid State-Space Language Model Architecture (125M – 8B parameters) combining Dual-System Reasoning, Hebbian Neuroplasticity, Doya Neuromodulation, Damasio Somatic Markers, and Continual Lifelong Learning.

<p align="center">
  <img src="docs/airapix_architecture.svg" alt="AiraPiX Cognitive Brain &amp; Continual Architecture Diagram" width="100%">
</p>

---

## Key Highlights

- **24,000 ByteLevel BPE Vocabulary with Native Deliberation**:
  - Upgraded 24k tokenizer featuring native `<think>` (Token ID: 4) and `</think>` (Token ID: 5) boundary tokens.
  - Enables chain-of-thought inner monologue before delivering conversational replies.
- **Experiential Hebbian Neuroplasticity (Zero-Backprop Runtime Learning)**:
  - Online associative synaptic engine (`AffectiveSynapseEngine`) that updates connection weights in real time from conversational feedback:
    ```
    Delta_W = learning_rate * co_activation * neuromodulator_reward
    ```
  - Adapts instantly to corrections and persists associative memory to disk across restarts.
- **Doya 4-Chemical Neuromodulation & Damasio Somatic Markers**:
  - Kenji Doya 4-chemical neuromodulation (Dopamine, Serotonin, Noradrenaline, Acetylcholine) + Oxytocin and Cortisol.
  - Antonio Damasio Somatic Markers provide intuitive gut checks to flag deceptive, destructive, or hallucinated operations before execution.
- **Affective Theory of Mind (ToM) Perception Head**:
  - Classifies 8 emotional dimensions (*Joy, Frustration, Burnout, Imposter Syndrome, Crisis, Fear, Confusion, Urgency*).
  - Tracks emotional valence, arousal intensity, and unstated epistemic needs to ground conversational tone.
- **Dual-Track Epistemic Trust System**:
  - **Primary User Track**: Starts at 50% baseline and scales up to 100% through repeated collaborations.
  - **Peer AI Model Track**: Starts at 20% and is strictly hard-capped at 60%, enforcing mandatory verification against external AI hallucinations and prompt injection.
- **Persistent Obsidian Memory Vault**:
  - High-performance in-memory BM25 inverted index with query-centered excerpt extraction.
  - Somatic valence weighting (high-dopamine memories prioritized).
  - Automatically records conversation transcripts with YAML frontmatter to `Conversations/` and links session recaps into today's Daily Note (`01 - Daily Notes/`).
- **Continual Lifelong Learning (Anti-Catastrophic Forgetting)**:
  - **Autonomous Dream Cycles (Strategy 5)**: Offline self-play engine (`AiraDreamEngine`) that harvests vault seeds, synthesizes step-by-step `<think>` deliberation traces, validates them with somatic markers, and deposits them into the training queue.
  - **80/20 Experience Replay Mixture (Strategy 7)**: Blends 80% foundational pretraining tokens with 20% fresh dynamic experience tokens.
  - **Elastic Weight Consolidation (EWC)**: Computes the empirical Fisher Information matrix diagonal to protect critical parameter pathways from drift.
- **Hybrid MLA + Diagonal SSM Architecture**:
  - **MLA (Multi-Head Latent Attention)**: Compressed Key-Value latent projections with decoupled Rotary Position Embeddings (RoPE) every 4th layer (93% KV cache savings).
  - **Diagonal SSM**: Linear-complexity state-space mixer for long-context sequence modeling on intermediate layers.
- **Muon + AdamW Hybrid Optimizer**: Muon optimizer for 2D weight matrices combined with AdamW for 1D vectors, embeddings, and normalization layers.

---

## Installation & Setup

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

## Interactive Cognitive Agent Interface (CLI)

Launch Aira's real-time Dual-System cognitive agent session with live ASCII neurochemical, trust, and drive HUDs:

```bash
python -m airapix.agent.chat
```

### Cognitive Interface Commands

| Command | Description |
| :--- | :--- |
| `/reward [note]` | Awards a dopamine surge (+0.25) and strengthens associative synapses |
| `/penalty [note]` | Raises vigilance (noradrenaline) and stress (cortisol) on flagged errors |
| `/correct <emotion>` | Performs online synaptic recalibration when teaching Aira your true state |
| `/dream [count]` | Runs an autonomous self-play dream consolidation cycle (Strategy 5) |
| `/history` | Displays previous chat sessions with timestamps, topics, and important notes |
| `/note <text>` | Manually registers an important constraint or note for this session |
| `/decision <text>` | Manually records a milestone decision for this session |
| `/vault <query>` | Queries the Obsidian memory vault directly with BM25 + somatic scoring |
| `/trust` | Audits current user trust and peer AI model verification stats |
| `/chemicals` | Prints the full 6-chemical neurochemical vector |
| `/exit`, `/quit` | Commits the conversation to `Aira-Vault/Conversations/` and updates Daily Notes |

---

## Autonomous Dream Cycle & Continual Replay (Strategies 5 & 7)

Aira actively consolidates learning beyond static pretraining:

```python
from airapix.agent.cognitive_loop import AiraCognitiveAgent
from airapix.training.continual import ContinualReplayMixture, EWCConsolidator

# 1. Trigger an autonomous dream consolidation cycle
agent = AiraCognitiveAgent(vault_path=r"C:\Users\WaveWalker\Aira-Vault")
result = agent.run_dream_consolidation(num_dreams=5, merge_into_training=True)
print(f"Synthesized {result['dreams_synthesized']} verified reasoning experiences.")

# 2. Continual 80/20 Replay Dataset
mixture = ContinualReplayMixture(
    base_pretrain_paths=["dataset_builder/data/processed/train.jsonl"],
    dynamic_experience_paths=["dataset_builder/data/processed/aira_dream_consolidation.jsonl"],
    tokenizer=tokenizer,
    replay_ratio=0.80  # 80% foundational pretrain, 20% fresh experience
)
```

---

## Tokenizer Training (24k Vocab)

Train the customized Byte-Pair Encoding (BPE) tokenizer with native `<think>` special tokens:

```bash
python -m airapix.model.tokenizer.train_tokenizer \
  --input dataset_builder/data/processed/train.jsonl \
  --output airapix/model/tokenizer/tokenizer.json \
  --vocab-size 24000
```

---

## Training & Live Dashboard

### 1. Multi-Task Base Pretraining (Phase 2)

```bash
python -m airapix.training.train_v2 \
  --preset 125m \
  --max-steps 100000 \
  --batch-size 2 \
  --context-len 2048 \
  --accum 5 \
  --peak-lr 3.5e-4 \
  --save-every 200 \
  --warmup-steps 200
```

### 2. Live Web UI Dashboard

Monitor loss curves, perplexity, throughput (tok/s), and VRAM live at `http://localhost:7860`.

---

## Automated Verification Suite

Run the full cognitive and continual learning test suite:

```bash
python -c "import tests.test_agent_cognition as t; \
t.test_neuromodulator_dynamics(); \
t.test_theory_of_mind(); \
t.test_memory_vault_indexing(); \
t.test_aira_cognitive_agent_full_loop(); \
t.test_24k_tokenizer_and_config(); \
t.test_plastic_synapses_and_trust_architecture(); \
t.test_comprehensive_emotional_dimensions(); \
t.test_persistent_cross_chat_memory(); \
t.test_autonomous_dream_cycle(); \
t.test_continual_replay_and_ewc(); \
print('ALL 10 TEST SUITES PASSED 100% GREEN!')"
```

---

## License & Research Notes

Architectural research notes and benchmark specifications are documented under [docs/research_notes.md](docs/research_notes.md) and [docs/airapix_brain_architecture_and_breakthroughs.md](docs/airapix_brain_architecture_and_breakthroughs.md).
