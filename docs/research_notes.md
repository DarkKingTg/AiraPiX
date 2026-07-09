# Research Notes for Phase 1-3

Last checked: 2026-07-09.

## Decisions Applied

- Use a 1024-token context from the start.
- Replace normal attention layers with a small MLA-style attention layer.
- Keep the hybrid stack: about 25% MLA attention layers and 75% SSM layers.
- Use a Muon + AdamW hybrid optimizer: Muon for hidden 2D matrices, AdamW for
  embeddings, norms, biases, and output/output-like heads.
- Treat Qwen2.5-7B as an offline sequence distillation teacher rather than a
  live logit teacher, because the student has its own small English tokenizer.
- Use a RegMix-inspired proxy workflow for data mixture search.
- Use compression ratio as a cheap data-quality and data-mixture prior.

## Why MLA

DeepSeek-V2 introduced Multi-head Latent Attention (MLA) to compress the KV
cache into a latent vector. The paper reports major inference efficiency gains,
including a 93.3% KV-cache reduction compared with DeepSeek 67B. DeepSeek-V3
kept MLA as part of its efficient MoE architecture.

In this codebase, MLA is implemented as a practical small-model approximation:
keys and values are produced from a compressed latent vector, while a small
RoPE key path is kept separate. It is not a byte-for-byte DeepSeek
implementation, but it preserves the useful design pressure: train the model to
route attention through a compressed KV representation.

Sources:

- DeepSeek-V2 paper: https://arxiv.org/abs/2405.04434
- DeepSeek-V3 repository: https://github.com/deepseek-ai/DeepSeek-V3
- Hardware-centric MLA analysis: https://arxiv.org/abs/2506.02523

## Why Muon + AdamW

Muon applies momentum plus matrix orthogonalization to hidden-layer matrices.
Recent reports and open training work suggest it can be competitive with, or
more efficient than, AdamW for language-model training. The safer practical
recipe is hybrid: use Muon only where it naturally fits, and keep AdamW for
parameters where orthogonalized matrix updates are a poor match.

This repository follows that recipe:

- Muon: 2D hidden weights, excluding embeddings and output heads.
- AdamW: embeddings, norms, biases, scalar/vector parameters, output head.

Sources:

- Muon explainer and implementation notes: https://kellerjordan.github.io/posts/muon/
- Muon scalability report: https://arxiv.org/html/2502.16982v1
- Kimi K2 repository notes that Kimi K2 was trained with Muon:
  https://github.com/moonshotai/kimi-k2

## Why Qwen2.5-7B for Distillation

Qwen2.5-7B and Qwen2.5-7B-Instruct are open models with strong coding, math,
structured-output, and instruction-following behavior for their size. The
student tokenizer is intentionally small and English-only, so we use
sequence-level distillation: generate teacher answers into JSONL, then train the
student on those examples.

Sources:

- Qwen2.5-7B model card: https://huggingface.co/Qwen/Qwen2.5-7B
- Qwen2.5-7B-Instruct model card:
  https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen2.5 collection:
  https://huggingface.co/collections/Qwen/qwen25

## Why RegMix Proxy Search

RegMix frames data mixture selection as a regression problem: train many small
proxy models on different mixtures, fit a regression model from mixture weights
to validation scores, and use the best predicted mixture for the larger model.

For this hobby-scale project, the full 512-proxy recipe is too expensive, so the
implementation starts with a small random mixture search and a ridge-regression
predictor. It is deliberately extensible: increase proxy count and proxy steps
when using Colab or better hardware.

Sources:

- RegMix paper: https://arxiv.org/abs/2407.01492
- RegMix code: https://github.com/sail-sg/regmix

## Why Compression Ratio

Compression-based signals are cheap and useful for spotting repeated, templated,
or noisy documents. Recent work also connects gzip compressibility to
data-dependent scaling behavior. Here, gzip ratio is used as a filtering signal
and as a prior for RegMix category weights, not as a magic quality label.

Sources:

- gzip data-dependent scaling laws: https://arxiv.org/html/2405.16684v1
- Compression-based quality filtering (Compel): https://openreview.net/forum?id=KFafeqE5fe
- RefinedWeb filtering/deduplication context: https://arxiv.org/pdf/2306.01116

## Current Frontier/Reference Model Context

These are not training dependencies. They are context for feature direction and
future evaluations.

- OpenAI API docs currently recommend GPT-5.5 as the starting point for complex
  reasoning and coding, with smaller GPT-5.4 variants for cost/latency:
  https://developers.openai.com/api/docs/models
- DeepSeek-V3.2 is presented in official DeepSeek API docs as a reasoning-first
  agent model:
  https://api-docs.deepseek.com/news/news251201
- Z.ai GLM-5.2 is documented as a flagship long-horizon model with a 1M-token
  context:
  https://docs.z.ai/guides/llm/glm-5.2
- Kimi K2.7 Code is documented as a coding-focused agentic model with improved
  long-horizon coding and lower thinking-token usage than K2.6:
  https://www.kimi.com/resources/kimi-k2-7-code
