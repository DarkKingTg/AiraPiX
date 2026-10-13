"""
AiraPix Generation Diagnostics & Hyperparameter Sweep
Tests the latest checkpoint with multiple generation strategies and reports results.
"""
from __future__ import annotations

import sys
import torch
from pathlib import Path

# Force UTF-8 on Windows stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airapix.inference.generate import load_model, generate_response
from airapix.training.tokenizer import TokenizerWrapper


def banner(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def find_latest_checkpoint() -> Path:
    ckpt_dir = Path("runs/checkpoints")
    import re
    pts = list(ckpt_dir.glob("*.pt"))
    pts.sort(
        key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1))
        if re.search(r"step_(\d+)", p.name) else 0,
        reverse=True,
    )
    return pts[0] if pts else None


def is_readable(text: str) -> bool:
    """Heuristic: mostly ASCII printable chars = readable English output."""
    if not text or not text.strip():
        return False
    readable = sum(1 for c in text if 32 <= ord(c) < 127 or c in '\n\r\t')
    return (readable / max(len(text), 1)) > 0.80


def test_tokenizer(tok: TokenizerWrapper):
    banner("1. TOKENIZER SANITY CHECK")
    probes = ["hi", "hello", "User:\nhi :)\n\nAssistant:\n", "The capital of France is", "def hello():"]
    for p in probes:
        ids = tok.encode(p)
        rt = tok.decode(ids)
        ok = "OK" if len(ids) > 0 else "FAIL"
        print(f"  [{ok}] '{p[:40]}' -> {len(ids)} tokens -> decoded='{rt[:40]}'")
    print(f"\n  Vocab size : {tok.vocab_size}")
    print(f"  EOS id     : {tok.eos_token_id}")
    print(f"  PAD id     : {tok.pad_token_id}")


def sweep_generation(model, tok, device, prompt: str):
    banner(f"2. GENERATION SWEEP --- prompt: '{prompt}'")

    configs = [
        # (label,               temp, top_k, max_tokens)
        ("Greedy temp=0.1",     0.1,  1,     80),
        ("Low-temp temp=0.3",   0.3,  20,    80),
        ("Default temp=0.7",    0.7,  50,    80),
        ("High-temp temp=1.0",  1.0,  50,    80),
        ("Tight top-k=5",       0.5,  5,     80),
        ("Wide top-k=100",      0.7,  100,   80),
        ("Long greedy",         0.1,  1,     200),
        ("Long default",        0.7,  50,    200),
    ]

    results = []
    for label, temp, top_k, max_tok in configs:
        try:
            out = generate_response(
                model, tok, prompt,
                max_new_tokens=max_tok,
                temperature=temp,
                top_k=top_k,
                device=device,
            )
            flag = "[GOOD]" if is_readable(out) else "[TRASH]"
            results.append((label, temp, top_k, max_tok, out, flag))
            print(f"\n  {flag} [{label}] temp={temp}, top_k={top_k}, max={max_tok}")
            print(f"     Output: {repr(out[:150])}")
        except Exception as e:
            print(f"\n  [CRASH] [{label}]: {e}")
            results.append((label, temp, top_k, max_tok, "", "[CRASH]"))

    return results


def check_logit_stats(model, tok, device, prompt: str):
    banner("3. LOGIT HEALTH CHECK")
    formatted = f"User:\n{prompt}\n\nAssistant:\n"
    ids = tok.encode(formatted)
    inp = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(inp)
        logits = out["logits"] if isinstance(out, dict) else out[0]

    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=10)

    print(f"  Logit range: min={last_logits.min():.3f}, max={last_logits.max():.3f}, std={last_logits.std():.3f}")
    print(f"  Top-10 next token predictions after 'hi :)':")
    for i, (prob, tid) in enumerate(zip(top_probs.tolist(), top_ids.tolist())):
        token_str = tok.decode([tid])
        print(f"    {i+1:2d}. id={tid:5d}  prob={prob:.4f}  token={repr(token_str)}")

    entropy = -(probs * (probs + 1e-10).log()).sum().item()
    print(f"\n  Distribution entropy: {entropy:.3f} nats")
    if entropy > 8.0:
        print("  >> VERY HIGH ENTROPY: model is near-random, output will be garbage. Still too early in training.")
    elif entropy > 6.0:
        print("  >> HIGH ENTROPY: model under-trained, use very low temperature (0.1-0.3) to force coherence.")
    elif entropy > 4.0:
        print("  >> MODERATE ENTROPY: use temperature <= 0.7 for best results.")
    else:
        print("  >> LOW ENTROPY: model fairly confident, safe to use temperature 0.5-1.0")


def recommend_settings(results):
    banner("4. BEST HYPERPARAMETERS SUMMARY")
    good = [(label, temp, top_k, max_tok, out) for label, temp, top_k, max_tok, out, flag in results if flag == "[GOOD]"]
    trash = [label for label, *_, flag in results if flag == "[TRASH]"]

    if good:
        print(f"  GOOD configs ({len(good)}/{len(results)}):")
        for label, temp, top_k, max_tok, out in good[:5]:
            print(f"    - [{label}] temp={temp}, top_k={top_k} => {repr(out[:80])}")
        # Pick lowest temperature (most coherent) that passes
        best = sorted(good, key=lambda x: x[1])[0]
        print(f"\n  RECOMMENDED: temp={best[1]}, top_k={best[2]}, max_tokens=200")
        print(f"\n  Suggested CLI command:")
        print(f"    python -m airapix.inference.generate --checkpoint runs/checkpoints/[latest].pt --temp {best[1]} --top-k {best[2]} --max-tokens 200")
    else:
        print("  No readable outputs found at any config!")
        print("  Root cause: Model perplexity is still HIGH (~172+). Needs more training steps.")
        print("  The loss needs to drop below ~3.0 for coherent English outputs.")
        print("  Use greedy decoding (temp=0.1, top_k=1) as the only semi-usable setting.")
        print("")
        print("  RECOMMENDED ACTION: Wait until step 100k completes, then retrain tokenizer.")

    if trash:
        print(f"\n  Configs producing garbage: {trash}")


def main():
    banner("AIRA GENERATION DIAGNOSTICS v2.0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tok = TokenizerWrapper("airapix/model/tokenizer/tokenizer.json")
    test_tokenizer(tok)

    ckpt = find_latest_checkpoint()
    if not ckpt:
        print("\nERROR: No checkpoint found in runs/checkpoints/")
        sys.exit(1)
    print(f"\n  Loading: {ckpt.name}")

    model = load_model(str(ckpt), preset="125m", device=device)
    model.eval()
    print("  Model loaded OK")

    test_prompts = ["hi :)", "What is 2 + 2?", "Tell me about the sun."]
    all_results = []
    for prompt in test_prompts:
        results = sweep_generation(model, tok, device, prompt)
        all_results.extend(results)
        check_logit_stats(model, tok, device, prompt)

    recommend_settings(all_results)
    banner("DIAGNOSTICS COMPLETE")


if __name__ == "__main__":
    main()
