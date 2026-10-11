from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airapix.model.config import config_from_preset
from airapix.model.model import AiraForCausalLM
from airapix.training.tokenizer import TokenizerWrapper


def load_model(
    checkpoint_path: str | Path | None,
    preset: str = "125m",
    device: torch.device = torch.device("cpu"),
    vocab_size: int = 24000,
    context_len: int = 2048,
) -> AiraForCausalLM:
    ckpt_file = None
    if checkpoint_path and Path(checkpoint_path).is_file():
        ckpt_file = Path(checkpoint_path)
    elif checkpoint_path and str(checkpoint_path).lower() in {"latest", "auto"}:
        ckpt_dir = Path("runs/checkpoints")
        if ckpt_dir.exists():
            pts = list(ckpt_dir.glob("*.pt"))
            if pts:
                import re
                pts.sort(key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)) if re.search(r"step_(\d+)", p.name) else 0, reverse=True)
                ckpt_file = pts[0]
    elif Path("runs/checkpoints").exists():
        pts = list(Path("runs/checkpoints").glob("*.pt"))
        if pts:
            import re
            pts.sort(key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)) if re.search(r"step_(\d+)", p.name) else 0, reverse=True)
            ckpt_file = pts[0]

    state = None
    if ckpt_file and ckpt_file.is_file():
        print(f"[inference] Loading checkpoint from {ckpt_file}")
        try:
            state = torch.load(ckpt_file, map_location=device, weights_only=False)
        except Exception:
            state = torch.load(ckpt_file, map_location=device, weights_only=True)

        # Auto-detect actual vocab_size from checkpoint embedding weight
        sd = state.get("model_state_dict") or state.get("model_state") or state.get("model") or state
        if isinstance(sd, dict) and "embed_tokens.weight" in sd:
            actual_vocab = sd["embed_tokens.weight"].shape[0]
            if actual_vocab != vocab_size:
                print(f"[inference] Auto-detected checkpoint vocabulary size: {actual_vocab}")
                vocab_size = actual_vocab
    else:
        print(f"[inference] No checkpoint found. Using initialized model architecture ({preset}).")

    cfg = config_from_preset(preset, vocab_size=vocab_size, context_len=context_len)
    model = AiraForCausalLM(cfg)

    if state is not None:
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"], strict=False)
        elif "model_state" in state:
            model.load_state_dict(state["model_state"], strict=False)
        elif "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model


def generate_response(
    model: AiraForCausalLM,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> str:
    # Format raw prompt into Aira's training chat template if needed
    if not prompt.startswith("User:") and "Assistant:" not in prompt:
        formatted_prompt = f"User:\n{prompt.strip()}\n\nAssistant:\n"
    else:
        formatted_prompt = prompt

    encoded = tokenizer.encode(formatted_prompt)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    input_len = input_ids.shape[1]
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Slice off prompt tokens to return only the AI's generated response
    new_tokens = output_ids[0, input_len:].tolist()
    decoded = tokenizer.decode(new_tokens).strip()
    return decoded


def main() -> None:
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(description="AiraPix Interactive Inference Generation Engine")
    parser.add_argument("--checkpoint", default="runs/phase3_125m/latest_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--tokenizer", default="airapix/model/tokenizer/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--preset", default="125m", help="Model preset (125m, 90m, 60m, tiny)")
    parser.add_argument("--prompt", type=str, default="", help="Single prompt text to generate")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling threshold")
    parser.add_argument("--num-samples", "-n", "--samples", type=int, default=1, help="Number of independent completions to generate for the prompt (default: 1)")
    parser.add_argument("--agent", action="store_true", help="Launch interactive Cognitive Agent interface with neurochemicals and Theory of Mind")
    args = parser.parse_args()

    if args.agent:
        from airapix.agent.chat import run_cognitive_chat
        run_cognitive_chat(
            checkpoint_path=args.checkpoint if Path(args.checkpoint).is_file() else None,
            tokenizer_path=args.tokenizer,
            preset=args.preset,
            max_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
        )
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Running on device: {device}")

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        print(f"[error] Tokenizer file not found at {tokenizer_path}")
        sys.exit(1)
        
    tokenizer = TokenizerWrapper(str(tokenizer_path))
    model = load_model(args.checkpoint, preset=args.preset, device=device)

    num_samples = max(1, args.num_samples)

    if args.prompt:
        if num_samples == 1:
            output = generate_response(
                model,
                tokenizer,
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temp,
                top_k=args.top_k,
                device=device,
            )
            print(f"\n--- Output ---\n{output}\n")
        else:
            print(f"\n[*] Generating {num_samples} candidate responses for prompt:\n    \"{args.prompt}\"\n")
            for idx in range(1, num_samples + 1):
                output = generate_response(
                    model,
                    tokenizer,
                    args.prompt,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temp,
                    top_k=args.top_k,
                    device=device,
                )
                print(f"[->] Response {idx}/{num_samples}:\n{output}\n")
    else:
        print("\n=== AiraPix Interactive Companion Chat ===")
        print("Type your prompt below (or 'exit' / 'quit' to end session).\n")
        while True:
            try:
                user_input = input("AiraPix> ").strip()
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    print("Exiting AiraPix CLI. Goodbye!")
                    break
                
                print("\n[Generating response...]")
                for idx in range(1, num_samples + 1):
                    response = generate_response(
                        model,
                        tokenizer,
                        user_input,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temp,
                        top_k=args.top_k,
                        device=device,
                    )
                    header = f"[->] Response {idx}/{num_samples}:\n" if num_samples > 1 else "\nResponse:\n"
                    print(f"{header}{response}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break


if __name__ == "__main__":
    main()
