from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from airapix.model.config import config_from_preset
from airapix.model.model import AiraForCausalLM
from airapix.training.tokenizer import TokenizerWrapper


def load_model(checkpoint_path: str | Path | None, preset: str = "125m", device: torch.device = torch.device("cpu")) -> AiraForCausalLM:
    cfg = config_from_preset(preset, vocab_size=12000, context_len=1024)
    model = AiraForCausalLM(cfg)
    
    if checkpoint_path and Path(checkpoint_path).is_file():
        print(f"[inference] Loading checkpoint from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        elif "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
    else:
        print(f"[inference] No checkpoint found at {checkpoint_path}. Using initialized model architecture ({preset}).")
    
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
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    decoded = tokenizer.decode(output_ids[0].tolist())
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[inference] Running on device: {device}")

    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.is_file():
        print(f"[error] Tokenizer file not found at {tokenizer_path}")
        sys.exit(1)
        
    tokenizer = TokenizerWrapper(str(tokenizer_path))
    model = load_model(args.checkpoint, preset=args.preset, device=device)

    if args.prompt:
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
        print("\n=== AiraPix Interactive Companion Chat ===")
        print("Type your prompt below (or 'exit' / 'quit' to end session).\n")
        while True:
            try:
                user_input = input("AiraPix> ").strip()
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    print("Exiting AiraPix CLI. Goodbye!")
                    break
                
                print("\n[Generating response...]")
                response = generate_response(
                    model,
                    tokenizer,
                    user_input,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temp,
                    top_k=args.top_k,
                    device=device,
                )
                print(f"\nResponse:\n{response}\n")
            except (KeyboardInterrupt, EOFError):
                print("\nSession ended.")
                break


if __name__ == "__main__":
    main()
