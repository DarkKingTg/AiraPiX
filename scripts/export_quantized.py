from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch

from airapix.model.config import AiraConfig, config_from_preset
from airapix.model.model import AiraForCausalLM, count_parameters


def export_quantized(checkpoint_path: Path, output_path: Path, bits: int = 4) -> None:
    print(f"[export] Loading checkpoint: {checkpoint_path}", flush=True)
    state = torch.load(checkpoint_path, map_location="cpu")
    model_config_dict = state.get("model_config", {})
    if model_config_dict:
        config = AiraConfig.from_dict(model_config_dict)
    else:
        config = config_from_preset("125m")

    model = AiraForCausalLM(config)
    model.load_state_dict(state["model"], strict=False)
    total_params = count_parameters(model)
    print(f"[export] Loaded model with {total_params / 1e6:.1f}M parameters.", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import bitsandbytes as bnb
        print(f"[export] Quantizing linear layers to {bits}-bit NF4 using bitsandbytes...", flush=True)
        for name, child in list(model.named_children()):
            if isinstance(child, torch.nn.Linear):
                qlayer = bnb.nn.Linear4bit(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    compute_dtype=torch.float16,
                    quant_type="nf4",
                )
                qlayer.weight = child.weight
                if child.bias is not None:
                    qlayer.bias = child.bias
                setattr(model, name, qlayer)
    except ImportError:
        print("[export] bitsandbytes not available. Saving standard FP16 checkpoint for low-memory load.", flush=True)
        model = model.to(dtype=torch.float16)

    # Save state dict
    save_dict = {
        "model": model.state_dict(),
        "model_config": config.to_dict(),
        "quantization": f"{bits}bit-nf4",
    }
    torch.save(save_dict, output_path)
    file_size_mb = output_path.stat().st_size / (1024**2)
    print(f"[export] Exported quantized model to: {output_path} ({file_size_mb:.2f} MB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Aira model checkpoints into low-memory 4-bit format.")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to input .pt checkpoint")
    parser.add_argument("--output", required=True, type=str, help="Path to output .pt or .safetensors file")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bit depth")
    args = parser.parse_args()

    export_quantized(Path(args.checkpoint), Path(args.output), bits=args.bits)


if __name__ == "__main__":
    main()
