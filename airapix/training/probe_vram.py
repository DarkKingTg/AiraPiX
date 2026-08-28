from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from airapix.model.config import config_from_preset
from airapix.model.model import AiraForCausalLM, count_parameters
from airapix.training.config import load_training_config, resolve_project_path
from airapix.training.optimizer import build_optimizer


def autocast_dtype(device: torch.device, precision: str):
    if device.type != "cuda":
        return None
    if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    if precision in {"fp16", "auto"}:
        return torch.float16
    return None


def try_batch(model_cfg, train_cfg, opt_cfg, micro_batch: int, device: torch.device) -> dict:
    model = AiraForCausalLM(model_cfg).to(device)
    model.train()
    optimizer = build_optimizer(model, opt_cfg)
    dtype = autocast_dtype(device, train_cfg.get("precision", "auto"))
    ok = True
    error = None
    peak_gb = 0.0
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        for _ in range(int(train_cfg.get("probe_steps", 3))):
            input_ids = torch.randint(
                0,
                model_cfg.vocab_size,
                (micro_batch, model_cfg.context_len),
                device=device,
            )
            labels = input_ids.clone()
            optimizer.zero_grad(set_to_none=True)
            if dtype is None:
                loss = model(input_ids, labels=labels)["loss"]
            else:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    loss = model(input_ids, labels=labels)["loss"]
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    except RuntimeError as exc:
        ok = False
        error = str(exc).splitlines()[0]
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            torch.cuda.empty_cache()
    finally:
        del model
        del optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"micro_batch_size": micro_batch, "ok": ok, "peak_gb": round(peak_gb, 3), "error": error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()

    config = load_training_config(args.config)
    probe_cfg = config.get("probe")
    if not probe_cfg:
        print("[probe] No 'probe' section in config. Skipping VRAM probe.", flush=True)
        return

    train_cfg = {**config["training"], "probe_steps": probe_cfg.get("steps", 3)}
    opt_cfg = config["optimizer"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {"device": str(device), "presets": []}
    if device.type != "cuda":
        results["note"] = "CUDA is not available. VRAM probing requires an NVIDIA GPU."

    for preset in probe_cfg["presets"]:
        model_cfg = config_from_preset(
            preset,
            vocab_size=int(config["model"]["vocab_size"]),
            context_len=int(config["model"]["context_len"]),
            dropout=float(config["model"].get("dropout", 0.0)),
            gradient_checkpointing=bool(config["model"].get("gradient_checkpointing", True)),
        )
        preset_result = {
            "preset": preset,
            "params": count_parameters(AiraForCausalLM(model_cfg)),
            "batches": [],
            "recommended_micro_batch_size": None,
        }
        for micro_batch in probe_cfg["micro_batch_sizes"]:
            print(f"[probe] preset={preset} micro_batch={micro_batch}")
            batch_result = try_batch(model_cfg, train_cfg, opt_cfg, int(micro_batch), device)
            preset_result["batches"].append(batch_result)
            if (
                batch_result["ok"]
                and device.type == "cuda"
                and batch_result["peak_gb"] <= float(probe_cfg["memory_limit_gb"])
            ):
                preset_result["recommended_micro_batch_size"] = int(micro_batch)
        results["presets"].append(preset_result)
        if preset_result["recommended_micro_batch_size"] is not None:
            break

    out_dir = resolve_project_path(config, config["training"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "vram_probe_result.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"[probe] wrote {out_path}")


if __name__ == "__main__":
    main()
