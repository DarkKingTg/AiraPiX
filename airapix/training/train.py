from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader

from airapix.model.config import config_from_preset
from airapix.model.model import AiraForCausalLM, count_parameters
from airapix.training.config import load_training_config, resolve_project_path
from airapix.training.dataset import JsonlLMDataset, MixtureJsonlDataset
from airapix.training.optimizer import build_optimizer, set_optimizer_lr
from airapix.training.regmix import apply_compression_prior, normalize_weights
from airapix.training.schedule import cosine_with_warmup
from airapix.training.tokenizer import TokenizerWrapper


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def cuda_memory_summary(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    idx = device.index or 0
    free, total = torch.cuda.mem_get_info(idx)
    return {
        "gpu_allocated_gb": round(torch.cuda.memory_allocated(idx) / (1024**3), 3),
        "gpu_reserved_gb": round(torch.cuda.memory_reserved(idx) / (1024**3), 3),
        "gpu_free_gb": round(free / (1024**3), 3),
        "gpu_total_gb": round(total / (1024**3), 3),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_precision(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return "fp32", None
    if precision == "bf16" or (precision == "auto" and torch.cuda.is_bf16_supported()):
        return "bf16", torch.bfloat16
    return "fp16", torch.float16


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: AiraForCausalLM, loader: DataLoader, device: torch.device, batches: int, autocast_ctx) -> float:
    model.eval()
    losses = []
    iterator = iter(loader)
    for _ in range(batches):
        try:
            batch = move_batch(next(iterator), device)
        except StopIteration:
            break
        with autocast_ctx():
            loss = model(batch["input_ids"], labels=batch["labels"])["loss"]
        losses.append(float(loss.detach().cpu()))
    model.train()
    return float(np.mean(losses)) if losses else math.inf


def save_checkpoint(path: Path, model: AiraForCausalLM, optimizer, step: int, best_val_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "model_config": model.config.to_dict(),
        },
        path,
    )


def maybe_plot_logs(log_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    steps, train_loss, val_loss = [], [], []
    with log_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["step"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]) if row["val_loss"] else np.nan)
    if not steps:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(steps, train_loss, label="train")
    if any(not np.isnan(v) for v in val_loss):
        plt.plot(steps, val_loss, label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(log_path.with_suffix(".png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--resume", type=str)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-every", type=int, default=None)
    args = parser.parse_args()

    config = load_training_config(args.config)
    train_cfg = config["training"]
    seed_everything(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision_name, dtype = select_precision(device, train_cfg.get("precision", "auto"))
    if dtype is None:
        autocast_ctx = nullcontext
    else:
        autocast_ctx = lambda: torch.autocast(device_type=device.type, dtype=dtype)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device.type, enabled=(precision_name == "fp16"))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(precision_name == "fp16"))

    tokenizer_path = resolve_project_path(config, config["data"]["tokenizer_path"])
    tokenizer = TokenizerWrapper(tokenizer_path)
    model_cfg = config_from_preset(
        config["model"]["preset"],
        vocab_size=tokenizer.vocab_size,
        context_len=int(config["model"]["context_len"]),
        dropout=float(config["model"].get("dropout", 0.0)),
        gradient_checkpointing=bool(config["model"].get("gradient_checkpointing", True)),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = AiraForCausalLM(model_cfg).to(device)
    optimizer = build_optimizer(model, config["optimizer"])

    weights = normalize_weights(config["data"]["mixture_weights"])
    if config["data"].get("use_compression_prior", True):
        weights = apply_compression_prior(
            weights,
            resolve_project_path(config, config["data"]["manifest_path"]),
        )
    category_paths = {
        cat: resolve_project_path(config, path)
        for cat, path in config["data"]["category_shards"].items()
    }
    train_dataset = MixtureJsonlDataset(
        category_paths=category_paths,
        weights=weights,
        tokenizer=tokenizer,
        context_len=model_cfg.context_len,
        seed=int(train_cfg.get("seed", 42)),
    )
    val_dataset = JsonlLMDataset(
        resolve_project_path(config, config["data"]["val_path"]),
        tokenizer=tokenizer,
        context_len=model_cfg.context_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["micro_batch_size"]),
        num_workers=int(config["data"].get("num_workers", 0)),
    )
    val_loader = DataLoader(val_dataset, batch_size=int(train_cfg["micro_batch_size"]), num_workers=0)
    train_iter = iter(train_loader)

    out_dir = resolve_project_path(config, train_cfg["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    start_step = 0
    best_val_loss = math.inf
    if args.resume:
        state = torch.load(args.resume, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"]) + 1
        best_val_loss = float(state.get("best_val_loss", math.inf))

    max_steps = args.max_steps or int(train_cfg["max_steps"])
    micro_batch = int(train_cfg["micro_batch_size"])
    target_effective = int(train_cfg["target_effective_batch_size"])
    grad_accum = max(1, math.ceil(target_effective / micro_batch))
    warmup_steps = int(max_steps * float(config["optimizer"].get("warmup_fraction", 0.02)))
    effective_batch = micro_batch * grad_accum
    log_every = args.log_every or int(train_cfg.get("log_interval", 10))
    run_info = {
        "config": str(Path(args.config).resolve()),
        "resume": args.resume,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": precision_name,
        "params": count_parameters(model),
        "model_preset": config["model"]["preset"],
        "context_len": model_cfg.context_len,
        "tokenizer_vocab": tokenizer.vocab_size,
        "max_steps": max_steps,
        "start_step": start_step,
        "micro_batch_size": micro_batch,
        "target_effective_batch_size": target_effective,
        "actual_effective_batch_size": effective_batch,
        "grad_accum_steps": grad_accum,
        "eval_interval": int(train_cfg["eval_interval"]),
        "checkpoint_interval": int(train_cfg["checkpoint_interval"]),
        "log_interval": log_every,
        "train_category_files": {k: str(v) for k, v in category_paths.items()},
        "val_examples": len(val_dataset),
        "mixture_weights": weights,
        **cuda_memory_summary(device),
    }
    print("[train] run configuration", flush=True)
    print(json.dumps(run_info, indent=2), flush=True)

    log_mode = "a" if start_step > 0 and log_path.exists() else "w"
    write_header = log_mode == "w"
    with log_path.open(log_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "lr",
                "train_loss",
                "val_loss",
                "elapsed_sec",
                "step_sec",
                "tokens_per_sec",
                "gpu_allocated_gb",
                "gpu_reserved_gb",
                "gpu_free_gb",
            ],
        )
        if write_header:
            writer.writeheader()

        patience_bad = 0
        train_started = time.time()
        step = start_step - 1
        for step in range(start_step, max_steps):
            step_started = time.time()
            lr = cosine_with_warmup(
                step,
                max_steps,
                warmup_steps,
                float(config["optimizer"]["peak_lr"]),
                float(config["optimizer"].get("min_lr_ratio", 0.1)),
            )
            set_optimizer_lr(optimizer, lr, config["optimizer"])
            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            for _ in range(grad_accum):
                batch = move_batch(next(train_iter), device)
                with autocast_ctx():
                    loss = model(batch["input_ids"], labels=batch["labels"])["loss"] / grad_accum
                scaler.scale(loss).backward()
                running_loss += float(loss.detach().cpu()) * grad_accum
            if precision_name == "fp16":
                if optimizer.muon is not None:
                    scaler.unscale_(optimizer.muon)
                if optimizer.adamw is not None:
                    scaler.unscale_(optimizer.adamw)
                if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                if optimizer.muon is not None:
                    scaler.step(optimizer.muon)
                if optimizer.adamw is not None:
                    scaler.step(optimizer.adamw)
                scaler.update()
            else:
                if float(train_cfg.get("grad_clip_norm", 0.0)) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip_norm"]))
                optimizer.step()

            val_loss = ""
            checkpoint_event = ""
            if (step + 1) % int(train_cfg["eval_interval"]) == 0:
                print(f"[eval] step={step + 1} running {train_cfg['eval_batches']} batches...", flush=True)
                val = evaluate(
                    model,
                    val_loader,
                    device,
                    int(train_cfg["eval_batches"]),
                    autocast_ctx,
                )
                val_loss = f"{val:.6f}"
                if val < best_val_loss:
                    best_val_loss = val
                    patience_bad = 0
                    save_checkpoint(ckpt_dir / "best.pt", model, optimizer, step, best_val_loss)
                    checkpoint_event = "best"
                    print(f"[eval] step={step + 1} val_loss={val_loss} new_best=true", flush=True)
                else:
                    patience_bad += 1
                    print(
                        f"[eval] step={step + 1} val_loss={val_loss} "
                        f"best={best_val_loss:.6f} patience={patience_bad}/{train_cfg['early_stopping_patience']}",
                        flush=True,
                    )
                if patience_bad >= int(train_cfg["early_stopping_patience"]):
                    print("[train] early stopping", flush=True)
                    break

            if (step + 1) % int(train_cfg["checkpoint_interval"]) == 0:
                save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, step, best_val_loss)
                checkpoint_event = "latest" if not checkpoint_event else f"{checkpoint_event}+latest"
                print(f"[checkpoint] step={step + 1} wrote latest.pt", flush=True)

            elapsed = time.time() - train_started
            step_sec = time.time() - step_started
            tokens_this_step = effective_batch * model_cfg.context_len
            tokens_per_sec = tokens_this_step / max(1e-6, step_sec)
            mem = cuda_memory_summary(device)
            writer.writerow(
                {
                    "step": step,
                    "lr": f"{lr:.8f}",
                    "train_loss": f"{running_loss:.6f}",
                    "val_loss": val_loss,
                    "elapsed_sec": f"{elapsed:.2f}",
                    "step_sec": f"{step_sec:.3f}",
                    "tokens_per_sec": f"{tokens_per_sec:.2f}",
                    "gpu_allocated_gb": mem.get("gpu_allocated_gb", ""),
                    "gpu_reserved_gb": mem.get("gpu_reserved_gb", ""),
                    "gpu_free_gb": mem.get("gpu_free_gb", ""),
                }
            )
            f.flush()
            should_log = step == start_step or (step + 1) % max(1, log_every) == 0 or bool(val_loss) or bool(checkpoint_event)
            if should_log:
                steps_done = step + 1 - start_step
                steps_left = max(0, max_steps - step - 1)
                avg_step_sec = elapsed / max(1, steps_done)
                eta = avg_step_sec * steps_left
                mem_text = ""
                if mem:
                    mem_text = (
                        f" gpu_alloc={mem['gpu_allocated_gb']:.2f}GB"
                        f" gpu_reserved={mem['gpu_reserved_gb']:.2f}GB"
                        f" gpu_free={mem['gpu_free_gb']:.2f}GB"
                    )
                print(
                    f"[train] step={step + 1}/{max_steps}"
                    f" lr={lr:.2e}"
                    f" loss={running_loss:.4f}"
                    f" val={val_loss or '-'}"
                    f" tok/s={tokens_per_sec:.0f}"
                    f" step_time={step_sec:.2f}s"
                    f" elapsed={format_duration(elapsed)}"
                    f" eta={format_duration(eta)}"
                    f"{mem_text}",
                    flush=True,
                )

    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, step, best_val_loss)
    maybe_plot_logs(log_path)
    print(f"[train] final checkpoint: {ckpt_dir / 'final.pt'}", flush=True)
    print(f"[train] logs: {log_path}", flush=True)
    print(f"[train] checkpoints: {ckpt_dir}", flush=True)


if __name__ == "__main__":
    main()
