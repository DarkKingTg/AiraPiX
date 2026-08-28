from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    printable = " ".join(str(part) for part in cmd)
    print(f"\n[colab] $ {printable}", flush=True)
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    elapsed = time.time() - started
    print(f"[colab] exit={result.returncode} elapsed={elapsed / 60:.1f} min", flush=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def in_colab() -> bool:
    try:
        import google.colab  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def maybe_mount_drive(enabled: bool, drive_project_dir: str | None) -> Path | None:
    drive_mounted = Path("/content/drive/MyDrive").exists()
    if not drive_mounted and enabled and in_colab():
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
            drive_mounted = Path("/content/drive/MyDrive").exists()
        except Exception as err:
            print(
                f"[colab] Notice: Could not trigger drive.mount inside script process ({err}). "
                "If you want to save checkpoints to Google Drive, run `from google.colab import drive; drive.mount('/content/drive')` in a Colab cell first.",
                flush=True,
            )

    if drive_mounted and drive_project_dir:
        target = Path(drive_project_dir)
        target.mkdir(parents=True, exist_ok=True)
        print(f"[colab] Drive project directory: {target}", flush=True)
        return target
    return None


def install_dependencies(skip: bool) -> None:
    if skip:
        print("[colab] skipping dependency install", flush=True)
        return
    packages = [
        "pyyaml",
        "datasets",
        "tokenizers",
        "ftfy",
        "langdetect",
        "datasketch",
        "tqdm",
        "matplotlib",
        "numpy",
        "huggingface_hub",
        "transformers",
        "accelerate",
        "safetensors",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
    ]
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", *packages, "--progress-bar", "off"])


def show_runtime() -> None:
    print("[colab] runtime", flush=True)
    print(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "cwd": str(ROOT),
                "in_colab": in_colab(),
            },
            indent=2,
        ),
        flush=True,
    )
    run(["nvidia-smi"], check=False)
    run(
        [
            sys.executable,
            "-c",
            "import torch; "
            "print('torch', torch.__version__); "
            "print('cuda', torch.cuda.is_available()); "
            "print('cuda_runtime', torch.version.cuda); "
            "print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')",
        ],
        check=False,
    )


def write_colab_config(
    base_config: Path,
    output_config: Path,
    processed_dir: Path,
    output_dir: str,
    max_steps: int | None,
    micro_batch_size: int | None,
    effective_batch_size: int | None,
    log_every: int,
    eval_interval: int | None,
    checkpoint_interval: int | None,
) -> Path:
    import yaml

    with base_config.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    def rel(path: Path) -> str:
        return str(path.relative_to(ROOT).as_posix()) if path.is_absolute() else str(path).replace("\\", "/")

    config["data"]["train_path"] = rel(processed_dir / "train.jsonl")
    config["data"]["val_path"] = rel(processed_dir / "val.jsonl")
    config["data"]["category_shards"] = {
        "text": rel(processed_dir / "train_text.jsonl"),
        "chat": rel(processed_dir / "train_chat.jsonl"),
        "reasoning": rel(processed_dir / "train_reasoning.jsonl"),
        "code": rel(processed_dir / "train_code.jsonl"),
        "tool_use": rel(processed_dir / "train_tool_use.jsonl"),
    }
    config["training"]["output_dir"] = output_dir
    config["training"]["log_interval"] = log_every
    if max_steps is not None:
        config["training"]["max_steps"] = max_steps
    if micro_batch_size is not None:
        config["training"]["micro_batch_size"] = micro_batch_size
    if effective_batch_size is not None:
        config["training"]["target_effective_batch_size"] = effective_batch_size
    if eval_interval is not None:
        config["training"]["eval_interval"] = eval_interval
    if checkpoint_interval is not None:
        config["training"]["checkpoint_interval"] = checkpoint_interval

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"[colab] wrote config: {output_config}", flush=True)
    return output_config


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def print_dataset_summary(processed_dir: Path) -> None:
    summary = {}
    for name in [
        "train.jsonl",
        "val.jsonl",
        "train_text.jsonl",
        "train_chat.jsonl",
        "train_reasoning.jsonl",
        "train_code.jsonl",
        "train_tool_use.jsonl",
    ]:
        path = processed_dir / name
        summary[name] = {"rows": count_jsonl(path), "mb": round(path.stat().st_size / (1024**2), 2) if path.exists() else 0}
    print("[colab] dataset summary", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


def maybe_copy_to_drive(drive_dir: Path | None, paths: list[Path]) -> None:
    if drive_dir is None:
        return
    for path in paths:
        if not path.exists():
            continue
        dest = drive_dir / path.relative_to(ROOT)
        if path.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(path, dest)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        print(f"[colab] copied to Drive: {dest}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Colab runner for AiraPix data, Qwen teacher data, and training.")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--mount-drive", action="store_true")
    parser.add_argument("--drive-project-dir", default="/content/drive/MyDrive/AiraPix")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--dataset-config", default="dataset_builder/config.trainable_v0.yaml")
    parser.add_argument("--base-training-config", default="training/config.yaml")
    parser.add_argument("--colab-config", default="training/config.colab.yaml")
    parser.add_argument("--processed-dir", default="dataset_builder/data/processed")
    parser.add_argument("--teacher-dir", default="dataset_builder/data/processed_teacher")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=12000)
    parser.add_argument("--download-teachers", action="store_true", help="Download teacher sub-models before distillation/training.")
    parser.add_argument("--teacher-models-manifest", default="training/teacher_models.yaml")
    parser.add_argument("--teacher-download-mode", choices=["metadata", "tokenizer", "full"], default="full")
    parser.add_argument("--teacher-model-names", nargs="+", default=["all"])
    parser.add_argument("--teacher-cache-dir", default="models/teachers/hf_cache")
    parser.add_argument("--teacher-max-workers", type=int, default=8)
    parser.add_argument("--distill", action="store_true", help="Generate Qwen2.5-7B teacher examples before training.")
    parser.add_argument("--distill-inputs", nargs="+", default=["train_chat.jsonl", "train_reasoning.jsonl", "train_tool_use.jsonl"])
    parser.add_argument("--distill-limit", type=int, default=None, help="Per-input limit. Omit for all rows.")
    parser.add_argument("--distill-log-every", type=int, default=10)
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--skip-tokenizer", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--preset", type=str, default=None, help="Model preset (e.g. 1.5b, 3b, 7b, 8b)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit weight quantization")
    parser.add_argument("--use-qlora", action="store_true", help="Apply QLoRA adapters for efficient 4-bit fine-tuning")
    parser.add_argument("--tpu", action="store_true", help="Enable TPU v5e-1 PyTorch-XLA execution mode")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=None)
    parser.add_argument("--effective-batch-size", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="runs/colab_phase3")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    os.chdir(ROOT)
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if args.hf_token:
        env["HF_TOKEN"] = args.hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    drive_dir = maybe_mount_drive(args.mount_drive, args.drive_project_dir)
    install_dependencies(args.skip_install)
    show_runtime()

    processed_dir = ROOT / args.processed_dir
    active_processed_dir = processed_dir

    if not args.skip_dataset:
        run([sys.executable, "dataset_builder/run_all.py", "--config", args.dataset_config], env=env)
    print_dataset_summary(processed_dir)

    if not args.skip_tokenizer:
        run(
            [
                sys.executable,
                "-m",
                "airapix.model.tokenizer.train_tokenizer",
                "--input",
                str(processed_dir / "train.jsonl"),
                "--output",
                "airapix/model/tokenizer/tokenizer.json",
                "--vocab-size",
                str(args.tokenizer_vocab_size),
            ],
            env=env,
        )

    if args.download_teachers:
        run(
            [
                sys.executable,
                "-m",
                "airapix.training.download_teacher_models",
                "--manifest",
                args.teacher_models_manifest,
                "--mode",
                args.teacher_download_mode,
                "--cache-dir",
                args.teacher_cache_dir,
                "--max-workers",
                str(args.teacher_max_workers),
                "--names",
                *args.teacher_model_names,
            ],
            env=env,
        )

    if args.distill:
        teacher_parts: list[Path] = []
        teacher_raw_dir = ROOT / "dataset_builder/data/teacher"
        teacher_raw_dir.mkdir(parents=True, exist_ok=True)
        for input_name in args.distill_inputs:
            input_path = processed_dir / input_name
            output_path = teacher_raw_dir / f"qwen25_7b_{Path(input_name).stem}.jsonl"
            cmd = [
                sys.executable,
                "-m",
                "airapix.training.distill_from_qwen",
                "--config",
                args.base_training_config,
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--log-every",
                str(args.distill_log_every),
                "--cache-dir",
                args.teacher_cache_dir,
            ]
            if args.distill_limit is not None:
                cmd.extend(["--limit", str(args.distill_limit)])
            run(cmd, env=env)
            teacher_parts.append(output_path)

        combined_teacher = teacher_raw_dir / "distilled_qwen25_7b_all.jsonl"
        with combined_teacher.open("w", encoding="utf-8", newline="\n") as out:
            for part in teacher_parts:
                if part.exists():
                    with part.open("r", encoding="utf-8") as f:
                        shutil.copyfileobj(f, out)
        print(f"[colab] combined teacher data: {combined_teacher}", flush=True)
        active_processed_dir = ROOT / args.teacher_dir
        run(
            [
                sys.executable,
                "-m",
                "airapix.training.merge_teacher_data",
                "--base-dir",
                str(processed_dir),
                "--teacher-jsonl",
                str(combined_teacher),
                "--output-dir",
                str(active_processed_dir),
            ],
            env=env,
        )
        print_dataset_summary(active_processed_dir)

    colab_config = write_colab_config(
        ROOT / args.base_training_config,
        ROOT / args.colab_config,
        active_processed_dir,
        args.output_dir,
        args.max_steps,
        args.micro_batch_size,
        args.effective_batch_size,
        args.log_every,
        args.eval_interval,
        args.checkpoint_interval,
    )

    should_probe = not args.skip_probe and not args.load_in_4bit and args.preset not in {"8b", "7b", "3b"}
    if should_probe:
        print("[colab] running VRAM probe...", flush=True)
        run([sys.executable, "-m", "airapix.training.probe_vram", "--config", str(colab_config)], check=False, env=env)
    else:
        print("[colab] skipping VRAM probe for large/4-bit model training", flush=True)

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "-m",
            "airapix.training.train",
            "--config",
            str(colab_config),
            "--log-every",
            str(args.log_every),
        ]
        if args.preset:
            train_cmd.extend(["--preset", args.preset])
        if args.load_in_4bit:
            train_cmd.append("--load-in-4bit")
        if args.use_qlora:
            train_cmd.append("--use-qlora")
        run(train_cmd, env=env)

    maybe_copy_to_drive(
        drive_dir,
        [
            ROOT / "airapix/model/tokenizer/tokenizer.json",
            active_processed_dir,
            ROOT / args.output_dir,
            colab_config,
        ],
    )
    print("[colab] complete", flush=True)


if __name__ == "__main__":
    main()
