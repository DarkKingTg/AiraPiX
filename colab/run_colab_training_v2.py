from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airapix.training.train_v2 import run_aira_training_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Colab Training Launcher v2 for Aira AI with Live Web UI")
    parser.add_argument("--preset", type=str, default="8b", help="Model preset (1.5b, 3b, 7b, 8b)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--micro-batch-size", type=int, default=2, help="Micro batch size per GPU step")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--peak-lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--port", type=int, default=7860, help="Live Web UI Dashboard port")
    parser.add_argument("--mount-drive", action="store_true", help="Mount Google Drive for persistent checkpoints")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Use 4-bit NF4 quantization streaming")
    parser.add_argument("--use-qlora", action="store_true", default=True, help="Enable QLoRA training")
    args = parser.parse_args()

    print(f"\n\033[1;35m============================================================\033[0m")
    print(f"\033[1;35m AIRA AI GOOGLE COLAB TRAINING LAUNCHER v2\033[0m")
    print(f"\033[1;35m============================================================\033[0m")

    # Check if running inside Google Colab
    in_colab = False
    try:
        import google.colab
        in_colab = True
        print("\033[1;32m[Colab]\033[0m Google Colab environment detected!")
    except ImportError:
        print("\033[1;33m[Colab]\033[0m Running in standard Python environment.")

    checkpoint_dir = "runs/checkpoints"

    # Mount Google Drive if requested
    if args.mount_drive and in_colab:
        try:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            checkpoint_dir = "/content/drive/MyDrive/AiraCheckpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            print(f"\033[1;32m[Google Drive]\033[0m Checkpoints will be saved to: {checkpoint_dir}")
        except Exception as e:
            print(f"\033[1;33m[Google Drive Notice]\033[0m Drive auto-mount skipped ({e}). Checkpoints saving locally to {checkpoint_dir}")

    # Expose Colab port if running in Colab (Embed Inline Frame)
    if in_colab:
        try:
            from google.colab import output
            try:
                output.serve_kernel_port_as_iframe(args.port, height="650")
                print(f"\033[1;36m[Colab Web UI]\033[0m Rendering Live Training Dashboard inline in Colab cell frame (Port {args.port}).")
            except Exception:
                output.serve_kernel_port_as_window(args.port)
                print(f"\033[1;36m[Colab Web UI Window]\033[0m Exposed port {args.port} for live dashboard monitor.")
        except Exception as e:
            print(f"\033[1;33m[Colab Port Notice]\033[0m Port window notice: {e}")

    # Launch Training Loop
    run_aira_training_v2(
        preset=args.preset,
        max_steps=args.max_steps,
        batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        peak_lr=args.peak_lr,
        save_every=200,
        checkpoint_dir=checkpoint_dir,
        dashboard_port=args.port,
        use_qlora=args.use_qlora,
        load_in_4bit=args.load_in_4bit,
        colab_mode=in_colab,
    )


if __name__ == "__main__":
    main()
