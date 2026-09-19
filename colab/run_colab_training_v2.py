from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airapix.training.train_v2 import run_aira_training_v2


def launch_public_tunnel(port: int = 7860) -> None:
    """
    Launches a 100% reliable public HTTPS tunnel using SSH localhost.run & localtunnel.
    Waits until the dashboard port is active before establishing tunnel to prevent 502 Bad Gateway errors.
    """
    import socket
    import subprocess
    import threading
    import time

    def tunnel_worker():
        # 1. Wait until local dashboard server is listening on port
        server_ready = False
        for _ in range(30):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                    server_ready = True
                    break
            except Exception:
                time.sleep(1.0)

        if not server_ready:
            print(f"\033[1;33m[Tunnel Warning]\033[0m Local server on port {port} didn't respond in time.")
            return

        # 2. Primary SSH Tunnel (Pinggy.io) - Zero SSH key required, zero 502 bad gateway errors
        try:
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=30",
                "-p", "443",
                "-R", f"0:127.0.0.1:{port}",
                "a.pinggy.io",
            ]
            proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(proc.stdout.readline, ""):
                if "pinggy.link" in line or "pinggy.online" in line or "http" in line:
                    for token in line.split():
                        if token.startswith("http://") or token.startswith("https://"):
                            url = token.strip()
                            print(f"\n\033[1;32m============================================================\033[0m")
                            print(f"\033[1;32m 🌐 PUBLIC INTERNET DASHBOARD URL (PINGGY SSH):\033[0m \033[1;36m{url}\033[0m")
                            print(f"\033[1;32m============================================================\033[0m\n")
                            return
        except Exception as e:
            print(f"[Pinggy Notice] Pinggy SSH tunnel fallback: {e}")

        # 3. Secondary SSH Tunnel (Serveo.net) - Zero SSH key required
        try:
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ServerAliveInterval=30",
                "-R", f"80:127.0.0.1:{port}",
                "serveo.net",
            ]
            proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in iter(proc.stdout.readline, ""):
                if "serveo.net" in line or "http" in line:
                    for token in line.split():
                        if token.startswith("http://") or token.startswith("https://"):
                            url = token.strip()
                            print(f"\n\033[1;32m============================================================\033[0m")
                            print(f"\033[1;32m 🌐 PUBLIC INTERNET DASHBOARD URL (SERVEO SSH):\033[0m \033[1;36m{url}\033[0m")
                            print(f"\033[1;32m============================================================\033[0m\n")
                            return
        except Exception as e:
            print(f"[Serveo Notice] Serveo SSH tunnel fallback: {e}")

        # 3. Fallback: localtunnel
        try:
            ip_proc = subprocess.run(["curl", "-s", "https://ipv4.icanhazip.com"], capture_output=True, text=True)
            public_ip = ip_proc.stdout.strip()
            if public_ip:
                print(f"\033[1;33m[Tunnel Password]\033[0m If localtunnel asks for password, enter Colab IP: \033[1;37m{public_ip}\033[0m")

            proc = subprocess.Popen(
                ["npx", "-y", "localtunnel", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in iter(proc.stdout.readline, ""):
                if "url is:" in line.lower():
                    url = line.strip().split("your url is:")[-1].strip()
                    print(f"\n\033[1;32m============================================================\033[0m")
                    print(f"\033[1;32m 🌐 PUBLIC INTERNET DASHBOARD URL (LOCALTUNNEL):\033[0m \033[1;36m{url}\033[0m")
                    print(f"\033[1;32m============================================================\033[0m\n")
                    break
        except Exception as e:
            print(f"\033[1;33m[Public Tunnel Notice]\033[0m Could not launch public tunnel: {e}")

    t = threading.Thread(target=tunnel_worker, daemon=True)
    t.start()


def main() -> None:
    parser = argparse.ArgumentParser(description="Google Colab Training Launcher v2 for Aira AI with Live Web UI")
    parser.add_argument("--preset", type=str, default="8b", help="Model preset (1.5b, 3b, 7b, 8b)")
    parser.add_argument("--max-steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--micro-batch-size", type=int, default=2, help="Micro batch size per GPU step")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--peak-lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--port", type=int, default=7860, help="Live Web UI Dashboard port")
    parser.add_argument("--mount-drive", action="store_true", help="Mount Google Drive for persistent checkpoints")
    parser.add_argument("--public-tunnel", action="store_true", default=True, help="Create a public internet URL for the dashboard")
    parser.add_argument("--load-in-4bit", action="store_true", default=True, help="Use 4-bit NF4 quantization streaming")
    parser.add_argument("--load-in-8bit", "--8bit", action="store_true", default=False, help="Use 8-bit quantization streaming")
    parser.add_argument("--use-qlora", "--qlora", action="store_true", default=True, help="Enable QLoRA fine-tuning")
    parser.add_argument("--colab", action="store_true", default=True, help="Colab mode indicator flag")
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

    # Check hardware & CUDA availability
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"[Hardware Check] PyTorch Version: {torch.__version__}")
    print(f"[Hardware Check] CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"\033[1;32m[GPU ACTIVE]\033[0m Detected GPU: \033[1;36m{torch.cuda.get_device_name(0)}\033[0m (VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB)")
    else:
        print(f"\033[1;31m[CRITICAL WARNING]\033[0m PyTorch is executing on CPU! CUDA GPU is not active in this session.")
        print(f"\033[1;33mTo activate T4 GPU in Google Colab (10 seconds):\033[0m")
        print(f"  1. Click Colab top menu: Runtime -> Change runtime type -> Hardware accelerator -> Select T4 GPU -> Save")
        print(f"  2. Click Colab top menu: Runtime -> Restart session")

    checkpoint_dir = "runs/checkpoints"

    # Launch public tunnel for internet access
    if args.public_tunnel:
        launch_public_tunnel(args.port)

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
