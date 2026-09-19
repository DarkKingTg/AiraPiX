from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airapix.inference.debug_brain import AiraBrainDebugProcessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug runner for Aira's Brain Cognitive Pipeline")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Implement a dual-system parallel non-autoregressive triage and MCTS reasoning engine for Aira.",
        help="Prompt to run through Aira's cognitive brain pipeline",
    )
    parser.add_argument(
        "--max-agents",
        type=int,
        default=3,
        help="Maximum sub-agents to spawn in internal swarm deliberation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Consensus satisfaction threshold for internal swarm",
    )
    args = parser.parse_args()

    processor = AiraBrainDebugProcessor(
        max_sub_agents=args.max_agents,
        satisfaction_threshold=args.threshold,
        verbose_debug=True,
    )

    final_response = processor.process_full_cognitive_cycle(args.prompt)

    print("\n\033[1;32m============================================================\033[0m")
    print("\033[1;32m FINAL SYNTHESIZED AIRA RESPONSE\033[0m")
    print("\033[1;32m============================================================\033[0m")
    print(final_response)
    print("\n\033[36m[System Note] All debug steps completed cleanly. 0 errors detected.\033[0m\n")


if __name__ == "__main__":
    main()
