from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

import torch

from airapix.agent.cognitive_loop import AiraCognitiveAgent, CognitiveTurnOutput
from airapix.inference.generate import TokenizerWrapper, load_model, generate_response


def render_bar(value: float, width: int = 10) -> str:
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    empty = width - filled
    return "[" + "=" * filled + "-" * empty + "]"


def print_cognitive_banner(turn: CognitiveTurnOutput) -> None:
    chems = turn.neurochemicals
    dop = chems.get("dopamine", 0.5)
    ser = chems.get("serotonin", 0.5)
    nor = chems.get("noradrenaline", 0.2)
    oxy = chems.get("oxytocin", 0.6)
    cor = chems.get("cortisol", 0.15)

    print("\n" + "=" * 76)
    print(f"[*] AIRA NEUROCHEMICAL STATE | Mode: {turn.cognitive_mode.upper()}")
    print("-" * 76)
    print(
        f"  [Dopamine     {render_bar(dop)} {dop:.2f}]   "
        f"  [Serotonin    {render_bar(ser)} {ser:.2f}]\n"
        f"  [Noradrenaline{render_bar(nor)} {nor:.2f}]   "
        f"  [Oxytocin     {render_bar(oxy)} {oxy:.2f}]\n"
        f"  [Cortisol     {render_bar(cor)} {cor:.2f}]"
    )
    print("-" * 76)

    # Trust Profile & Cognitive Drives HUD
    tp = turn.trust_profile
    u_trust = tp.get("user_trust", 0.50) * 100.0
    ai_trust = tp.get("ai_model_trust", 0.20) * 100.0
    cd = turn.cognitive_drives
    stamina = cd.get("stamina", 1.0) * 100.0
    curiosity = cd.get("curiosity", 0.75) * 100.0
    flow = " [*FLOW STATE*]" if cd.get("flow_state") else ""
    print(
        f"  [User Trust   {render_bar(u_trust/100)} {u_trust:.1f}% (Max: 100%)]   "
        f"  [AI Peer Trust{render_bar(ai_trust/60)} {ai_trust:.1f}% (Cap: 60%)]\n"
        f"  [Stamina      {render_bar(stamina/100)} {stamina:.0f}%]{flow}   "
        f"  [Curiosity    {render_bar(curiosity/100)} {curiosity:.0f}%]"
    )
    print("-" * 76)

    # Theory of Mind display
    em = turn.user_emotion
    sec = f", {em.secondary_emotion.upper()}" if em.secondary_emotion else ""
    print(
        f"[Perception Head] Emotion: {em.primary_emotion.upper()}{sec} "
        f"| Valence: {em.valence:+.2f} | Arousal: {em.arousal:.2f} | Intent: {em.epistemic_intent.upper()}"
    )
    if em.unstated_needs:
        print(f"[Unspoken Needs]  {'; '.join(em.unstated_needs)}")

    # Vault memories
    if turn.retrieved_memories:
        print("-" * 76)
        print(f"[Memory Vault] Found {len(turn.retrieved_memories)} relevant notes from C:\\Users\\WaveWalker\\Aira-Vault:")
        for idx, mem in enumerate(turn.retrieved_memories, 1):
            val_tag = " [High Dopamine]" if mem['somatic_valence'] > 0.8 else ""
            print(f"  {idx}. {mem['title']} (Score: {mem['score']:.2f}){val_tag}")
    print("=" * 76 + "\n")


def find_latest_checkpoint(checkpoint_dir: str = "runs/checkpoints") -> Optional[Path]:
    p = Path(checkpoint_dir)
    if not p.exists():
        alt = Path("runs/phase3_125m")
        if alt.exists():
            p = alt
        else:
            return None
    pts = list(p.glob("*.pt"))
    if not pts:
        return None
    import re

    def sort_key(x: Path):
        m = re.search(r"step_(\d+)", x.name)
        if m:
            return (1, int(m.group(1)))
        return (0, x.stat().st_mtime)

    pts.sort(key=sort_key, reverse=True)
    return pts[0]


def run_cognitive_chat(
    checkpoint_path: Optional[str] = None,
    tokenizer_path: str = "airapix/model/tokenizer/tokenizer.json",
    preset: str = "125m",
    vault_path: Optional[str] = None,
    max_tokens: int = 256,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> None:
    agent = AiraCognitiveAgent(vault_path=vault_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve checkpoint
    ckpt_file = Path(checkpoint_path) if checkpoint_path else find_latest_checkpoint()
    model = None
    tokenizer = None

    if ckpt_file and ckpt_file.is_file():
        try:
            print(f"[*] Loading model from checkpoint: {ckpt_file} onto {device}...")
            tokenizer = TokenizerWrapper(tokenizer_path)
            model = load_model(str(ckpt_file), preset=preset, device=device)
            print("[+] Model loaded successfully!")
        except Exception as e:
            print(f"[Error]: Could not load checkpoint weights ({e}). Model generation will fail.")
            model = None
    else:
        print("[Error]: No checkpoint found or specified. Model generation will fail.")

    print("\n" + "#" * 76)
    print("           AIRA COGNITIVE AGENT INTERFACE — LIVE DUAL-SYSTEM          ")
    print("#" * 76)
    print("Commands:")
    print("  /reward [note]   - Give Aira a dopamine reward hit on a correct answer")
    print("  /penalty [note]  - Register an error/correction (raises noradrenaline)")
    print("  /chemicals       - Print full neurochemical state vector")
    print("  /vault <query>   - Query the Obsidian memory vault directly")
    print("  /history         - View previous chat sessions with topics and notes")
    print("  /dream [count]   - Run autonomous self-play dream consolidation cycle (Strategy 5)")
    print("  /note <text>     - Record an important constraint or note for this session")
    print("  /decision <text> - Record a milestone decision for this session")
    print("  /topic <name>    - Set or update the current session topic title")
    print("  /exit, /quit     - End session & commit conversation to Vault\n")

    agent.start_session(topic="Interactive Consultation")
    last_user_query = ""

    while True:
        try:
            user_input = input("You> ").strip()
            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit"}:
                print("\n[Aira]: Saving session memory to vault...")
                saved_path = agent.end_session()
                if saved_path:
                    print(f"[+] Conversation successfully recorded to: {saved_path}")
                    print("[+] Daily Note updated with session transcript wikilink.")
                print("Goodbye! Until next time.\n")
                break

            # Slash commands
            if user_input.startswith("/reward"):
                parts = user_input.split(" ", 1)
                note = parts[1] if len(parts) > 1 else (last_user_query or "Interactive milestone")
                res = agent.reward_task_success(
                    task_name=f"Verified: {note[:30]}",
                    summary=f"User confirmed correct response for: {note}",
                    complexity=1.2
                )
                print(f"\n[+] Dopamine Hit! (+{res['reward']['delta']:.2f}) New Level: {res['reward']['dopamine']:.2f}")
                print(f"[+] Episodic memory logged to: {res['memory_written']}\n")
                continue

            if user_input.startswith("/penalty"):
                parts = user_input.split(" ", 1)
                reason = parts[1] if len(parts) > 1 else "User flagged incorrect result"
                res = agent.penalize_anomaly(reason, severity=1.0)
                print(f"\n[!] Anomaly Flagged. Vigilance (Noradrenaline) raised to: {res['alert']['noradrenaline']:.2f}\n")
                continue

            if user_input.startswith("/chemicals"):
                chems = agent.neuromodulator.state
                print(f"\n[*] Neurochemical Vector: {chems}\n")
                continue

            if user_input.startswith("/trust"):
                t = agent.synapse_engine.trust
                print(f"\n[*] Trust Profile Audit:")
                print(f"  - User Trust: {t.user_trust * 100:.1f}% (Max Ceiling: 100%)")
                print(f"  - Peer AI Model Trust: {t.ai_model_trust * 100:.1f}% (Strict Cap: 60%)")
                print(f"  - Collaborations: {t.total_collaborations} | Milestones: {t.successful_milestones}")
                print(f"  - Peer AI Claims Audited: {t.ai_claims_audited} | Hallucinations Caught: {t.ai_hallucinations_caught}\n")
                continue

            if user_input.startswith("/correct"):
                parts = user_input.split(" ", 1)
                if len(parts) > 1 and last_user_query:
                    correct_em = parts[1].strip()
                    res = agent.correct_emotion(last_user_query, "neutral", correct_em)
                    print(f"\n[+] Synaptic Recalibration Complete! Learned '{correct_em}' for query: '{last_user_query}'")
                    print(f"[+] Mutual trust deepened to: {res['new_user_trust'] * 100:.1f}%\n")
                else:
                    print("\n[!] Usage: /correct <emotion_name> (e.g. /correct burnout, /correct crisis)\n")
                continue

            if user_input.startswith("/vault"):
                query = user_input.replace("/vault", "").strip()
                results = agent.memory_vault.search(query, top_k=5)
                print(f"\n[*] Search results for '{query}':")
                for r in results:
                    print(f"  - [{r.score:.2f}] {r.title} ({r.path}): {r.snippet[:120]}...")
                print()
                continue

            if user_input.startswith("/history"):
                sessions = agent.memory_vault.get_chat_sessions(limit=5)
                print(f"\n[*] Recent Chat Sessions in C:\\Users\\WaveWalker\\Aira-Vault\\Conversations:")
                if not sessions:
                    print("  - No previous recorded chat sessions found.")
                for s in sessions:
                    print(f"  - [{s.get('created', 'N/A')}] {s.get('title', 'Session')}")
                    if s.get("overview"):
                        print(f"    Overview: {s['overview'][:140]}...")
                    if s.get("notes"):
                        for n in s["notes"][:2]:
                            print(f"    [Note]: {n}")
                print()
                continue

            if user_input.startswith("/dream"):
                parts = user_input.split(" ", 1)
                count = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 5
                print(f"\n[*] Initiating Autonomous Dream Consolidation ({count} reasoning seeds from Vault)...")
                res = agent.run_dream_consolidation(num_dreams=count, merge_into_training=True)
                print(f"[+] Dream Cycle Complete! Synthesized: {res['dreams_synthesized']} reasoning experiences.")
                print(f"[+] Somatic Valence: {res['average_somatic_valence']:.2f} (Damasio Verified).")
                print(f"[+] Ingested into continual training stream: {res['processed_dataset_path']}\n")
                continue

            if user_input.startswith("/note"):
                note_text = user_input.replace("/note", "").strip()
                if note_text:
                    agent.add_session_note(note_text)
                    print(f"\n[+] Recorded Session Note: '{note_text}'\n")
                else:
                    print("\n[!] Usage: /note <important fact or constraint>\n")
                continue

            if user_input.startswith("/decision"):
                dec_text = user_input.replace("/decision", "").strip()
                if dec_text:
                    agent.add_session_decision(dec_text)
                    print(f"\n[+] Recorded Session Decision: '{dec_text}'\n")
                else:
                    print("\n[!] Usage: /decision <architecture or roadmap decision>\n")
                continue

            if user_input.startswith("/topic"):
                topic_text = user_input.replace("/topic", "").strip()
                if topic_text:
                    agent.session_topic = topic_text
                    print(f"\n[+] Updated Session Topic to: '{topic_text}'\n")
                else:
                    print("\n[!] Usage: /topic <topic title>\n")
                continue

            last_user_query = user_input

            # Run full cognitive loop
            turn = agent.process_turn(user_input)
            print_cognitive_banner(turn)

            response_text = ""

            # Generate response
            if model is None or tokenizer is None:
                print("\n[Error]: Model checkpoint is not loaded. Cannot generate response.\n")
                continue

            print("[Aira Thinking...]")
            gen_params = turn.generation_params
            actual_temp = temperature if temperature is not None else gen_params.get("temperature", 0.7)
            actual_top_k = top_k if top_k is not None else 50

            # Format strictly according to Aira's pre-training chat template
            if turn.retrieved_memories:
                vault_context = "\n".join(
                    f"[Memory Vault: {m['title']}]: {m['snippet'][:140]}"
                    for m in turn.retrieved_memories
                )
                prompt_input = f"{vault_context}\n\n{user_input}"
            else:
                prompt_input = user_input

            full_prompt = f"User:\n{prompt_input}\n\nAssistant:\n"

            try:
                response_text = generate_response(
                    model,
                    tokenizer,
                    full_prompt,
                    max_new_tokens=max_tokens,
                    temperature=actual_temp,
                    top_k=actual_top_k,
                    device=device,
                )
                if response_text:
                    print(f"\n[Aira]:\n{response_text}\n")
                else:
                    print("\n[Error]: Model generated an empty response.\n")
            except Exception as e:
                print(f"\n[Error]: Model generation failed: {e}\n")
                response_text = ""

            # Log turn into active session only on valid response
            if response_text:
                agent.log_turn(
                    user_text=user_input,
                    aira_response=response_text,
                    turn_output=turn
                )

        except (KeyboardInterrupt, EOFError):
            print("\n[!] Session interrupted. Committing memory to vault...")
            saved = agent.end_session()
            if saved:
                print(f"[+] Session saved to: {saved}")
            break


def main():
    parser = argparse.ArgumentParser(description="Aira Cognitive Agent Interactive Interface")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint .pt (default: finds latest in runs/checkpoints)")
    parser.add_argument("--tokenizer", default="airapix/model/tokenizer/tokenizer.json", help="Path to 24k tokenizer.json")
    parser.add_argument("--preset", default="125m", help="Model preset (125m, 90m, 60m, tiny)")
    parser.add_argument("--vault", default=None, help="Path to Obsidian vault (default: C:\\Users\\WaveWalker\\Aira-Vault)")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--temp", "--temperature", type=float, default=None, help="Sampling temperature (e.g. 0.3)")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling threshold (e.g. 5)")
    args = parser.parse_args()

    run_cognitive_chat(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        preset=args.preset,
        vault_path=args.vault,
        max_tokens=args.max_tokens,
        temperature=args.temp,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
