from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import time
from typing import Any, Dict, List, Optional, Set

from airapix.agent.neuromodulator import Neuromodulator
from airapix.agent.theory_of_mind import TheoryOfMindEngine, AffectiveCognitiveState
from airapix.agent.memory_vault import MemoryVaultEngine, MemorySearchResult
from airapix.agent.synapses import AffectiveSynapseEngine, TrustProfile, CognitiveDrives
from airapix.agent.dream_engine import AiraDreamEngine


@dataclass
class CognitiveTurnOutput:
    user_emotion: AffectiveCognitiveState
    neurochemicals: Dict[str, float]
    trust_profile: Dict[str, float]
    cognitive_drives: Dict[str, Any]
    cognitive_mode: str
    generation_params: Dict[str, float]
    retrieved_memories: List[Dict[str, Any]]
    thinking_prompt: str


class AiraCognitiveAgent:
    """
    Unified Cognitive Agent Runtime for Aira.
    Integrates:
    - Plastic Affective Synapses & Online Hebbian Learning
    - Dual-Track Trust Architecture (User: 50-100%, Peer AI: 20-60%)
    - Neuromodulation (Doya 4-Chemicals & Damasio Somatic Markers)
    - Affective Theory of Mind (Emotion & Epistemic Need Perception)
    - High-Performance Inverted Index Memory Vault
    - Metacognitive System 2 Thinking Traces (<think>)
    """

    def __init__(self, vault_path: Optional[str] = None):
        self.synapse_engine = AffectiveSynapseEngine()
        self.neuromodulator = Neuromodulator()
        self.tom_engine = TheoryOfMindEngine(synapse_engine=self.synapse_engine)
        self.memory_vault = MemoryVaultEngine(vault_path)
        self.dream_engine = AiraDreamEngine(
            vault_path=self.memory_vault.vault_path,
            neuromodulator=self.neuromodulator,
            synapse_engine=self.synapse_engine,
        )

        # Active Session State
        self.current_session_id = f"aira_sess_{int(time.time())}"
        self.session_start_time = time.time()
        self.session_topic = "Interactive Consultation"
        self.session_description = ""
        self.session_turns: List[Dict[str, Any]] = []
        self.session_notes: List[str] = []
        self.session_decisions: List[str] = []
        self.session_keywords: Set[str] = set()

    def start_session(self, topic: str = "Interactive Consultation", description: str = "") -> None:
        """Starts a fresh conversation session with empty turn buffers."""
        self.current_session_id = f"aira_sess_{int(time.time())}"
        self.session_start_time = time.time()
        self.session_topic = topic
        self.session_description = description
        self.session_turns = []
        self.session_notes = []
        self.session_decisions = []
        self.session_keywords = set()

    def log_turn(
        self,
        user_text: str,
        aira_response: str,
        turn_output: CognitiveTurnOutput,
        important_note: Optional[str] = None,
        decision: Optional[str] = None
    ) -> None:
        """Logs a completed dialogue turn and extracts topics, notes, and decisions."""
        now_str = time.strftime("%H:%M:%S")

        # 1. Append User Turn
        self.session_turns.append({
            "timestamp": now_str,
            "sender": "User",
            "text": user_text,
            "emotion": turn_output.user_emotion.primary_emotion,
            "valence": turn_output.user_emotion.valence
        })

        # 2. Append Aira Turn
        self.session_turns.append({
            "timestamp": now_str,
            "sender": "Aira",
            "text": aira_response,
            "cognitive_mode": turn_output.cognitive_mode,
            "somatic_valence": self.neuromodulator.state.dopamine
        })

        # 3. Explicit Notes / Decisions
        if important_note:
            self.session_notes.append(important_note)
        if decision:
            self.session_decisions.append(decision)

        # 4. Heuristic note and decision extraction
        lower_user = user_text.lower()
        if any(kw in lower_user for kw in ["remember that", "take note", "important:", "keep in mind"]):
            clean_note = re.sub(r"^(?:remember that|take note|important:|keep in mind)\s*", "", user_text, flags=re.IGNORECASE).strip()
            if clean_note and clean_note not in self.session_notes:
                self.session_notes.append(clean_note)

        if any(kw in lower_user for kw in ["decision:", "let's decide", "we will use", "we chose", "going with"]):
            clean_dec = re.sub(r"^(?:decision:|let's decide|we will use|we chose)\s*", "", user_text, flags=re.IGNORECASE).strip()
            if clean_dec and clean_dec not in self.session_decisions:
                self.session_decisions.append(clean_dec)

        # 5. Extract topic keywords
        words = re.findall(r"[a-zA-Z0-9_]{3,}", user_text.lower())
        stop_words = {"the", "and", "that", "this", "what", "with", "from", "have", "will", "your", "about"}
        for w in words:
            if w not in stop_words:
                self.session_keywords.add(w)

        # Dynamic Topic Title inference if still default
        if self.session_topic == "Interactive Consultation" and len(self.session_turns) >= 2:
            top_words = list(self.session_keywords)[:4]
            if top_words:
                self.session_topic = " ".join(w.capitalize() for w in top_words)

    def add_session_note(self, note: str) -> None:
        """Manually records an important note into the active session."""
        self.session_notes.append(note.strip())

    def add_session_decision(self, decision: str) -> None:
        """Manually records a decision into the active session."""
        self.session_decisions.append(decision.strip())

    def end_session(
        self,
        topic_override: Optional[str] = None,
        topic_description: Optional[str] = None
    ) -> Optional[str]:
        """
        Commits the active session into the Obsidian memory vault across all chats:
        - Writes markdown note with YAML frontmatter to Conversations/
        - Updates today's daily note
        - Re-indexes memory vault
        """
        if not self.session_turns:
            return None

        topic = topic_override or self.session_topic
        desc = topic_description or self.session_description
        if not desc:
            desc = (
                f"Interactive cognitive session covering {topic}. "
                f"Logged {len(self.session_turns) // 2} conversational exchanges with "
                f"final user trust at {self.synapse_engine.trust.user_trust * 100:.1f}%."
            )

        saved_path = self.memory_vault.record_chat_session(
            session_id=self.current_session_id,
            topic=topic,
            topic_description=desc,
            turns=self.session_turns,
            start_time=self.session_start_time,
            end_time=time.time(),
            important_notes=self.session_notes,
            decisions_made=self.session_decisions,
            keywords=list(self.session_keywords),
            user_trust=self.synapse_engine.trust.user_trust,
            somatic_valence=self.neuromodulator.state.dopamine
        )

        return saved_path

    def process_turn(self, user_input: str) -> CognitiveTurnOutput:
        """
        Processes a human interaction through the full cognitive loop:
        1. Decays chemicals towards homeostatic baselines.
        2. Detects user emotion & epistemic state (Theory of Mind).
        3. Updates Aira's internal neurochemical vector.
        4. Queries memory vault for relevant past solutions and past chat history.
        5. Formulates generation directives and thinking trace prompts.
        """
        # 1. Decay toward baseline
        self.neuromodulator.state.decay()

        # 2. Theory of Mind perception
        emotion_state = self.tom_engine.analyze(user_input)

        # 3. Modulate Aira's neurochemicals
        self.neuromodulator.on_user_emotion_perceived(
            emotion=emotion_state.primary_emotion,
            intensity=emotion_state.arousal
        )

        # 4. Retrieve from memory vault
        search_results = self.memory_vault.search(user_input, top_k=3)
        retrieved_data = [
            {
                "title": r.title,
                "path": r.path,
                "score": r.score,
                "snippet": r.snippet,
                "somatic_valence": r.somatic_valence
            }
            for r in search_results
        ]

        # 4b. Cross-chat history recall if user references previous interactions
        memory_trigger = any(phrase in user_input.lower() for phrase in [
            "remember", "last chat", "previous conversation", "earlier session",
            "yesterday", "we discussed", "what did we talk about", "our last topic",
            "past note", "last time"
        ])
        past_session_snippets = []
        if memory_trigger:
            past_sessions = self.memory_vault.search_chat_history(user_input, top_k=2)
            for ps in past_sessions:
                past_session_snippets.append(f"- Past Session {ps.title}: {ps.snippet[:160]}...")

        # 5. Gating & Thinking trace prompt
        cognitive_mode = self.neuromodulator.get_cognitive_mode()
        gen_params = self.neuromodulator.get_generation_parameters()

        # Assemble metacognitive guidance
        guidance_parts = []
        tom_prompt = self.tom_engine.generate_empathy_prompt(emotion_state)
        if tom_prompt:
            guidance_parts.append(tom_prompt)

        if past_session_snippets:
            guidance_parts.append("[Recalled Cross-Chat Past Sessions from Vault]")
            guidance_parts.extend(past_session_snippets)

        if retrieved_data:
            guidance_parts.append("[Retrieved Vault Context]")
            for item in retrieved_data:
                guidance_parts.append(f"- Note {item['title']}: {item['snippet'][:150]}...")

        if cognitive_mode in {"deliberate", "protective_grounding"}:
            guidance_parts.append(
                "[System 2 Metacognitive Directive]\n"
                "- Enforce internal verification before responding.\n"
                "- Use <think> to verify intermediate calculations, check assumptions, and ground tone."
            )

        thinking_prompt = "\n\n".join(guidance_parts)

        # Update cognitive drives for the turn
        self.synapse_engine.drives.on_turn_executed(
            arousal=emotion_state.arousal,
            success=True
        )

        return CognitiveTurnOutput(
            user_emotion=emotion_state,
            neurochemicals=asdict(self.neuromodulator.state),
            trust_profile=asdict(self.synapse_engine.trust),
            cognitive_drives=asdict(self.synapse_engine.drives),
            cognitive_mode=cognitive_mode,
            generation_params=gen_params,
            retrieved_memories=retrieved_data,
            thinking_prompt=thinking_prompt
        )

    def reward_task_success(self, task_name: str, summary: str, complexity: float = 1.0) -> Dict[str, Any]:
        """Awards dopamine surge, reinforces synapses, logs episodic memory, and increases user trust."""
        reward_event = self.neuromodulator.on_task_success(complexity=complexity)
        syn_update = self.synapse_engine.learn_from_experience(
            text=f"{task_name} {summary}",
            experienced_emotion="joy",
            user_confirmed=True,
            user_reward=1.0 * complexity
        )
        memory_path = self.memory_vault.record_episodic_memory(
            title=task_name,
            summary=summary,
            details=f"Solved successfully with complexity {complexity}. Dopamine surge: {reward_event['delta']:.2f}. User trust: {self.synapse_engine.trust.user_trust * 100:.1f}%.",
            dopamine_verified=True
        )
        return {
            "reward": reward_event,
            "synaptic_update": syn_update,
            "memory_written": memory_path,
            "new_state": asdict(self.neuromodulator.state),
            "user_trust": self.synapse_engine.trust.user_trust
        }

    def penalize_anomaly(self, error_message: str, severity: float = 1.0) -> Dict[str, Any]:
        """Penalizes errors: raises vigilance (noradrenaline), stress (cortisol), and weakens associations."""
        alert_event = self.neuromodulator.on_task_failure(severity=severity)
        self.synapse_engine.learn_from_experience(
            text=error_message,
            experienced_emotion="crisis",
            user_confirmed=False
        )
        return {
            "alert": alert_event,
            "new_state": asdict(self.neuromodulator.state)
        }

    def correct_emotion(self, text: str, wrong_emotion: str, correct_emotion: str) -> Dict[str, Any]:
        """Online delta correction when user teaches Aira their true emotional state."""
        self.synapse_engine.correct_emotion_synapse(text, wrong_emotion, correct_emotion)
        return {
            "status": "synapse_recalibrated",
            "text": text,
            "learned_emotion": correct_emotion,
            "new_user_trust": self.synapse_engine.trust.user_trust
        }

    def audit_external_ai_claim(self, claim: str, is_valid: bool) -> float:
        """Audits third-party AI models. Trust is strictly capped at 60%."""
        return self.synapse_engine.trust.audit_external_ai_claim(is_valid)

    def run_dream_consolidation(self, num_dreams: int = 5, merge_into_training: bool = True) -> Dict[str, Any]:
        """
        Runs autonomous sleep & self-play consolidation cycle (Strategy 5).
        Synthesizes System 2 deliberative reasoning from vault seeds,
        validates via Somatic Markers, and deposits into continual training stream.
        """
        return self.dream_engine.run_dream_cycle(
            num_dreams=num_dreams,
            merge_into_processed=merge_into_training
        )
