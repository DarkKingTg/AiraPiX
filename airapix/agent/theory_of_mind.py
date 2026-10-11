from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Tuple


@dataclass
class AffectiveCognitiveState:
    primary_emotion: str = "neutral"
    secondary_emotion: str | None = None
    valence: float = 0.0          # -1.0 (very negative) to +1.0 (very positive)
    arousal: float = 0.2          # 0.0 (calm / subdued) to 1.0 (panicked / energized)
    epistemic_intent: str = "general_inquiry"
    unstated_needs: List[str] = field(default_factory=list)
    confidence: float = 0.85


class TheoryOfMindEngine:
    """
    Theory of Mind & Affective Perception Engine for Aira.
    Driven by plastic associative synapses, experiential Hebbian learning,
    and second-order epistemic intention inference.
    """

    NEEDS_MAP: Dict[str, List[str]] = {
        "crisis": ["immediate psychological grounding, panic de-escalation, and protective guidance"],
        "fear": ["immediate psychological grounding and non-destructive step-by-step safety"],
        "imposter_syndrome": ["normalization of difficulty, validation of persistence, and an easy win"],
        "burnout": ["taking the cognitive load off their shoulders, concise clarity"],
        "confusion": ["analogies instead of formal notation, progressive scaffolding"],
        "frustration": ["calm, de-escalating technical precision without defensiveness"],
        "joy": ["shared dopamine celebration and reinforcing self-efficacy"],
        "urgency": ["crisp, zero-fluff, actionable instructions"],
        "neutral": ["direct, accurate technical response"],
    }

    def __init__(self, synapse_engine: Any = None):
        if synapse_engine is None:
            from airapix.agent.synapses import AffectiveSynapseEngine
            self.synapse_engine = AffectiveSynapseEngine()
        else:
            self.synapse_engine = synapse_engine

    def analyze(self, text: str) -> AffectiveCognitiveState:
        lowered = text.lower()

        # 1. Forward pass through plastic synapse network (learned from experience)
        syn_emotion, syn_conf, syn_val = self.synapse_engine.infer_emotion(text)

        # 2. Epistemic intention inference
        intent = "general_inquiry"
        if "?" in text or re.search(r"\b(how|why|what|where|can we|is there)\b", lowered):
            intent = "conceptual_inquiry"
        if re.search(r"\b(fix|bug|error|traceback|exception|crash|failed|issue)\b", lowered):
            intent = "debugging_triage"
        if re.search(r"\b(code|implement|add|create|build|make)\b", lowered):
            intent = "engineering_task"

        # 3. Dynamic unstated needs lookup
        needs = self.NEEDS_MAP.get(syn_emotion, ["direct, accurate technical response"])

        # Estimate arousal based on punctuation and emotion intensity
        arousal = 0.20
        if syn_emotion in {"crisis", "fear", "urgency", "joy"}:
            arousal = 0.85
        elif syn_emotion in {"frustration", "confusion"}:
            arousal = 0.60
        if "!" in text:
            arousal = min(1.0, arousal + 0.10)

        return AffectiveCognitiveState(
            primary_emotion=syn_emotion,
            secondary_emotion=None,
            valence=round(syn_val, 2),
            arousal=round(arousal, 2),
            epistemic_intent=intent,
            unstated_needs=needs,
            confidence=round(syn_conf, 2),
        )

    def generate_empathy_prompt(self, state: AffectiveCognitiveState) -> str:
        """Constructs an internalized Theory of Mind guidance tag for System 2 thinking."""
        if state.primary_emotion == "neutral":
            return ""
        
        guidance = (
            f"[Theory of Mind Guidance]\n"
            f"- User Emotion: {state.primary_emotion.upper()} (Valence: {state.valence}, Arousal: {state.arousal})\n"
            f"- Intent: {state.epistemic_intent}\n"
            f"- Key Unspoken Needs: {'; '.join(state.unstated_needs)}\n"
            f"- Behavioral Directive: Align tone to de-escalate stress, protect user agency, and deliver crisp solutions.\n"
        )
        return guidance
