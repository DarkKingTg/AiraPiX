from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Dict


@dataclass
class NeurochemicalState:
    # Kenji Doya's 4-neurotransmitter model + Cortisol stress biomarker
    dopamine: float = 0.55        # Reward prediction error, motivation, learning reinforcement [0.0, 1.0]
    serotonin: float = 0.50       # Patience, impulse control, long-term discounting [0.0, 1.0]
    noradrenaline: float = 0.20   # Uncertainty, environmental entropy, vigilance [0.0, 1.0]
    acetylcholine: float = 0.50   # Sensory precision, attention focus, suppressing prior bias [0.0, 1.0]
    cortisol: float = 0.15        # Stress, risk aversion, anomaly fatigue [0.0, 1.0]
    oxytocin: float = 0.60        # Empathy, user bonding, protective warmth [0.0, 1.0]
    last_updated: float = 0.0

    def __post_init__(self):
        if self.last_updated == 0.0:
            self.last_updated = time.time()
        self.clamp()

    def clamp(self) -> None:
        self.dopamine = max(0.0, min(1.0, self.dopamine))
        self.serotonin = max(0.0, min(1.0, self.serotonin))
        self.noradrenaline = max(0.0, min(1.0, self.noradrenaline))
        self.acetylcholine = max(0.0, min(1.0, self.acetylcholine))
        self.cortisol = max(0.0, min(1.0, self.cortisol))
        self.oxytocin = max(0.0, min(1.0, self.oxytocin))

    def decay(self, elapsed_seconds: float | None = None) -> None:
        """Homeostatic decay returning chemicals towards natural resting baselines."""
        now = time.time()
        dt = (now - self.last_updated) if elapsed_seconds is None else elapsed_seconds
        self.last_updated = now
        decay_factor = math.exp(-0.01 * min(dt, 300.0))

        # Baselines
        self.dopamine = 0.50 + (self.dopamine - 0.50) * decay_factor
        self.serotonin = 0.50 + (self.serotonin - 0.50) * decay_factor
        self.noradrenaline = 0.20 + (self.noradrenaline - 0.20) * decay_factor
        self.acetylcholine = 0.50 + (self.acetylcholine - 0.50) * decay_factor
        self.cortisol = 0.15 + (self.cortisol - 0.15) * decay_factor
        self.oxytocin = 0.60 + (self.oxytocin - 0.60) * decay_factor
        self.clamp()


class SomaticMarkerEngine:
    """
    Antonio Damasio's Somatic Marker System:
    Attaches visceral emotional tags (positive valence vs risk aversion) to prospective actions.
    Enables rapid heuristic pruning of dangerous or useless cognitive paths in milliseconds.
    """
    def __init__(self):
        # Action category -> valence score in [-1.0, +1.0]
        self.markers: Dict[str, float] = {
            "safe_read": 0.85,
            "verification_test": 0.90,
            "clear_explanation": 0.80,
            "cautious_fix": 0.70,
            "destructive_overwrite": -0.85,
            "unverified_guess": -0.70,
            "premature_optimization": -0.40,
        }

    def evaluate_action(self, action_tag: str) -> float:
        """Returns somatic valence [-1.0 to 1.0]. Values < 0 indicate intuitive risk warning."""
        return self.markers.get(action_tag, 0.0)

    def update_marker(self, action_tag: str, outcome_success: bool, intensity: float = 0.2) -> None:
        current = self.markers.get(action_tag, 0.0)
        delta = intensity if outcome_success else -intensity
        self.markers[action_tag] = max(-1.0, min(1.0, current + delta))


class Neuromodulator:
    """
    Full Neuromodulatory and Affective Engine for Aira.
    Combines Doya's 4-chemical engine, Damasio Somatic Markers, and homeostatic reward loops.
    """
    def __init__(self, state: NeurochemicalState | None = None):
        self.state = state or NeurochemicalState()
        self.somatic_markers = SomaticMarkerEngine()

    def on_task_success(self, complexity: float = 1.0) -> Dict[str, float]:
        """Dopamine reward hit on successful step, test pass, or user confirmation."""
        dopamine_boost = 0.25 * min(2.0, max(0.5, complexity))
        self.state.dopamine += dopamine_boost
        self.state.cortisol -= 0.10
        self.state.serotonin += 0.05
        self.state.noradrenaline -= 0.05
        self.state.clamp()
        return {
            "event": "dopamine_hit",
            "dopamine": self.state.dopamine,
            "cortisol": self.state.cortisol,
            "delta": dopamine_boost
        }

    def on_task_failure(self, severity: float = 1.0) -> Dict[str, float]:
        """Penalty on error, rollback, or anomaly: spikes noradrenaline and cortisol."""
        self.state.dopamine -= 0.15 * severity
        self.state.cortisol += 0.20 * severity
        self.state.noradrenaline += 0.25 * severity  # Increased vigilance to prevent repeated mistakes
        self.state.clamp()
        return {
            "event": "anomaly_alert",
            "cortisol": self.state.cortisol,
            "noradrenaline": self.state.noradrenaline
        }

    def on_user_emotion_perceived(self, emotion: str, intensity: float = 0.5) -> None:
        """Adapts Aira's neurochemical state in response to the user's emotional state."""
        emotion = emotion.lower()
        if emotion in {"fear", "panic", "urgency", "crisis"}:
            # Human is panicked/in crisis -> Aira raises vigilance (noradrenaline) & protective empathy (oxytocin)
            self.state.noradrenaline += 0.25 * intensity
            self.state.oxytocin += 0.30 * intensity
            self.state.serotonin += 0.10 * intensity  # Maintain calm stability
        elif emotion in {"burnout", "exhaustion", "imposter_syndrome", "sadness"}:
            # Human is drained -> Maximize oxytocin and reduce cognitive friction
            self.state.oxytocin += 0.35 * intensity
            self.state.cortisol -= 0.05
        elif emotion in {"joy", "celebration", "pride"}:
            # Shared victory -> Dopamine and oxytocin surge
            self.state.dopamine += 0.30 * intensity
            self.state.oxytocin += 0.20 * intensity
            self.state.cortisol -= 0.15
        elif emotion in {"confusion", "cognitive_overload"}:
            # Human is confused -> High acetylcholine (focus on simple clarity) and high patience (serotonin)
            self.state.acetylcholine += 0.25 * intensity
            self.state.serotonin += 0.20 * intensity
        self.state.clamp()

    def get_cognitive_mode(self) -> str:
        """
        Determines the optimal thinking mode for the current cognitive state:
        - 'reflex': High dopamine, low cortisol/noradrenaline -> fast, confident System 1 response (<50ms).
        - 'deliberate': Moderate/high noradrenaline or complex task -> System 2 thinking trace (<think>).
        - 'protective_grounding': High user distress or high cortisol -> patient, supportive de-escalation.
        """
        self.state.decay()
        if self.state.oxytocin > 0.75 and self.state.noradrenaline > 0.40:
            return "protective_grounding"
        if self.state.noradrenaline > 0.45 or self.state.cortisol > 0.50:
            return "deliberate"
        if self.state.dopamine > 0.65 and self.state.noradrenaline < 0.30:
            return "reflex"
        return "deliberate"

    def get_generation_parameters(self) -> Dict[str, float]:
        """Dynamically tunes inference sampling based on neurochemistry."""
        # High noradrenaline -> lower temperature (more deterministic, fewer wild guesses)
        # High dopamine + serotonin -> healthy exploration (slightly higher temperature)
        temp = 0.70 - (self.state.noradrenaline * 0.30) + (self.state.dopamine * 0.10)
        temp = max(0.20, min(0.95, temp))

        # Top-p sampling tightened during high stress
        top_p = 0.90 if self.state.cortisol < 0.40 else 0.80

        return {
            "temperature": round(temp, 3),
            "top_p": top_p,
            "cognitive_mode": self.get_cognitive_mode()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chemicals": asdict(self.state),
            "somatic_markers": dict(self.somatic_markers.markers),
            "cognitive_mode": self.get_cognitive_mode(),
            "generation_params": self.get_generation_parameters()
        }
