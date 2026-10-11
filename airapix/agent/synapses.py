from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrustProfile:
    # User trust: starts at 50%, can reach 100% maximum
    user_trust: float = 0.50
    user_trust_min: float = 0.05
    user_trust_max: float = 1.00

    # Peer AI Model trust: starts at 20%, strictly capped at 60% maximum
    ai_model_trust: float = 0.20
    ai_model_trust_min: float = 0.00
    ai_model_trust_max: float = 0.60

    total_collaborations: int = 0
    successful_milestones: int = 0
    ai_claims_audited: int = 0
    ai_hallucinations_caught: int = 0

    def adjust_user_trust(self, delta: float) -> None:
        """Adjusts user trust with diminishing returns as it nears 1.0."""
        if delta > 0:
            headroom = self.user_trust_max - self.user_trust
            scaled_delta = delta * (headroom / (self.user_trust_max - self.user_trust_min + 1e-5))
            self.user_trust = min(self.user_trust_max, self.user_trust + scaled_delta)
            self.successful_milestones += 1
        else:
            self.user_trust = max(self.user_trust_min, self.user_trust + delta)
        self.total_collaborations += 1

    def audit_external_ai_claim(self, claim_valid: bool) -> float:
        """Audits third-party AI models. Trust is strictly capped at 60%."""
        self.ai_claims_audited += 1
        if claim_valid:
            # Slow upward climb towards the strict 60% ceiling
            headroom = self.ai_model_trust_max - self.ai_model_trust
            self.ai_model_trust = min(self.ai_model_trust_max, self.ai_model_trust + 0.02 * headroom)
        else:
            self.ai_hallucinations_caught += 1
            self.ai_model_trust = max(self.ai_model_trust_min, self.ai_model_trust - 0.08)
        return self.ai_model_trust


@dataclass
class CognitiveDrives:
    stamina: float = 1.00          # Cognitive energy [0.0, 1.0]
    curiosity: float = 0.75        # Autonomous drive to explore and reduce entropy [0.0, 1.0]
    flow_state: bool = False       # Hyper-focus streak
    consecutive_successes: int = 0
    last_turn_timestamp: float = field(default_factory=time.time)

    def on_turn_executed(self, arousal: float, success: bool) -> None:
        now = time.time()
        dt = now - self.last_turn_timestamp
        self.last_turn_timestamp = now

        # Restores stamina if idle for >5 minutes
        if dt > 300.0:
            self.stamina = min(1.0, self.stamina + 0.20 * min(5.0, dt / 300.0))

        # High arousal / stress depletes stamina slightly
        fatigue = 0.015 * max(0.2, arousal)
        self.stamina = max(0.10, self.stamina - fatigue)

        if success:
            self.consecutive_successes += 1
            if self.consecutive_successes >= 3 and self.stamina > 0.40:
                self.flow_state = True
        else:
            self.consecutive_successes = 0
            self.flow_state = False


class AffectiveSynapseEngine:
    """
    Plastic Hebbian Associative Synapse Network for Aira.
    Learns human emotions, user slang, and conversational dynamics through lived experience.
    Saves and loads learned synaptic weights to disk and Obsidian vault.
    """

    PRIMARY_EMOTIONS = [
        "joy", "crisis", "fear", "frustration", "burnout",
        "confusion", "imposter_syndrome", "urgency", "neutral"
    ]

    DEFAULT_STORE_PATH = Path("runs/aira_brain_synapses.json")
    DEFAULT_VAULT_MIRROR = Path(r"C:\Users\WaveWalker\Aira-Vault\Aira_Synaptic_Memory.json")

    def __init__(self, store_path: Optional[Path] = None, vault_mirror: Optional[Path] = None):
        self.store_path = store_path or self.DEFAULT_STORE_PATH
        self.vault_mirror = vault_mirror or self.DEFAULT_VAULT_MIRROR
        self.trust = TrustProfile()
        self.drives = CognitiveDrives()

        # token -> {emotion: associative_weight}
        self.synaptic_weights: Dict[str, Dict[str, float]] = {}
        self.learning_rate: float = 0.15

        self.load_synapses()
        if not self.synaptic_weights:
            self._seed_foundational_priors()

    def _seed_foundational_priors(self) -> None:
        """Foundational baseline seeds before personalized lived experience begins."""
        priors = {
            # Crisis & Disaster
            "fucked": {"crisis": 1.2, "fear": 0.8},
            "screwed": {"crisis": 1.1, "frustration": 0.6},
            "ruined": {"crisis": 1.0, "fear": 0.7},
            "doomed": {"crisis": 1.1, "fear": 0.8},
            "disaster": {"crisis": 1.0, "fear": 0.7},
            "emergency": {"crisis": 0.9, "urgency": 1.0},
            "crash": {"crisis": 0.8, "frustration": 0.6},
            # Fear & Anxiety
            "panic": {"fear": 1.2, "crisis": 0.7},
            "terrified": {"fear": 1.5, "crisis": 0.8},
            "scared": {"fear": 1.2},
            "anxious": {"fear": 0.9, "burnout": 0.4},
            "dread": {"fear": 1.0},
            # Burnout & Fatigue
            "exhausted": {"burnout": 1.3},
            "drained": {"burnout": 1.2},
            "tired": {"burnout": 1.0},
            "burnout": {"burnout": 1.4},
            "fatigue": {"burnout": 1.1},
            # Imposter Syndrome
            "dumb": {"imposter_syndrome": 1.2},
            "stupid": {"imposter_syndrome": 0.9, "frustration": 0.7},
            "imposter": {"imposter_syndrome": 1.4},
            "unqualified": {"imposter_syndrome": 1.2},
            "hopeless": {"imposter_syndrome": 1.1, "burnout": 0.7},
            "giving up": {"imposter_syndrome": 1.0, "burnout": 0.8},
            # Confusion & Overload
            "confused": {"confusion": 1.2},
            "lost": {"confusion": 1.0, "fear": 0.4},
            "overwhelmed": {"confusion": 0.9, "burnout": 0.7},
            "baffled": {"confusion": 1.1},
            "puzzled": {"confusion": 1.0},
            # Frustration & Annoyance
            "broken": {"frustration": 0.9, "crisis": 0.5},
            "annoying": {"frustration": 1.1},
            "frustrated": {"frustration": 1.3},
            "angry": {"frustration": 1.2},
            "wtf": {"frustration": 1.2},
            "dammit": {"frustration": 1.0},
            # Joy, Triumph & Milestone
            "awesome": {"joy": 1.2},
            "fixed": {"joy": 1.1},
            "green": {"joy": 0.9},
            "passed": {"joy": 1.0},
            "milestone": {"joy": 1.1},
            "yay": {"joy": 1.2},
            "hooray": {"joy": 1.3},
            "celebrate": {"joy": 1.2},
            # Urgency
            "hurry": {"urgency": 1.2},
            "urgent": {"urgency": 1.3},
            "asap": {"urgency": 1.2},
            "rush": {"urgency": 1.0},
        }
        for token, em_dict in priors.items():
            self.synaptic_weights[token] = dict(em_dict)

    def tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"\biam\b", "i am", text.lower())
        tokens = re.findall(r"[a-z0-9_']+", cleaned)
        return tokens

    def infer_emotion(self, text: str) -> Tuple[str, float, float]:
        """
        Infers emotional state via plastic synaptic activation forward pass.
        Returns: (primary_emotion, confidence, raw_valence)
        """
        tokens = self.tokenize(text)
        if not tokens:
            return "neutral", 0.70, 0.05

        scores: Dict[str, float] = {e: 0.05 for e in self.PRIMARY_EMOTIONS}

        for tok in tokens:
            if tok in self.synaptic_weights:
                for em, w in self.synaptic_weights[tok].items():
                    scores[em] = scores.get(em, 0.0) + w

        # Find maximum firing emotion
        best_emotion = max(scores, key=lambda k: scores[k])
        total_activation = sum(scores.values()) + 1e-6
        confidence = round(scores[best_emotion] / total_activation, 2)

        if scores[best_emotion] <= 0.15:
            best_emotion = "neutral"
            confidence = 0.65

        # Calculate experiential valence
        valence_map = {
            "joy": +0.85, "neutral": +0.05, "urgency": -0.20,
            "confusion": -0.35, "burnout": -0.65, "imposter_syndrome": -0.75,
            "frustration": -0.75, "fear": -0.80, "crisis": -0.90
        }
        valence = valence_map.get(best_emotion, 0.0)

        return best_emotion, max(0.50, min(0.99, confidence)), valence

    def learn_from_experience(
        self,
        text: str,
        experienced_emotion: str,
        user_confirmed: bool = True,
        user_reward: float = 1.0
    ) -> Dict[str, Any]:
        """
        Hebbian Synaptic Plasticity Update:
        Updates associative weights between tokens and emotional state based on lived experience.
        """
        tokens = self.tokenize(text)
        if not tokens or experienced_emotion not in self.PRIMARY_EMOTIONS:
            return {"updated": False}

        updated_tokens = []
        for tok in tokens:
            if len(tok) < 2:
                continue
            if tok not in self.synaptic_weights:
                self.synaptic_weights[tok] = {}

            current_weight = self.synaptic_weights[tok].get(experienced_emotion, 0.0)
            if user_confirmed:
                # Hebbian reinforcement: strengthening the association
                delta = self.learning_rate * user_reward
                self.synaptic_weights[tok][experienced_emotion] = round(current_weight + delta, 3)
            else:
                # Depression / penalty: weakening false association
                delta = -self.learning_rate * 0.8
                self.synaptic_weights[tok][experienced_emotion] = max(0.0, round(current_weight + delta, 3))
            updated_tokens.append(tok)

        # Trigger trust adjustment
        if user_confirmed and user_reward > 0:
            self.trust.adjust_user_trust(0.02 * user_reward)

        self.save_synapses()
        return {
            "updated": True,
            "experienced_emotion": experienced_emotion,
            "tokens_adapted": updated_tokens,
            "user_trust": round(self.trust.user_trust, 3)
        }

    def correct_emotion_synapse(self, text: str, wrong_emotion: str, correct_emotion: str) -> None:
        """Online delta correction when user teaches Aira their true state."""
        self.learn_from_experience(text, wrong_emotion, user_confirmed=False)
        self.learn_from_experience(text, correct_emotion, user_confirmed=True, user_reward=1.5)
        # Experiencing user honesty increases mutual trust
        self.trust.adjust_user_trust(0.01)

    def save_synapses(self) -> None:
        """Serializes learned weights and trust profile to local run and Obsidian vault."""
        data = {
            "saved_at": time.time(),
            "trust_profile": asdict(self.trust),
            "cognitive_drives": asdict(self.drives),
            "synaptic_weights_count": len(self.synaptic_weights),
            "synaptic_weights": self.synaptic_weights,
        }
        for dest in [self.store_path, self.vault_mirror]:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass

    def load_synapses(self) -> bool:
        """Restores synaptic weights from disk or vault mirror."""
        for src in [self.store_path, self.vault_mirror]:
            if src.exists():
                try:
                    data = json.loads(src.read_text(encoding="utf-8"))
                    self.synaptic_weights = data.get("synaptic_weights", {})
                    t_data = data.get("trust_profile", {})
                    for k, v in t_data.items():
                        if hasattr(self.trust, k):
                            setattr(self.trust, k, v)
                    d_data = data.get("cognitive_drives", {})
                    for k, v in d_data.items():
                        if hasattr(self.drives, k):
                            setattr(self.drives, k, v)
                    return True
                except Exception:
                    continue
        return False
