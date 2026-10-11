from __future__ import annotations

import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from tokenizers import Tokenizer

from airapix.agent.neuromodulator import Neuromodulator, NeurochemicalState
from airapix.agent.theory_of_mind import TheoryOfMindEngine
from airapix.agent.memory_vault import MemoryVaultEngine
from airapix.agent.cognitive_loop import AiraCognitiveAgent
from airapix.agent.dream_engine import AiraDreamEngine
from airapix.training.continual import ContinualReplayMixture, EWCConsolidator
from airapix.model.config import AiraConfig


def test_neuromodulator_dynamics():
    nm = Neuromodulator()
    initial_dopamine = nm.state.dopamine

    # Task success -> dopamine hit
    res = nm.on_task_success(complexity=1.5)
    assert res["event"] == "dopamine_hit"
    assert nm.state.dopamine > initial_dopamine

    # Task failure -> cortisol and noradrenaline spike
    res_fail = nm.on_task_failure(severity=1.0)
    assert res_fail["event"] == "anomaly_alert"
    assert nm.state.noradrenaline > 0.20

    # User emotion perceived (panic) -> increases vigilance and oxytocin
    ox_before = nm.state.oxytocin
    nm.on_user_emotion_perceived("fear", intensity=0.8)
    assert nm.state.oxytocin > ox_before

    # Somatic markers
    assert nm.somatic_markers.evaluate_action("destructive_overwrite") < 0.0
    assert nm.somatic_markers.evaluate_action("safe_read") > 0.0


def test_theory_of_mind():
    tom = TheoryOfMindEngine()

    state_fear = tom.analyze("Our database is down in prod, I am terrified of breaking it worse!")
    assert state_fear.primary_emotion in {"fear", "crisis"}
    assert state_fear.valence < -0.5
    assert state_fear.arousal > 0.7

    state_joy = tom.analyze("Step 500 completed green and loss dropped to 2.8, we fixed it!")
    assert state_joy.primary_emotion == "joy"
    assert state_joy.valence > 0.7

    state_burnout = tom.analyze("I spent 10 hours on this bug, I feel dumb and exhausted.")
    assert state_burnout.primary_emotion in {"burnout", "imposter_syndrome"}

    guidance = tom.generate_empathy_prompt(state_fear)
    assert ("FEAR" in guidance or "CRISIS" in guidance)
    assert "Theory of Mind" in guidance


def test_memory_vault_indexing():
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault = Path(tmp_dir)
        (vault / "note1.md").write_text("# PyTorch CUDA VRAM\nPyTorch allocates memory in chunks and caches it.\n[dopamine_verified: True]", encoding="utf-8")
        (vault / "note2.md").write_text("# Python Decorators\nDecorators wrap functions and alter their behavior.", encoding="utf-8")

        engine = MemoryVaultEngine(vault)
        assert len(engine.notes) == 2

        # Search for CUDA
        results = engine.search("CUDA memory allocator", top_k=2)
        assert len(results) >= 1
        assert results[0].title == "note1"
        assert results[0].somatic_valence > 0.8  # Dopamine verified boost

        # Record episodic memory
        new_path = engine.record_episodic_memory("Bug Solved", "Fixed scan bug", "Scan now completes without overflow.")
        assert Path(new_path).exists()
        assert len(engine.notes) == 3


def test_aira_cognitive_agent_full_loop():
    with tempfile.TemporaryDirectory() as tmp_dir:
        agent = AiraCognitiveAgent(vault_path=tmp_dir)
        
        turn = agent.process_turn("I am terrified our production service crashed!")
        assert turn.user_emotion.primary_emotion == "fear"
        assert turn.cognitive_mode in {"deliberate", "protective_grounding"}
        assert "<think>" in turn.thinking_prompt

        # Reward success
        reward_res = agent.reward_task_success("Crash Isolated", "Pinpointed dead connection pool", complexity=1.2)
        assert reward_res["reward"]["event"] == "dopamine_hit"
        assert reward_res["new_state"]["dopamine"] > 0.5


def test_24k_tokenizer_and_config():
    cfg = AiraConfig()
    assert cfg.vocab_size == 24000

    tok = Tokenizer.from_file("airapix/model/tokenizer/tokenizer.json")
    assert tok.get_vocab_size() == 24000
    assert tok.token_to_id("<think>") is not None
    assert tok.token_to_id("</think>") is not None


def test_plastic_synapses_and_trust_architecture():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = Path(tmp_dir) / "synapses.json"
        mirror = Path(tmp_dir) / "mirror.json"
        from airapix.agent.synapses import AffectiveSynapseEngine, TrustProfile

        engine = AffectiveSynapseEngine(store_path=store, vault_mirror=mirror)

        # 1. Verify Trust Profile defaults and constraints
        assert engine.trust.user_trust == 0.50
        assert engine.trust.ai_model_trust == 0.20

        # User trust grows towards 100%
        engine.trust.adjust_user_trust(0.10)
        assert engine.trust.user_trust > 0.50

        # Peer AI trust is strictly capped at 60%
        for _ in range(50):
            engine.trust.audit_external_ai_claim(claim_valid=True)
        assert engine.trust.ai_model_trust <= 0.60
        assert engine.trust.ai_model_trust > 0.40

        # 2. Hebbian learning from experience
        phrase = "iam sooo fucked"
        em, conf, val = engine.infer_emotion(phrase)
        assert em == "crisis"
        assert val < -0.7

        # User teaches Aira a new custom word
        custom_slang = "my gpu is totally kaput"
        engine.learn_from_experience(custom_slang, "crisis", user_confirmed=True)
        em_learned, _, _ = engine.infer_emotion("kaput")
        assert em_learned == "crisis"

        # Persistence: reload from disk
        engine_reloaded = AffectiveSynapseEngine(store_path=store)
        assert "kaput" in engine_reloaded.synaptic_weights
        assert engine_reloaded.synaptic_weights["kaput"]["crisis"] > 0.0


def test_comprehensive_emotional_dimensions():
    with tempfile.TemporaryDirectory() as tmp_dir:
        store = Path(tmp_dir) / "synapses.json"
        mirror = Path(tmp_dir) / "mirror.json"
        from airapix.agent.synapses import AffectiveSynapseEngine

        engine = AffectiveSynapseEngine(store_path=store, vault_mirror=mirror)

        # 1. Burnout & Cognitive Exhaustion
        em, _, val = engine.infer_emotion("I have been staring at this CUDA trace for 14 hours and am completely drained")
        assert em == "burnout"
        assert val < -0.5

        # 2. Imposter Syndrome & Self-Deprecation
        em, _, val = engine.infer_emotion("I feel so dumb and unqualified, maybe I should just give up")
        assert em == "imposter_syndrome"
        assert val < -0.6

        # 3. Frustration & Annoyance
        em, _, val = engine.infer_emotion("This compiler error is so damn annoying, wtf happened")
        assert em == "frustration"
        assert val < -0.6

        # 4. Confusion & Cognitive Overload
        em, _, val = engine.infer_emotion("I am completely lost and confused by this dense math paper")
        assert em == "confusion"
        assert val < -0.2

        # 5. Joy, Triumph & Milestone Celebration
        em, _, val = engine.infer_emotion("Step 5000 green! We fixed the scan bug and loss dropped, awesome milestone!")
        assert em == "joy"
        assert val > 0.7

        # 6. Urgency & Emergency
        em, _, val = engine.infer_emotion("Hurry, we need to rollback this deployment asap, it is urgent!")
        assert em == "urgency"
        assert val < 0.0


def test_persistent_cross_chat_memory():
    with tempfile.TemporaryDirectory() as tmp_dir:
        agent = AiraCognitiveAgent(vault_path=tmp_dir)

        # 1. Start Session
        agent.start_session(topic="Architecture Planning")
        assert agent.session_topic == "Architecture Planning"

        # 2. Turn 1 with heuristic note extraction
        u1 = "Remember that we must always use 24k tokenizer vocab size."
        t1 = agent.process_turn(u1)
        r1 = "Understood, 24k vocabulary is locked in for all training runs."
        agent.log_turn(user_text=u1, aira_response=r1, turn_output=t1)
        assert len(agent.session_notes) >= 1
        assert "24k" in agent.session_notes[0]

        # 3. Turn 2 with explicit decision extraction
        u2 = "Decision: we will use Hebbian synaptic learning for experiential adaptation."
        t2 = agent.process_turn(u2)
        r2 = "Hebbian online learning matrix is fully active."
        agent.log_turn(user_text=u2, aira_response=r2, turn_output=t2)
        assert len(agent.session_decisions) >= 1
        assert "Hebbian" in agent.session_decisions[0]

        # 4. Commit session to Vault
        saved_file = agent.end_session()
        assert saved_file is not None
        assert Path(saved_file).exists()

        content = Path(saved_file).read_text(encoding="utf-8")
        assert "project: aira-chat" in content
        assert "type: log" in content
        assert "Architecture Planning" in content
        assert "24k tokenizer" in content
        assert "Hebbian" in content

        # 5. Verify Daily Note link was created
        daily_notes = list((Path(tmp_dir) / "01 - Daily Notes").rglob("*.md"))
        assert len(daily_notes) >= 1
        daily_text = daily_notes[0].read_text(encoding="utf-8")
        assert "[[Conversations/" in daily_text

        # 6. Cross-chat recall in a brand new session
        agent.start_session(topic="Followup Session")
        u3 = "Do you remember our last chat about Hebbian learning?"
        t3 = agent.process_turn(u3)
        assert "[Recalled Cross-Chat Past Sessions from Vault]" in t3.thinking_prompt
        assert "Hebbian" in t3.thinking_prompt

        # 7. Check session indexing list
        recent = agent.memory_vault.get_chat_sessions(limit=5)
        assert len(recent) >= 1
        assert "Architecture Planning" in recent[0]["title"]


def test_autonomous_dream_cycle():
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault = Path(tmp_dir) / "vault"
        vault.mkdir()
        (vault / "note.md").write_text("# PyTorch Autograd\nAutograd records computation graph dynamically.", encoding="utf-8")

        dream_eng = AiraDreamEngine(vault_path=vault)
        dream_eng.DEFAULT_DREAM_BUFFER = Path(tmp_dir) / "dream_buffer.jsonl"
        dream_eng.PROCESSED_DREAM_DATA = Path(tmp_dir) / "dream_processed.jsonl"

        res = dream_eng.run_dream_cycle(num_dreams=3, merge_into_processed=False)
        assert res["status"] == "consolidation_complete"
        assert res["dreams_synthesized"] >= 1
        assert res["average_somatic_valence"] >= 0.60
        assert dream_eng.DEFAULT_DREAM_BUFFER.exists()

        content = dream_eng.DEFAULT_DREAM_BUFFER.read_text(encoding="utf-8")
        assert "<think>" in content
        assert "</think>" in content
        assert "reasoning" in content


def test_continual_replay_and_ewc():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        # 1. Test ContinualReplayMixture
        base_f = Path(tmp_dir) / "base.jsonl"
        dyn_f = Path(tmp_dir) / "dyn.jsonl"

        base_f.write_text('{"input": "def add(a, b):", "output": "return a + b", "category": "code"}\n', encoding="utf-8")
        dyn_f.write_text('{"input": "Explain EWC", "output": "<think>Analysis</think>EWC protects weights.", "category": "reasoning"}\n', encoding="utf-8")

        from airapix.training.tokenizer import TokenizerWrapper
        tok = TokenizerWrapper("airapix/model/tokenizer/tokenizer.json")

        mixture = ContinualReplayMixture(
            base_pretrain_paths=[base_f],
            dynamic_experience_paths=[dyn_f],
            tokenizer=tok,
            context_len=64,
            replay_ratio=0.80
        )

        iterator = iter(mixture)
        samples = [next(iterator) for _ in range(5)]
        assert len(samples) == 5
        assert "input_ids" in samples[0]
        assert "labels" in samples[0]
        del iterator, mixture
        import gc
        gc.collect()

        # 2. Test EWCConsolidator
        dummy_model = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        ewc = EWCConsolidator(dummy_model, lambda_reg=200.0)
        ewc.register_reference_weights()
        assert len(ewc.reference_params) > 0

        # Create dummy batch for Fisher calculation
        dummy_inputs = torch.randn(4, 16)
        dummy_targets = torch.randint(0, 8, (4,))
        loss_fn = nn.CrossEntropyLoss()

        dummy_model.zero_grad()
        out = dummy_model(dummy_inputs)
        loss = loss_fn(out, dummy_targets)
        loss.backward()

        # Manually populate fisher diagonal from gradients for unit testing
        ewc.fisher_diagonal = {
            name: param.grad.data.pow(2)
            for name, param in dummy_model.named_parameters()
            if param.grad is not None
        }
        ewc.is_initialized = True

        # Before weight modification: EWC penalty is 0.0
        penalty_before = ewc.compute_ewc_loss()
        assert penalty_before.item() == 0.0

        # After weight modification: EWC penalty increases quadratically
        with torch.no_grad():
            dummy_model[0].weight.add_(0.5)

        penalty_after = ewc.compute_ewc_loss()
        assert penalty_after.item() > 0.0
