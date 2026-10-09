import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from airapix.model.config import config_from_preset
from airapix.model.model import AiraForCausalLM, count_parameters

cfg = config_from_preset("tiny")
print("=== AiraPix v2 Smoke Test ===")
print(f"Config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
print(f"MoE: use_moe={cfg.use_moe}, experts={cfg.moe_num_experts}, top_k={cfg.moe_top_k}")
print(f"MTP heads: {cfg.num_mtp_heads}")
print()

m = AiraForCausalLM(cfg)
print(f"Total parameters: {count_parameters(m):,}")
print()

print("Layer schedule:")
for i, b in enumerate(m.blocks):
    print(f"  Layer {i}: {b.kind:>3s} | MoE={b.use_moe}")
print()

ids = torch.randint(0, cfg.vocab_size, (1, 32))
out = m(ids, labels=ids)
print(f"Loss: {out['loss'].item():.4f}")
print(f"Logits shape: {out['logits'].shape}")
print()
print("=== PASS ===")
