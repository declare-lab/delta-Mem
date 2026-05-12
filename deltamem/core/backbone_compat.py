from __future__ import annotations

from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3Attention,
    apply_rotary_pos_emb as smollm3_apply_rotary_pos_emb,
    eager_attention_forward as smollm3_eager_attention_forward,
)

HAS_SMOLLM3 = True


def ensure_attention_compat_views(module):
    return module
