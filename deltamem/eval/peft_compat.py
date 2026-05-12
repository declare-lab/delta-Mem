from __future__ import annotations

from typing import Any

import transformers
from peft import peft_model as peft_model_module
from peft.utils import integrations as peft_integrations


_PATCHED = False
_ORIGINAL_MAP_CACHE_TO_LAYER_DEVICE_MAP = peft_integrations.map_cache_to_layer_device_map


def _patched_map_cache_to_layer_device_map(model, cache) -> None:
    if not (isinstance(cache, transformers.Cache) and hasattr(model, "hf_device_map")):
        return

    if isinstance(cache, transformers.EncoderDecoderCache):
        _patched_map_cache_to_layer_device_map(model, cache.self_attention_cache)
        return

    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        _ORIGINAL_MAP_CACHE_TO_LAYER_DEVICE_MAP(model, cache)
        return

    if not hasattr(cache, "layers"):
        return

    layer_device_map = peft_integrations.get_layer_device_map(model)
    num_hidden_layers = getattr(model.config, "num_hidden_layers", 0)
    for idx in range(min(num_hidden_layers, len(cache.layers))):
        layer_device = layer_device_map[idx]
        layer = cache.layers[idx]
        if hasattr(layer, "keys"):
            layer.keys = layer.keys.to(layer_device)
        if hasattr(layer, "values"):
            layer.values = layer.values.to(layer_device)


def patch_peft_cache_compat() -> None:
    global _PATCHED
    if _PATCHED:
        return
    peft_integrations.map_cache_to_layer_device_map = _patched_map_cache_to_layer_device_map
    peft_model_module.map_cache_to_layer_device_map = _patched_map_cache_to_layer_device_map
    _PATCHED = True
