from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from deltamem.model_loading import DEFAULT_LOCAL_MODEL_PATH

_ALTERNATION_ERROR_TEXT = "Conversation roles must alternate user/assistant/user/assistant/..."


def _tokenizer_name(tokenizer: Any) -> str:
    for attr in ("name_or_path", "_name_or_path"):
        value = getattr(tokenizer, attr, None)
        if value:
            return str(value)
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        value = init_kwargs.get("name_or_path")
        if value:
            return str(value)
    return ""


def qwen3_enable_thinking_override(tokenizer: Any) -> bool | None:
    name = _tokenizer_name(tokenizer).lower()
    if "qwen3" not in name:
        return None
    if "thinking" in name:
        # Thinking-only models should keep their default behavior.
        return None
    if "instruct" in name:
        # Instruct-only models are already non-thinking.
        return None
    # Unified Qwen3 models default to thinking mode; disable it for this project.
    return False


def smollm3_enable_thinking_override(tokenizer: Any) -> bool | None:
    name = _tokenizer_name(tokenizer).lower()
    if "smollm3" not in name:
        return None
    # SmolLM3 chat templates default to /think; keep benchmark and training prompts
    # in direct-answer mode unless the caller explicitly overrides it.
    return False


@lru_cache(maxsize=4)
def _load_chat_template_from_tokenizer_config(model_path: str) -> str | None:
    config_path = Path(model_path) / "tokenizer_config.json"
    if not config_path.is_file():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    template = data.get("chat_template")
    return str(template) if template else None


def _restore_missing_chat_template(tokenizer: Any) -> None:
    if getattr(tokenizer, "chat_template", None):
        return
    name = _tokenizer_name(tokenizer).lower()
    candidate_paths: list[str] = []
    if name:
        candidate_paths.append(name)
    if "qwen3" in name:
        candidate_paths.append(DEFAULT_LOCAL_MODEL_PATH)
    for candidate in candidate_paths:
        template = _load_chat_template_from_tokenizer_config(candidate)
        if template:
            tokenizer.chat_template = template
            return


def _messages_violate_role_alternation(messages: list[dict[str, str]]) -> bool:
    previous_role: str | None = None
    for message in messages:
        role = str(message.get("role", ""))
        if role == "system":
            continue
        if previous_role == role:
            return True
        previous_role = role
    return False


@lru_cache(maxsize=8)
def _remove_alternation_guard(chat_template: str) -> str | None:
    if _ALTERNATION_ERROR_TEXT not in chat_template:
        return None
    pattern = re.compile(
        r"\{%-?\s*if\s*\(message\['role'\]\s*==\s*'user'\)\s*!=\s*\(loop\.index0\s*%\s*2\s*==\s*0\)\s*-?%\}"
        r"\s*\{\{\s*raise_exception\(\"Conversation roles must alternate user/assistant/user/assistant/\.\.\.\"\)\s*\}\}"
        r"\s*\{%-?\s*endif\s*-?%\}",
        flags=re.DOTALL,
    )
    sanitized, count = pattern.subn("", chat_template, count=1)
    return sanitized if count == 1 else None


def _apply_tokenizer_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None,
    **kwargs: Any,
):
    if enable_thinking is not None and "enable_thinking" not in kwargs:
        try:
            return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def apply_chat_template(tokenizer: Any, messages: list[dict[str, str]], /, **kwargs: Any):
    _restore_missing_chat_template(tokenizer)
    enable_thinking = qwen3_enable_thinking_override(tokenizer)
    if enable_thinking is None:
        enable_thinking = smollm3_enable_thinking_override(tokenizer)
    try:
        return _apply_tokenizer_chat_template(
            tokenizer,
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except Exception as exc:
        chat_template = getattr(tokenizer, "chat_template", None)
        if (
            not _messages_violate_role_alternation(messages)
            or _ALTERNATION_ERROR_TEXT not in str(exc)
            or not isinstance(chat_template, str)
        ):
            raise
        permissive_template = _remove_alternation_guard(chat_template)
        if permissive_template is None:
            raise
        original_template = tokenizer.chat_template
        tokenizer.chat_template = permissive_template
        try:
            return _apply_tokenizer_chat_template(
                tokenizer,
                messages,
                enable_thinking=enable_thinking,
                **kwargs,
            )
        finally:
            tokenizer.chat_template = original_template
