from __future__ import annotations

import importlib
import re
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

_SCRIPT_PATTERNS = {
    "ar": regex.compile(r"\p{Script=Arabic}"),
    "bn": regex.compile(r"\p{Script=Bengali}"),
    "gu": regex.compile(r"\p{Script=Gujarati}"),
    "hi": regex.compile(r"\p{Script=Devanagari}"),
    "kn": regex.compile(r"\p{Script=Kannada}"),
    "ko": regex.compile(r"\p{Script=Hangul}"),
    "pa": regex.compile(r"\p{Script=Gurmukhi}"),
    "ru": regex.compile(r"\p{Script=Cyrillic}"),
    "ta": regex.compile(r"\p{Script=Tamil}"),
    "te": regex.compile(r"\p{Script=Telugu}"),
    "th": regex.compile(r"\p{Script=Thai}"),
}

_LATIN_LANGUAGE_STOPWORDS = {
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "mit", "für", "von", "zu"},
    "en": {"the", "and", "is", "are", "this", "that", "with", "for", "from", "not", "have", "you"},
    "fi": {"ja", "on", "ei", "se", "että", "kuin", "olen", "tämä", "mutta", "kun", "oli"},
    "it": {"il", "la", "di", "e", "che", "non", "un", "una", "per", "con", "sono", "del"},
    "pt": {"de", "que", "e", "o", "a", "os", "as", "um", "uma", "não", "para", "com"},
    "sw": {"na", "ya", "kwa", "ni", "katika", "kwamba", "kutoka", "kuwa", "hii", "lakini", "kama"},
    "vi": {"la", "va", "cua", "mot", "nhung", "cac", "cho", "khong", "duoc", "voi", "trong"},
}

_ARABIC_VARIANT_HINTS = {
    "fa": set("گچپژکۍیە"),
    "ur": set("ٹڈڑںھہے"),
}

_DEVANAGARI_HINTS = {
    "mr": {"आहे", "आणि", "नाही", "हे", "एक"},
    "ne": {"छ", "र", "को", "मा", "यो", "हो"},
    "hi": {"है", "और", "नहीं", "यह", "एक"},
}

_CYRILLIC_HINTS = {
    "bg": {"и", "в", "не", "се", "за", "че", "с", "на"},
    "ru": {"и", "в", "не", "что", "это", "на", "я", "с"},
}


class LangDetectException(Exception):
    pass


def _immutabledict(value):
    return dict(value)


class _RegexpTokenizer:
    def __init__(self, pattern: str) -> None:
        self._pattern = re.compile(pattern, flags=re.UNICODE)

    def tokenize(self, text: str) -> list[str]:
        return self._pattern.findall(text)


class _SentenceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        pieces = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text) if part.strip()]
        if pieces:
            return pieces
        text = text.strip()
        return [text] if text else []


class _NltkDataModule:
    def find(self, resource_name: str):
        return resource_name

    def load(self, resource_name: str):
        return _SentenceTokenizer()


def _install_immutabledict_shim() -> None:
    if "immutabledict" in sys.modules:
        return
    module = types.ModuleType("immutabledict")
    module.__dict__["immutabledict"] = _immutabledict
    sys.modules["immutabledict"] = module


def _install_nltk_shim() -> None:
    if "nltk" in sys.modules:
        return
    try:
        importlib.import_module("nltk")
        return
    except Exception:
        pass
    module = types.ModuleType("nltk")
    module.__dict__["data"] = _NltkDataModule()
    module.__dict__["download"] = lambda *args, **kwargs: True
    module.__dict__["tokenize"] = types.SimpleNamespace(RegexpTokenizer=_RegexpTokenizer)
    module.__dict__["word_tokenize"] = lambda text: _WORD_RE.findall(text)
    sys.modules["nltk"] = module


def _normalize_latin_for_detection(text: str) -> list[str]:
    ascii_text = (
        text.lower()
        .replace("á", "a")
        .replace("à", "a")
        .replace("â", "a")
        .replace("ã", "a")
        .replace("ä", "a")
        .replace("å", "a")
        .replace("ç", "c")
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("í", "i")
        .replace("ì", "i")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ñ", "n")
        .replace("ó", "o")
        .replace("ò", "o")
        .replace("ô", "o")
        .replace("õ", "o")
        .replace("ö", "o")
        .replace("ú", "u")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ü", "u")
        .replace("ý", "y")
        .replace("ÿ", "y")
        .replace("ă", "a")
        .replace("đ", "d")
        .replace("ê", "e")
        .replace("ô", "o")
        .replace("ơ", "o")
        .replace("ư", "u")
    )
    return _WORD_RE.findall(ascii_text)


def _latin_language_from_tokens(tokens: list[str]) -> str:
    if not tokens:
        return "en"
    scores: dict[str, int] = {language: 0 for language in _LATIN_LANGUAGE_STOPWORDS}
    for token in tokens:
        for language, stopwords in _LATIN_LANGUAGE_STOPWORDS.items():
            if token in stopwords:
                scores[language] += 1
    best_language, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_language
    return "en"


def _script_count(text: str, pattern: regex.Pattern) -> int:
    return len(pattern.findall(text))


def detect(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        raise LangDetectException("Cannot detect language from empty text")

    script_counts = {language: _script_count(normalized, pattern) for language, pattern in _SCRIPT_PATTERNS.items()}
    dominant_script, dominant_count = max(script_counts.items(), key=lambda item: item[1])
    if dominant_count > 0:
        if dominant_script == "ar":
            chars = set(normalized)
            if chars & _ARABIC_VARIANT_HINTS["ur"]:
                return "ur"
            if chars & _ARABIC_VARIANT_HINTS["fa"]:
                return "fa"
            return "ar"
        if dominant_script == "hi":
            for language, hints in _DEVANAGARI_HINTS.items():
                if any(hint in normalized for hint in hints):
                    return language
            return "hi"
        if dominant_script == "ru":
            tokens = set(_WORD_RE.findall(normalized.lower()))
            bg_score = sum(1 for token in tokens if token in _CYRILLIC_HINTS["bg"])
            ru_score = sum(1 for token in tokens if token in _CYRILLIC_HINTS["ru"])
            return "bg" if bg_score > ru_score else "ru"
        return dominant_script

    tokens = _normalize_latin_for_detection(normalized)
    return _latin_language_from_tokens(tokens)


def _install_langdetect_shim() -> None:
    if "langdetect" in sys.modules:
        return
    module = types.ModuleType("langdetect")
    module.__dict__["detect"] = detect
    module.__dict__["LangDetectException"] = LangDetectException
    sys.modules["langdetect"] = module


def _ensure_ifeval_deps() -> None:
    _install_immutabledict_shim()
    _install_nltk_shim()
    _install_langdetect_shim()


def _ensure_lm_eval_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    harness_root = repo_root.parent / "lm-evaluation-harness"
    if harness_root.is_dir():
        harness_root_str = str(harness_root)
        if harness_root_str not in sys.path:
            sys.path.insert(0, harness_root_str)


@lru_cache(maxsize=1)
def _ifeval_utils_module():
    _ensure_ifeval_deps()
    _ensure_lm_eval_on_path()
    return importlib.import_module("lm_eval.tasks.ifeval.utils")


@dataclass(frozen=True)
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[dict[str, object | None]]


def _build_input_example(item: dict) -> InputExample:
    instruction_ids = [str(value).strip() for value in item.get("instruction_id_list", []) if str(value).strip()]
    kwargs_list: list[dict[str, object | None]] = []
    for entry in item.get("kwargs", []) or []:
        if isinstance(entry, dict):
            kwargs_list.append(dict(entry))
        else:
            kwargs_list.append({})
    if len(kwargs_list) < len(instruction_ids):
        kwargs_list.extend({} for _ in range(len(instruction_ids) - len(kwargs_list)))
    elif len(kwargs_list) > len(instruction_ids):
        kwargs_list = kwargs_list[: len(instruction_ids)]
    return InputExample(
        key=int(item.get("key", 0)),
        instruction_id_list=instruction_ids,
        prompt=str(item.get("prompt", "")),
        kwargs=kwargs_list,
    )


def evaluate_ifeval_response(item: dict, response: str) -> dict[str, object]:
    utils = _ifeval_utils_module()
    inp = _build_input_example(item)
    strict = utils.test_instruction_following_strict(inp, response)
    loose = utils.test_instruction_following_loose(inp, response)
    return {
        "prompt_level_strict_acc": bool(strict.follow_all_instructions),
        "inst_level_strict_acc": [bool(flag) for flag in strict.follow_instruction_list],
        "prompt_level_loose_acc": bool(loose.follow_all_instructions),
        "inst_level_loose_acc": [bool(flag) for flag in loose.follow_instruction_list],
    }
