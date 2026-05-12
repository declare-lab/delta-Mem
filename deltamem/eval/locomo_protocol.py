from __future__ import annotations

import hashlib
import random
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass

try:
    import regex as re_lib
except ImportError:  # pragma: no cover
    import re as re_lib  # type: ignore[no-redef]

try:
    from nltk.stem import PorterStemmer
except ImportError:  # pragma: no cover
    PorterStemmer = None


CATEGORY_NAMES = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

ALL_LOCOMO_MECHANISMS = (
    "full_history_replay",
)

DEFAULT_LOCOMO_MECHANISMS = (
    "full_history_replay",
)

OFFICIAL_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant whose job is to understand "
    "the following conversation and answer questions based on the conversation. "
    "If you don't know the answer to a question, please don't share false information."
)

OFFICIAL_QA_PROMPT = (
    "Based on the above conversations, write a short answer for the following question "
    "in a few words. Do not write complete and lengthy sentences. "
    "Answer with exact words from the conversations whenever possible.\n\n"
    "Question: {}"
)

OFFICIAL_CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. "
    "The conversation takes place over multiple days and the date of each conversation "
    "is wriiten at the beginning of the conversation.\n\n"
)

OFFICIAL_ANSWER_RESERVE_TOKENS = 50
OFFICIAL_MAX_NEW_TOKENS = 50
OFFICIAL_TOP_K = 10
OFFICIAL_TOP_P = 0.9
OFFICIAL_TEMPERATURE = 0.4

_STEMMER = PorterStemmer() if PorterStemmer is not None else None


@dataclass(frozen=True)
class LoCoMoQuestionSpec:
    prompt_text: str
    category: int
    option_answers: dict[str, str] | None = None


def normalize_answer(text: str) -> str:
    text = text.replace(",", "")

    def remove_articles(value: str) -> str:
        return re_lib.sub(r"\b(a|an|the|and)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    def lower(value: str) -> str:
        return value.lower()

    normalized = unicodedata.normalize("NFD", text)
    return white_space_fix(remove_articles(remove_punc(lower(normalized))))


def _stem_tokens(text: str) -> list[str]:
    tokens = normalize_answer(text).split()
    if _STEMMER is None:
        return tokens
    return [_STEMMER.stem(token) for token in tokens]


def single_answer_f1(prediction: str, ground_truth: str) -> float:
    prediction_tokens = _stem_tokens(prediction)
    ground_truth_tokens = _stem_tokens(ground_truth)
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def multi_answer_f1(prediction: str, ground_truth: str) -> float:
    predictions = [item.strip() for item in prediction.split(",") if item.strip()]
    answers = [item.strip() for item in ground_truth.split(",") if item.strip()]
    if not predictions or not answers:
        return 0.0
    return sum(
        max(single_answer_f1(candidate, answer) for candidate in predictions)
        for answer in answers
    ) / len(answers)


def score_locomo_prediction(question: dict, prediction: str) -> float:
    category = int(question["category"])
    prediction = prediction.strip()
    if category == 5:
        normalized = prediction.lower()
        return 1.0 if ("no information available" in normalized or "not mentioned" in normalized) else 0.0

    answer = str(question["answer"])
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        return multi_answer_f1(prediction, answer)
    return single_answer_f1(prediction, answer)


def summarize_locomo_records(
    records: list[dict[str, object]],
    *,
    condition_names: list[str],
) -> dict[str, object]:
    summary: dict[str, object] = {}
    for condition_name in condition_names:
        total_score = 0.0
        total_questions = 0
        category_scores: defaultdict[int, float] = defaultdict(float)
        category_counts: defaultdict[int, int] = defaultdict(int)
        for record in records:
            for qa in record["qa"]:
                if condition_name not in qa["conditions"]:
                    continue
                score = float(qa["conditions"][condition_name]["score"])
                category = int(qa["category"])
                total_score += score
                total_questions += 1
                category_scores[category] += score
                category_counts[category] += 1
        summary[condition_name] = {
            "overall_score": 0.0 if total_questions == 0 else round(total_score / total_questions, 4),
            "num_questions": total_questions,
            "category_scores": {
                str(category): {
                    "name": CATEGORY_NAMES.get(category, "unknown"),
                    "score": round(category_scores[category] / category_counts[category], 4),
                    "count": category_counts[category],
                }
                for category in sorted(category_counts)
            },
        }
    return summary


def _stable_seed(base_seed: int, sample_id: str, question_index: int, salt: str) -> int:
    payload = f"{base_seed}|{sample_id}|{question_index}|{salt}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def prepare_locomo_question(
    question: dict,
    *,
    sample_id: str,
    question_index: int,
    seed: int,
) -> LoCoMoQuestionSpec:
    category = int(question["category"])
    if category == 2:
        return LoCoMoQuestionSpec(
            prompt_text=question["question"] + " Use DATE of CONVERSATION to answer with an approximate date.",
            category=category,
        )
    if category == 5:
        distractor_answer = question.get("answer", question.get("adversarial_answer"))
        if distractor_answer is None:
            distractor_answer = "No information available"
        flip = (_stable_seed(seed, sample_id, question_index, "cat5") % 2) == 0
        if flip:
            option_answers = {"a": "No information available", "b": str(distractor_answer)}
        else:
            option_answers = {"a": str(distractor_answer), "b": "No information available"}
        prompt_text = (
            question["question"]
            + f" (a) {option_answers['a']} (b) {option_answers['b']}. "
            + "Select the correct answer by writing (a) or (b)."
        )
        return LoCoMoQuestionSpec(
            prompt_text=prompt_text,
            category=category,
            option_answers=option_answers,
        )
    return LoCoMoQuestionSpec(
        prompt_text=question["question"],
        category=category,
    )


def canonicalize_locomo_prediction(raw_prediction: str, spec: LoCoMoQuestionSpec) -> str:
    prediction = raw_prediction.replace('\\"', "'").strip()
    lines = [line.strip() for line in prediction.splitlines() if line.strip()]
    if lines:
        prediction = lines[0]
    lowered = prediction.lower()
    if spec.category == 5 and spec.option_answers is not None:
        if "(a)" in lowered or lowered == "a" or lowered.startswith("a)"):
            return spec.option_answers["a"]
        if "(b)" in lowered or lowered == "b" or lowered.startswith("b)"):
            return spec.option_answers["b"]
        if "no information available" in lowered or "not mentioned" in lowered:
            return "No information available"
    return (
        lowered.replace("(a)", "")
        .replace("(b)", "")
        .replace("a)", "")
        .replace("b)", "")
        .replace("answer:", "")
        .strip()
    )


def build_official_question_prompt(spec: LoCoMoQuestionSpec) -> str:
    return OFFICIAL_QA_PROMPT.format(spec.prompt_text)


def render_locomo_turn(dialog: dict) -> str:
    turn = f'{dialog["speaker"]} said, "{dialog["text"]}"\n'
    if dialog.get("blip_caption"):
        turn += f' and shared {dialog["blip_caption"]}.'
    turn += "\n"
    return turn


def build_session_message(conversation: dict, session_num: int) -> dict[str, str]:
    session_key = f"session_{session_num}"
    date_key = f"{session_key}_date_time"
    if session_key not in conversation or date_key not in conversation:
        raise KeyError(f"Missing LoCoMo session keys for session {session_num}")
    turns = "".join(render_locomo_turn(dialog) for dialog in conversation[session_key]).rstrip()
    return {
        "role": "user",
        "content": f"DATE: {conversation[date_key]}\nCONVERSATION:\n{turns}",
    }


def build_turn_message(conversation: dict, session_num: int, dialog: dict) -> dict[str, str]:
    session_key = f"session_{session_num}"
    date_key = f"{session_key}_date_time"
    if session_key not in conversation or date_key not in conversation:
        raise KeyError(f"Missing LoCoMo session keys for session {session_num}")
    return {
        "role": "user",
        "content": f"DATE: {conversation[date_key]}\nCONVERSATION:\n{render_locomo_turn(dialog).rstrip()}",
    }


def _build_locomo_session_message_groups(
    sample: dict,
    *,
    message_granularity: str = "session",
) -> list[list[dict[str, str]]]:
    if message_granularity not in {"session", "turn"}:
        raise ValueError(f"Unsupported LoCoMo history message granularity: {message_granularity}")

    conversation = sample["conversation"]
    session_groups: list[list[dict[str, str]]] = []
    session_nums = sorted(
        int(key.split("_")[-1])
        for key in conversation
        if key.startswith("session_") and not key.endswith("date_time")
    )
    for session_num in session_nums:
        session_key = f"session_{session_num}"
        session_turns = conversation[session_key]
        if not session_turns:
            continue
        if message_granularity == "session":
            session_groups.append([build_session_message(conversation, session_num)])
            continue
        session_groups.append(
            [
                build_turn_message(conversation, session_num, dialog)
                for dialog in session_turns
            ]
        )
    return session_groups


def build_locomo_history_messages(
    sample: dict,
    *,
    message_granularity: str = "session",
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT}
    ]
    for session_messages in _build_locomo_session_message_groups(
        sample,
        message_granularity=message_granularity,
    ):
        messages.extend(session_messages)
    return messages


def infer_model_context_window(model, tokenizer) -> int:
    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "sliding_window"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0 and value < 10**7:
            return value
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and value > 0 and value < 10**7:
        return value
    return 32768


def build_official_context_text(
    sample: dict,
    tokenizer,
    question_prompt: str,
    *,
    max_context_tokens: int,
    answer_reserve_tokens: int = OFFICIAL_ANSWER_RESERVE_TOKENS,
) -> str:
    conversation = sample["conversation"]
    speakers = [conversation["speaker_a"], conversation["speaker_b"]]
    start_prompt = OFFICIAL_CONV_START_PROMPT.format(*speakers)
    start_tokens = len(tokenizer.encode(start_prompt))
    question_tokens = len(tokenizer.encode(question_prompt))

    query_conv = ""
    total_tokens = 0
    stop = False
    session_nums = sorted(
        int(key.split("_")[-1])
        for key in conversation
        if key.startswith("session_") and not key.endswith("date_time")
    )
    for session_num in session_nums:
        session_key = f"session_{session_num}"
        date_key = f"{session_key}_date_time"
        if session_key not in conversation:
            continue
        for dialog in conversation[session_key][::-1]:
            turn = render_locomo_turn(dialog)
            new_tokens = len(
                tokenizer.encode(
                    f"DATE: {conversation[date_key]}\nCONVERSATION:\n{turn}"
                )
            )
            if (
                start_tokens + new_tokens + total_tokens + question_tokens
                < (max_context_tokens - answer_reserve_tokens)
            ):
                query_conv = turn + query_conv
                total_tokens += len(tokenizer.encode(turn))
            else:
                stop = True
                break
        query_conv = f"\nDATE: {conversation[date_key]}\nCONVERSATION:\n" + query_conv
        if stop:
            break
    return start_prompt + query_conv


def build_official_full_history_messages(
    sample: dict,
    tokenizer,
    question_spec: LoCoMoQuestionSpec,
    *,
    max_context_tokens: int,
    answer_reserve_tokens: int = OFFICIAL_ANSWER_RESERVE_TOKENS,
) -> list[dict[str, str]]:
    question_prompt = build_official_question_prompt(question_spec)
    context_text = build_official_context_text(
        sample,
        tokenizer,
        question_prompt,
        max_context_tokens=max_context_tokens,
        answer_reserve_tokens=answer_reserve_tokens,
    )
    return [
        {"role": "system", "content": OFFICIAL_SYSTEM_PROMPT},
        {"role": "user", "content": context_text + "\n\n" + question_prompt},
    ]
