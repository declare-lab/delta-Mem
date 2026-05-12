from __future__ import annotations

import re

_SENTENCE_BOUNDARY_RE = re.compile(
    r"\n{2,}|(?<=[.!?;:])(?:[\"')\]\u201d\u2019]+)?\s+"
)


def split_text_into_sentence_chunks(text: str) -> list[str]:
    if text == "":
        return [""]

    chunks: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end()
        if end <= start:
            continue
        chunk = text[start:end]
        if chunk:
            chunks.append(chunk)
        start = end
    if start < len(text):
        chunks.append(text[start:])
    if not chunks:
        return [text]
    return chunks


def split_text_into_sentence_token_chunks(text: str) -> list[str]:
    if text == "":
        return [""]

    chunks: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        boundary = match.start()
        if boundary > start:
            chunks.append(text[start:boundary])
        start = boundary
    if start < len(text):
        chunks.append(text[start:])
    if not chunks:
        return [text]
    return chunks


__all__ = ["split_text_into_sentence_chunks", "split_text_into_sentence_token_chunks"]
