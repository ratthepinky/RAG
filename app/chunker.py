from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.config import BuildConfig, load_config


CAPTION_LINE_RE = re.compile(r"^\s*(fig(?:ure)?|table)\s*[\dA-Za-z.\-:()]+", re.IGNORECASE)
LOW_VALUE_LINE_RE = re.compile(
    r"(supplementary information|peer review file|catalog(?:ue)?\s*(?:no\.?|number)?|lot\s*(?:no\.?|number)?|version\s*\d|\bsku\b|\breagent\b|\bmanufacturer\b)",
    re.IGNORECASE,
)
INLINE_NOISE_RE = re.compile(
    r"\b(?:supplementary information|peer review file)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?\.])\s+")
NON_WORD_RE = re.compile(r"\W+", re.UNICODE)


def _ensure_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(item for item in value if isinstance(item, str) and item.strip())
    return ""


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalized_signature(text: str) -> str:
    return NON_WORD_RE.sub("", text.lower())


def _is_redundant_paragraph(signature: str, recent_signatures: list[str], prefix_set: set[str]) -> bool:
    if not signature:
        return True
    prefix = signature[:120]
    if prefix and prefix in prefix_set:
        return True
    for previous in recent_signatures[-10:]:
        if prefix and previous.startswith(prefix):
            return True
        previous_prefix = previous[:120]
        if previous_prefix and signature.startswith(previous_prefix):
            return True
        if abs(len(previous) - len(signature)) <= 40:
            if SequenceMatcher(None, previous[:220], signature[:220]).ratio() >= 0.97:
                return True
    return False


def _split_into_paragraphs(text: str, chunk_size: int) -> list[str]:
    blocks = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if len(blocks) > 1:
        return blocks
    single = text.strip()
    if len(single) <= chunk_size:
        return [single] if single else []
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(single) if part.strip()]
    return sentences if sentences else [single]


def _clean_text(text: str, config: BuildConfig) -> str:
    text = _normalize_whitespace(text)
    if not text:
        return ""

    cleaned_parts: list[str] = []
    exact_seen: set[str] = set()
    prefix_seen: set[str] = set()
    recent_signatures: list[str] = []

    for part in _split_into_paragraphs(text, config.chunk_size):
        paragraph = _normalize_whitespace(INLINE_NOISE_RE.sub(" ", part))
        if len(paragraph) < config.min_paragraph_len:
            continue
        if CAPTION_LINE_RE.match(paragraph):
            continue
        if LOW_VALUE_LINE_RE.search(paragraph):
            continue

        signature = _normalized_signature(paragraph)
        if signature in exact_seen:
            continue
        if _is_redundant_paragraph(signature, recent_signatures, prefix_seen):
            continue

        cleaned_parts.append(paragraph)
        exact_seen.add(signature)
        if signature[:120]:
            prefix_seen.add(signature[:120])
        recent_signatures.append(signature)

    return "\n\n".join(cleaned_parts).strip()


def _truncate_for_sections(text: str, config: BuildConfig) -> str:
    max_chars = config.chunk_size * config.max_section_splits
    return text[:max_chars].strip()


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]
    sentences = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(paragraph) if piece.strip()]
    if len(sentences) <= 1:
        return [paragraph[index : index + chunk_size].strip() for index in range(0, len(paragraph), chunk_size)]

    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if current and len(candidate) > chunk_size:
            pieces.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current.strip():
        pieces.append(current.strip())
    return pieces


def _split_cleaned_text(text: str, chunk_size: int, max_splits: int) -> list[str]:
    if not text:
        return []
    paragraphs = _split_into_paragraphs(text, chunk_size)
    pieces: list[str] = []
    for paragraph in paragraphs:
        pieces.extend(_split_long_paragraph(paragraph, chunk_size))
    if not pieces:
        return []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    for piece in pieces:
        separator = 2 if current_parts else 0
        if chunks and len(chunks) >= max_splits - 1:
            current_parts.append(piece)
            continue
        if current_parts and current_length + separator + len(piece) > chunk_size:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = [piece]
            current_length = len(piece)
        else:
            current_parts.append(piece)
            current_length += separator + len(piece)

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())
    return [chunk for chunk in chunks[:max_splits] if chunk]


def _find_offsets(source_text: str, chunk_texts: list[str]) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for chunk_text in chunk_texts:
        start = source_text.find(chunk_text, cursor)
        if start < 0:
            start = cursor
        end = start + len(chunk_text)
        offsets.append((start, end))
        cursor = end
    return offsets


def _extract_section_entries(doc: dict[str, Any]) -> list[tuple[str, str]]:
    entries = doc.get("section_texts") or []
    results: list[tuple[str, str]] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            text = entry.strip()
            if text:
                results.append((f"section_{index}", text))
        elif isinstance(entry, dict):
            section_name = _ensure_text(
                entry.get("section_name") or entry.get("heading") or entry.get("title") or f"section_{index}"
            )
            text = _ensure_text(entry.get("text") or entry.get("content") or entry.get("body"))
            if text.strip():
                results.append((section_name.strip() or f"section_{index}", text.strip()))
    return results


def _build_summary_text(doc: dict[str, Any], config: BuildConfig) -> str:
    title = _ensure_text(doc.get("title"))
    abstract = _ensure_text(doc.get("abstract"))
    keywords = _ensure_text(doc.get("keywords"))
    pieces: list[str] = []
    if title:
        pieces.append(title)
    if abstract:
        pieces.append(abstract)
    if keywords:
        pieces.append(f"Keywords: {keywords}")

    summary = _clean_text("\n\n".join(pieces), config)
    if summary:
        return summary[: config.chunk_size].strip()

    body_text = _clean_text(_ensure_text(doc.get("body_text")), config)
    if not body_text:
        return ""
    fallback = body_text.split("\n\n", 1)[0]
    return fallback[: config.chunk_size].strip()


def _build_chunk_record(
    doc: dict[str, Any],
    chunk_index: int,
    chunk_type: str,
    section_name: str,
    chunk_text: str,
    char_start: int,
    char_end: int,
) -> dict[str, Any]:
    doc_id = _ensure_text(doc.get("doc_id"))
    if not doc_id:
        basis = _ensure_text(doc.get("title")) + _ensure_text(doc.get("body_text"))
        doc_id = hashlib.sha1(basis.encode("utf-8")).hexdigest()
    signature = _normalized_signature(chunk_text)
    chunk_id = hashlib.sha1(
        f"{doc_id}|{chunk_type}|{section_name}|{chunk_index}|{signature}".encode("utf-8")
    ).hexdigest()
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "chunk_index": chunk_index,
        "chunk_type": chunk_type,
        "section_name": section_name,
        "chunk_text": chunk_text,
        "char_start": char_start,
        "char_end": char_end,
        "title": _ensure_text(doc.get("title")),
        "journal": _ensure_text(doc.get("journal")),
        "year": _ensure_text(doc.get("year")),
        "source_type": _ensure_text(doc.get("source_type")),
    }


def build_chunks_for_document(doc: dict[str, Any], source_path: str = "", config: BuildConfig | None = None) -> list[dict[str, Any]]:
    del source_path
    runtime_config = config or load_config()
    chunks: list[dict[str, Any]] = []

    summary_text = _build_summary_text(doc, runtime_config)
    if summary_text:
        chunks.append(_build_chunk_record(doc, 0, "summary", "summary", summary_text, 0, len(summary_text)))

    next_index = len(chunks)
    remaining_slots = runtime_config.max_chunks_per_doc - next_index
    for section_name, raw_section in _extract_section_entries(doc):
        if remaining_slots <= 0:
            break
        cleaned = _clean_text(raw_section, runtime_config)
        if not cleaned:
            continue
        cleaned = _truncate_for_sections(cleaned, runtime_config)
        max_splits = min(runtime_config.max_section_splits, remaining_slots)
        section_chunks = _split_cleaned_text(cleaned, runtime_config.chunk_size, max_splits)
        offsets = _find_offsets(cleaned, section_chunks)
        for text, (start, end) in zip(section_chunks, offsets):
            chunks.append(_build_chunk_record(doc, next_index, "section", section_name, text, start, end))
            next_index += 1
            remaining_slots -= 1
            if remaining_slots <= 0:
                break

    if remaining_slots > 0:
        cleaned_body = _clean_text(_ensure_text(doc.get("body_text")), runtime_config)
        if cleaned_body:
            needed = min(3, remaining_slots)
            body_chunks = _split_cleaned_text(cleaned_body, runtime_config.chunk_size, max(needed, 1))
            offsets = _find_offsets(cleaned_body, body_chunks)
            for text, (start, end) in zip(body_chunks[:needed], offsets[:needed]):
                chunks.append(_build_chunk_record(doc, next_index, "body_fallback", "body", text, start, end))
                next_index += 1
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break

    return chunks[: runtime_config.max_chunks_per_doc]


def build_chunks_for_corpus(files: list[str], config: BuildConfig | None = None) -> list[dict[str, Any]]:
    runtime_config = config or load_config()
    records: list[dict[str, Any]] = []
    for file_name in files:
        file_path = Path(file_name)
        with file_path.open("r", encoding="utf-8") as handle:
            doc = json.load(handle)
        records.extend(build_chunks_for_document(doc, str(file_path), runtime_config))
    return records
