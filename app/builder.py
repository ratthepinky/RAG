from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.chunker import build_chunks_for_document
from app.config import BuildConfig, load_config
from app.embedder import encode_texts
from app.stores import (
    bulk_index_opensearch,
    ensure_opensearch_index,
    ensure_qdrant_collections,
    upsert_qdrant_points_multi,
)

LOGGER = logging.getLogger(__name__)
CHUNK_PROGRESS_EVERY = 20


def configure_logging(log_file: str | Path | None = None) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def _default_state_file(config: BuildConfig, start_offset: int, sample_size: int) -> Path:
    end_offset = start_offset + sample_size
    return config.artifacts_dir / "resume_state" / f"build_resume_{start_offset}_{end_offset}.state"


def _write_state_file(state_path: Path, next_offset: int, sample_size: int, doc_count: int, chunk_count: int) -> None:
    payload = {
        "current_offset": next_offset,
        "current_sample_size": sample_size,
        "last_success_docs": doc_count,
        "last_success_chunks": chunk_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_batch_documents(files: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    documents: list[tuple[Path, dict[str, Any]]] = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            documents.append((file_path, json.load(handle)))
    return documents


def build_index_batch(
    sample_size: int,
    start_offset: int,
    config: BuildConfig | None = None,
    state_file: str | Path | None = None,
) -> dict[str, Any]:
    runtime_config = config or load_config({"sample_size": sample_size, "start_offset": start_offset})
    batch_sample_size = int(sample_size if sample_size is not None else runtime_config.sample_size)
    batch_start_offset = int(start_offset if start_offset is not None else runtime_config.start_offset)
    state_path = Path(state_file) if state_file else _default_state_file(runtime_config, batch_start_offset, batch_sample_size)
    files = sorted(runtime_config.cleaned_corpus_dir.glob("*.json"))
    selected_files = files[batch_start_offset : batch_start_offset + batch_sample_size]
    next_offset = batch_start_offset + len(selected_files)

    LOGGER.info(
        "Build batch started | sample_size=%s start_offset=%s end_offset=%s selected_docs=%s corpus_dir=%s",
        batch_sample_size,
        batch_start_offset,
        batch_start_offset + batch_sample_size,
        len(selected_files),
        runtime_config.cleaned_corpus_dir,
    )
    LOGGER.info(
        "Embedding config | batch_size=%s max_seq_length=%s model=%s",
        runtime_config.embed_batch_size,
        runtime_config.embed_max_seq_length,
        runtime_config.model_name_or_path,
    )

    if not selected_files:
        result = {
            "sample_docs": 0,
            "sample_chunks": 0,
            "qdrant_points_written": 0,
            "opensearch_docs_written": 0,
            "next_offset": batch_start_offset,
            "elapsed_seconds": 0.0,
        }
        _write_state_file(state_path, batch_start_offset, batch_sample_size, 0, 0)
        return result

    started = time.perf_counter()
    try:
        documents = _load_batch_documents(selected_files)
        chunk_records: list[dict[str, Any]] = []

        total_docs = len(documents)
        LOGGER.info("Chunking started | docs=%s", total_docs)

        for idx, (file_path, doc) in enumerate(documents, start=1):
            doc_chunks = build_chunks_for_document(doc, str(file_path), runtime_config)
            chunk_records.extend(doc_chunks)

            if (
                total_docs <= CHUNK_PROGRESS_EVERY
                or idx == total_docs
                or idx % CHUNK_PROGRESS_EVERY == 0
            ):
                LOGGER.info(
                    "Chunking progress | docs=%s/%s chunks=%s last_doc_chunks=%s file=%s",
                    idx,
                    total_docs,
                    len(chunk_records),
                    len(doc_chunks),
                    file_path.name,
                )

        LOGGER.info("Chunking finished | docs=%s chunks=%s", len(documents), len(chunk_records))

        chunk_texts = [record["chunk_text"] for record in chunk_records]
        vectors = encode_texts(chunk_texts, runtime_config)
        vector_size = len(vectors[0]) if vectors else 0
        LOGGER.info("Embedding finished | vectors=%s vector_size=%s", len(vectors), vector_size)

        qdrant_written = 0
        opensearch_written = 0
        if vectors:
            ensure_qdrant_collections(runtime_config, vector_size)
            ensure_opensearch_index(runtime_config)
            qdrant_stats = upsert_qdrant_points_multi(runtime_config, chunk_records, vectors)
            qdrant_written = sum(qdrant_stats.values())
            opensearch_written = bulk_index_opensearch(runtime_config, chunk_records)

        elapsed = round(time.perf_counter() - started, 3)
        result = {
            "sample_docs": len(documents),
            "sample_chunks": len(chunk_records),
            "qdrant_points_written": qdrant_written,
            "opensearch_docs_written": opensearch_written,
            "next_offset": next_offset,
            "elapsed_seconds": elapsed,
        }
        _write_state_file(state_path, next_offset, batch_sample_size, len(documents), len(chunk_records))
        LOGGER.info(
            "Build batch finished | qdrant=%s opensearch=%s next_offset=%s elapsed_seconds=%s",
            qdrant_written,
            opensearch_written,
            next_offset,
            elapsed,
        )
        return result
    except Exception:
        LOGGER.exception("Build batch failed | sample_size=%s start_offset=%s", batch_sample_size, batch_start_offset)
        raise
