"""
Full Build Script - 支持多Collection分片建库
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunker import build_chunks_for_document
from app.config import BuildConfig, load_config
from app.embedder import encode_texts
from app.stores import (
    bulk_index_opensearch,
    delete_all_qdrant_collections,
    delete_opensearch_index,
    ensure_opensearch_index,
    ensure_qdrant_collections,
    get_qdrant_collection_counts,
    get_qdrant_storage_size,
    upsert_qdrant_points_multi,
)

LOGGER = logging.getLogger(__name__)


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


def _load_batch_documents(files: list[Path]) -> list[tuple[Path, dict[str, Any]]]:
    documents = []
    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            documents.append((file_path, json.load(handle)))
    return documents


def estimate_build_time(config: BuildConfig, sample_size: int = 100) -> dict[str, Any]:
    """
    估算全量建库时间
    返回：预估时间、每批耗时、预估总chunks数
    """
    LOGGER.info(f"Running estimation with {sample_size} documents...")

    # 取少量文档测试
    files = sorted(config.cleaned_corpus_dir.glob("*.json"))[:sample_size]
    if not files:
        return {"error": "No corpus files found"}

    started = time.perf_counter()

    # 测试 chunking
    documents = _load_batch_documents(files)
    chunk_records = []
    for _, doc in documents:
        chunks = build_chunks_for_document(doc, "", config)
        chunk_records.extend(chunks)

    # 测试 embedding
    chunk_texts = [r["chunk_text"] for r in chunk_records]
    vectors = encode_texts(chunk_texts, config)

    elapsed = time.perf_counter() - started

    # 估算
    total_files = len(list(config.cleaned_corpus_dir.glob("*.json")))
    docs_per_second = sample_size / elapsed if elapsed > 0 else 0
    chunks_per_doc = len(chunk_records) / sample_size if sample_size > 0 else 0

    estimated_total_docs = total_files
    estimated_total_chunks = int(estimated_total_docs * chunks_per_doc)
    estimated_seconds = estimated_total_docs / docs_per_second if docs_per_second > 0 else 0
    estimated_minutes = estimated_seconds / 60

    # 估算存储
    avg_vector_size = 1024 * 4  # bytes per float32 vector (实际量化后约1KB)
    estimated_qdrant_mb = (estimated_total_chunks * avg_vector_size) / (1024 * 1024)
    estimated_qdrant_gb = estimated_qdrant_mb / 1024

    return {
        "sample_size": sample_size,
        "sample_chunks": len(chunk_records),
        "chunks_per_doc": round(chunks_per_doc, 2),
        "elapsed_seconds": round(elapsed, 2),
        "docs_per_second": round(docs_per_second, 2),
        "total_corpus_docs": total_files,
        "estimated_total_chunks": estimated_total_chunks,
        "estimated_minutes": round(estimated_minutes, 1),
        "estimated_hours": round(estimated_minutes / 60, 1),
        "estimated_qdrant_gb": round(estimated_qdrant_gb, 2),
    }


def build_full(
    num_docs: int,
    config: BuildConfig | None = None,
    force_rebuild: bool = False,
    skip_estimate: bool = False,
) -> dict[str, Any]:
    """
    全量建库主函数

    Args:
        num_docs: 建库的文档数量
        config: 配置对象
        force_rebuild: 是否强制重建（清空旧数据）
        skip_estimate: 是否跳过估算步骤
    """
    runtime_config = config or load_config()

    LOGGER.info("=" * 60)
    LOGGER.info("Full Build Started")
    LOGGER.info(f"Target documents: {num_docs}")
    LOGGER.info(f"Config: chunk_size={runtime_config.chunk_size}, "
                f"max_chunks_per_doc={runtime_config.max_chunks_per_doc}, "
                f"num_collections={runtime_config.num_qdrant_collections}, "
                f"quantization={runtime_config.use_quantization}")
    LOGGER.info("=" * 60)

    # 1. 估算时间
    if not skip_estimate:
        LOGGER.info("\n[Step 0] Estimating build time...")
        est = estimate_build_time(runtime_config, sample_size=min(100, num_docs))
        LOGGER.info(f"Estimation: {est['estimated_minutes']} minutes for {num_docs} docs")
        LOGGER.info(f"Estimated total chunks: {est['estimated_total_chunks']}")
        LOGGER.info(f"Estimated Qdrant size: {est['estimated_qdrant_gb']} GB")
        LOGGER.info(f"Speed: {est['docs_per_second']} docs/second")
        LOGGER.info("")

        # 如果预估超过10分钟，建议减少数量
        if est['estimated_minutes'] > 10 and num_docs > 500:
            LOGGER.warning(f"Estimated time ({est['estimated_minutes']:.1f} min) > 10 minutes!")
            LOGGER.warning("Will use 500 docs instead.")
            num_docs = 500
            LOGGER.info(f"Adjusted to {num_docs} documents")

    # 2. 清理旧数据
    if force_rebuild:
        LOGGER.info("\n[Step 1] Cleaning old data...")
        delete_all_qdrant_collections(runtime_config)
        delete_opensearch_index(runtime_config)
        LOGGER.info("Old data cleaned")

    # 3. 获取文件列表
    files = sorted(runtime_config.cleaned_corpus_dir.glob("*.json"))
    selected_files = files[:num_docs]

    LOGGER.info(f"\n[Step 2] Selected {len(selected_files)} documents from {len(files)} total")

    if not selected_files:
        return {"error": "No documents to process"}

    # 4. 加载文档
    LOGGER.info("\n[Step 3] Loading documents...")
    started = time.perf_counter()
    documents = _load_batch_documents(selected_files)
    load_time = time.perf_counter() - started
    LOGGER.info(f"Loaded {len(documents)} documents in {load_time:.2f}s")

    # 5. Chunking
    LOGGER.info("\n[Step 4] Chunking...")
    started = time.perf_counter()
    chunk_records = []

    for idx, (_, doc) in enumerate(documents, start=1):
        chunks = build_chunks_for_document(doc, "", runtime_config)
        chunk_records.extend(chunks)

        if idx % 100 == 0:
            LOGGER.info(f"Chunked {idx}/{len(documents)} docs, total chunks: {len(chunk_records)}")

    chunk_time = time.perf_counter() - started
    LOGGER.info(f"Chunked {len(documents)} docs -> {len(chunk_records)} chunks in {chunk_time:.2f}s")
    LOGGER.info(f"Average chunks per doc: {len(chunk_records)/len(documents):.2f}")

    # 6. Embedding
    LOGGER.info("\n[Step 5] Generating embeddings...")
    started = time.perf_counter()
    chunk_texts = [r["chunk_text"] for r in chunk_records]
    vectors = encode_texts(chunk_texts, runtime_config)
    embed_time = time.perf_counter() - started
    LOGGER.info(f"Embedded {len(vectors)} chunks in {embed_time:.2f}s")

    vector_size = len(vectors[0]) if vectors else 0

    # 7. 创建 Collection
    LOGGER.info("\n[Step 6] Creating Qdrant collections...")
    coll_info = ensure_qdrant_collections(runtime_config, vector_size)
    LOGGER.info(f"Collections: created={coll_info['created']}, existing={coll_info['existing']}")

    # 8. 确保 OpenSearch 索引
    LOGGER.info("\n[Step 7] Creating OpenSearch index...")
    ensure_opensearch_index(runtime_config)

    # 9. 写入 Qdrant
    LOGGER.info("\n[Step 8] Writing to Qdrant (multi-collection)...")
    started = time.perf_counter()
    qdrant_stats = upsert_qdrant_points_multi(runtime_config, chunk_records, vectors)
    qdrant_time = time.perf_counter() - started

    total_qdrant = sum(qdrant_stats.values())
    LOGGER.info(f"Wrote {total_qdrant} points to Qdrant in {qdrant_time:.2f}s")
    for coll, count in qdrant_stats.items():
        LOGGER.info(f"  - {coll}: {count} points")

    # 10. 写入 OpenSearch
    LOGGER.info("\n[Step 9] Writing to OpenSearch...")
    started = time.perf_counter()
    os_written = bulk_index_opensearch(runtime_config, chunk_records)
    os_time = time.perf_counter() - started
    LOGGER.info(f"Wrote {os_written} docs to OpenSearch in {os_time:.2f}s")

    # 11. 统计信息
    LOGGER.info("\n[Step 10] Storage statistics...")
    qdrant_size_gb = get_qdrant_storage_size(runtime_config)
    coll_counts = get_qdrant_collection_counts(runtime_config)

    LOGGER.info(f"Qdrant total size: {qdrant_size_gb:.3f} GB")
    LOGGER.info("Collection point counts:")
    for coll, count in coll_counts.items():
        LOGGER.info(f"  - {coll}: {count} points")

    # 12. 总结
    total_time = chunk_time + embed_time + qdrant_time + os_time

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info("Build Complete!")
    LOGGER.info(f"Documents processed: {len(documents)}")
    LOGGER.info(f"Total chunks: {len(chunk_records)}")
    LOGGER.info(f"Qdrant points: {total_qdrant}")
    LOGGER.info(f"OpenSearch docs: {os_written}")
    LOGGER.info(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    LOGGER.info(f"Qdrant storage: {qdrant_size_gb:.3f} GB")
    LOGGER.info("=" * 60)

    return {
        "docs": len(documents),
        "chunks": len(chunk_records),
        "qdrant_points": total_qdrant,
        "opensearch_docs": os_written,
        "qdrant_size_gb": qdrant_size_gb,
        "collection_counts": coll_counts,
        "total_time_seconds": total_time,
        "qdrant_stats": qdrant_stats,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Full Build Script with Multi-Collection Support")
    parser.add_argument("--docs", type=int, default=1000, help="Number of documents to process (default: 1000)")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild (delete old data)")
    parser.add_argument("--skip-estimate", action="store_true", help="Skip time estimation")
    parser.add_argument("--chunk-size", type=int, help="Override chunk size")
    parser.add_argument("--max-chunks", type=int, help="Override max chunks per doc")
    parser.add_argument("--batch-size", type=int, help="Override embed batch size")
    parser.add_argument("--collections", type=int, help="Override number of collections")
    parser.add_argument("--no-quantize", action="store_true", help="Disable vector quantization")

    args = parser.parse_args()

    # 配置日志
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(log_file)

    # 构建配置覆盖
    overrides = {}
    if args.chunk_size:
        overrides["chunk_size"] = args.chunk_size
    if args.max_chunks:
        overrides["max_chunks_per_doc"] = args.max_chunks
    if args.batch_size:
        overrides["embed_batch_size"] = args.batch_size
    if args.collections:
        overrides["num_qdrant_collections"] = args.collections
    if args.no_quantize:
        overrides["use_quantization"] = False

    config = load_config(overrides)

    # 执行建库
    result = build_full(
        num_docs=args.docs,
        config=config,
        force_rebuild=args.force_rebuild,
        skip_estimate=args.skip_estimate,
    )

    print("\n" + "=" * 60)
    print("Result Summary:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 60)
