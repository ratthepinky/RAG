from __future__ import annotations

import logging
from typing import Any

from app.config import BuildConfig
from app.stores import (
    _build_opensearch_client,
    query_qdrant_multi_collection,
)
from app.retrieval_postprocess import filter_rerank_dedupe_retrieval_hits

LOGGER = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _minmax_normalize(items: list[dict[str, Any]], score_key: str, out_key: str) -> None:
    if not items:
        return

    scores = [_safe_float(item.get(score_key, 0.0)) for item in items]
    min_score = min(scores)
    max_score = max(scores)

    if max_score <= min_score:
        for item in items:
            item[out_key] = 1.0
        return

    for item in items:
        score = _safe_float(item.get(score_key, 0.0))
        item[out_key] = (score - min_score) / (max_score - min_score)


def query_opensearch_sparse(
    config: BuildConfig,
    query_text: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    OpenSearch / BM25 sparse retrieval.
    返回字段会尽量与 Qdrant payload 对齐。
    """
    if limit is None:
        limit = config.query_sparse_top_k

    client = _build_opensearch_client(config)

    if not client.indices.exists(index=config.opensearch_index_name):
        LOGGER.warning("OpenSearch index does not exist: %s", config.opensearch_index_name)
        return []

    body = {
        "size": limit,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "chunk_text^4",
                    "title^3",
                    "section_name^2",
                    "abstract^2",
                    "journal",
                    "year",
                    "chunk_type",
                ],
                "type": "best_fields",
                "operator": "or",
            }
        },
        "_source": [
            "chunk_id",
            "doc_id",
            "chunk_text",
            "title",
            "section_name",
            "journal",
            "year",
            "chunk_type",
            "abstract",
            "source_type",
        ],
    }

    try:
        response = client.search(index=config.opensearch_index_name, body=body)
    except Exception as exc:
        LOGGER.warning("OpenSearch query failed: %s", exc)
        return []

    hits: list[dict[str, Any]] = []

    for item in response.get("hits", {}).get("hits", []):
        src = item.get("_source") or {}
        src = dict(src)
        src["os_score"] = item.get("_score", 0.0)
        src["_retrieval_sources"] = ["opensearch"]
        hits.append(src)

    return hits


def _merge_dense_sparse_hits(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    *,
    dense_weight: float = 0.65,
    sparse_weight: float = 0.35,
) -> list[dict[str, Any]]:
    """
    合并 dense Qdrant 与 sparse OpenSearch 结果。
    使用 chunk_id 优先合并；没有 chunk_id 时退化到 doc_id + title。
    """
    dense_hits = [dict(h) for h in dense_hits]
    sparse_hits = [dict(h) for h in sparse_hits]

    # Qdrant 原 score 通常在 score 或 _rerank_score 里。
    for h in dense_hits:
        h["dense_score"] = _safe_float(h.get("_rerank_score", h.get("score", 0.0)))
        h.setdefault("_retrieval_sources", [])
        if "qdrant" not in h["_retrieval_sources"]:
            h["_retrieval_sources"].append("qdrant")

    for h in sparse_hits:
        h["sparse_score"] = _safe_float(h.get("os_score", 0.0))
        h.setdefault("_retrieval_sources", [])
        if "opensearch" not in h["_retrieval_sources"]:
            h["_retrieval_sources"].append("opensearch")

    _minmax_normalize(dense_hits, "dense_score", "dense_norm")
    _minmax_normalize(sparse_hits, "sparse_score", "sparse_norm")

    merged: dict[str, dict[str, Any]] = {}

    def key_of(hit: dict[str, Any]) -> str:
        chunk_id = hit.get("chunk_id")
        if chunk_id:
            return f"chunk:{chunk_id}"
        doc_id = hit.get("doc_id") or ""
        title = hit.get("title") or ""
        section = hit.get("section_name") or ""
        return f"fallback:{doc_id}:{title}:{section}"

    for h in dense_hits:
        key = key_of(h)
        item = merged.setdefault(key, dict(h))
        item["dense_score"] = max(
            _safe_float(item.get("dense_score", 0.0)),
            _safe_float(h.get("dense_score", 0.0)),
        )
        item["dense_norm"] = max(
            _safe_float(item.get("dense_norm", 0.0)),
            _safe_float(h.get("dense_norm", 0.0)),
        )
        sources = set(item.get("_retrieval_sources", []))
        sources.add("qdrant")
        item["_retrieval_sources"] = sorted(sources)

    for h in sparse_hits:
        key = key_of(h)
        if key not in merged:
            merged[key] = dict(h)
        item = merged[key]

        # 如果 sparse 命中有更完整字段，补到已有 dense hit 上。
        for field in [
            "chunk_id",
            "doc_id",
            "chunk_text",
            "title",
            "section_name",
            "journal",
            "year",
            "chunk_type",
            "abstract",
            "source_type",
        ]:
            if not item.get(field) and h.get(field):
                item[field] = h[field]

        item["sparse_score"] = max(
            _safe_float(item.get("sparse_score", 0.0)),
            _safe_float(h.get("sparse_score", 0.0)),
        )
        item["sparse_norm"] = max(
            _safe_float(item.get("sparse_norm", 0.0)),
            _safe_float(h.get("sparse_norm", 0.0)),
        )

        sources = set(item.get("_retrieval_sources", []))
        sources.add("opensearch")
        item["_retrieval_sources"] = sorted(sources)

    results = []

    for item in merged.values():
        dense_norm = _safe_float(item.get("dense_norm", 0.0))
        sparse_norm = _safe_float(item.get("sparse_norm", 0.0))

        hybrid_score = dense_weight * dense_norm + sparse_weight * sparse_norm

        # 统一 score 字段，方便 retrieval_postprocess 和后续 prompt 使用
        item["hybrid_score"] = hybrid_score
        item["score"] = hybrid_score
        results.append(item)

    results.sort(key=lambda x: _safe_float(x.get("hybrid_score", 0.0)), reverse=True)
    return results


def query_hybrid_retrieval(
    config: BuildConfig,
    query_text: str,
    query_vector: list[float],
    *,
    dense_top_k: int | None = None,
    sparse_top_k: int | None = None,
    fused_top_k: int | None = None,
    final_top_k: int | None = None,
    dense_weight: float = 0.75,
    sparse_weight: float = 0.25,
) -> list[dict[str, Any]]:
    """
    正式混合检索入口：
    Qdrant dense retrieval + OpenSearch BM25 retrieval + merge + postprocess.
    """
    dense_top_k = dense_top_k or config.query_dense_top_k
    sparse_top_k = sparse_top_k or config.query_sparse_top_k
    fused_top_k = fused_top_k or config.query_fused_top_k
    final_top_k = final_top_k or config.query_final_top_k

    dense_hits = query_qdrant_multi_collection(
        config,
        query_vector,
        limit_per_coll=dense_top_k,
    )

    sparse_hits = query_opensearch_sparse(
        config,
        query_text,
        limit=sparse_top_k,
    )

    merged = _merge_dense_sparse_hits(
        dense_hits,
        sparse_hits,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )

    merged = merged[:fused_top_k]

    postprocessed = filter_rerank_dedupe_retrieval_hits(
        merged,
        dedupe_by_doc=True,
    )

    return postprocessed[:final_top_k]