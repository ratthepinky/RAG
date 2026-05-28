#!/usr/bin/env python3
"""Evaluate dense, sparse, fused, and rerank retrieval stages."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config
from app.embedder import load_embedding_model
from app.reranker import Reranker
from app.stores import query_qdrant_single_collection, query_sparse


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 2.0,
) -> list[dict]:
    scores: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        chunk_id = hit.get("chunk_id", "")
        if chunk_id not in scores:
            scores[chunk_id] = dict(hit)
            scores[chunk_id]["dense_rank"] = rank
            scores[chunk_id]["sparse_rank"] = None

    for rank, hit in enumerate(sparse_hits, start=1):
        chunk_id = hit.get("chunk_id", "")
        if chunk_id not in scores:
            scores[chunk_id] = dict(hit)
            scores[chunk_id]["dense_rank"] = None
            scores[chunk_id]["sparse_rank"] = rank
        else:
            scores[chunk_id]["sparse_rank"] = rank

    for hit in scores.values():
        score = 0.0
        if hit["dense_rank"]:
            score += dense_weight / (k + hit["dense_rank"])
        if hit["sparse_rank"]:
            score += sparse_weight / (k + hit["sparse_rank"])
        hit["rrf_score"] = score

    return sorted(scores.values(), key=lambda item: item["rrf_score"], reverse=True)


def dense_search_sequential(config, query_vector: list[float], limit_per_coll: int) -> list[dict]:
    hits: list[dict] = []
    for collection_name in config.get_collection_names():
        hits.extend(query_qdrant_single_collection(config, collection_name, query_vector, limit_per_coll))
    hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return hits


def is_relevant(hit: dict, keywords: list[str]) -> bool:
    haystack = " ".join(
        str(hit.get(field, "") or "")
        for field in ("title", "chunk_text", "section_name", "journal")
    ).lower()
    return any(str(keyword).lower() in haystack for keyword in keywords)


def calc_metrics(hits: list[dict], keywords: list[str], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    relevance = [is_relevant(hit, keywords) for hit in hits]
    metrics: dict[str, float] = {}

    for top_k in ks:
        top_matches = relevance[:top_k]
        hit_count = sum(1 for matched in top_matches if matched)
        metrics[f"Hit@{top_k}"] = 1.0 if any(top_matches) else 0.0
        metrics[f"Precision@{top_k}"] = hit_count / top_k if top_k else 0.0
        denom = min(len(keywords), top_k) if keywords else top_k
        metrics[f"Recall@{top_k}"] = min(1.0, hit_count / denom) if denom else 0.0

    mrr = 0.0
    for rank, matched in enumerate(relevance, start=1):
        if matched:
            mrr = 1.0 / rank
            break
    metrics["MRR"] = mrr
    return metrics


def summarize_top_hits(hits: list[dict]) -> list[dict]:
    return [
        {
            "rank": idx + 1,
            "title": hit.get("title", "")[:60],
            "score": hit.get("rerank_score", hit.get("rrf_score", hit.get("score", 0.0))),
        }
        for idx, hit in enumerate(hits[:3])
    ]


def evaluate_query(question: str, keywords: list[str], config) -> dict:
    embedder = load_embedding_model(config)
    query_vector = embedder.encode([question], normalize_embeddings=True)[0].tolist()
    del embedder

    dense_hits = dense_search_sequential(config, query_vector, limit_per_coll=config.query_dense_top_k)[
        : config.query_dense_top_k
    ]
    sparse_hits = query_sparse(config, question, limit=config.query_sparse_top_k)
    fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits)[: config.query_fused_top_k]
    reranked_hits = Reranker(config).rerank(
        question,
        fused_hits,
        top_k=min(config.reranker_top_k, len(fused_hits)),
    )

    stages = {
        "dense": dense_hits,
        "sparse": sparse_hits,
        "fused": fused_hits,
        "rerank": reranked_hits,
    }

    return {
        "question": question,
        "counts": {stage: len(hits) for stage, hits in stages.items()},
        "metrics": {stage: calc_metrics(hits, keywords) for stage, hits in stages.items()},
        "top3": {stage: summarize_top_hits(hits) for stage, hits in stages.items()},
    }


def main() -> None:
    config = load_config()
    eval_path = PROJECT_ROOT / "eval_queries.json"
    with eval_path.open("r", encoding="utf-8") as handle:
        queries = json.load(handle)["queries"]

    print("=" * 60)
    print(
        "Retrieval Evaluation | "
        f"dense_top_k={config.query_dense_top_k} "
        f"sparse_top_k={config.query_sparse_top_k} "
        f"fused_top_k={config.query_fused_top_k} "
        f"rerank_top_k={config.reranker_top_k}"
    )
    print("=" * 60)

    results = []
    for query in queries:
        result = evaluate_query(query["question"], query["expected_keywords"], config)
        results.append({"id": query["id"], **result})
        print(f"\n[{query['id']}] {query['question']}")
        for stage in ("dense", "sparse", "fused", "rerank"):
            metrics = result["metrics"][stage]
            print(
                f"  {stage:<6} "
                f"Hit@1={metrics['Hit@1']:.3f} "
                f"Hit@3={metrics['Hit@3']:.3f} "
                f"Recall@3={metrics['Recall@3']:.3f} "
                f"MRR={metrics['MRR']:.3f}"
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    for stage in ("dense", "sparse", "fused", "rerank"):
        print(f"  [{stage}]")
        for metric_name in ("Hit@1", "Hit@3", "Hit@5", "Recall@3", "Recall@5", "MRR"):
            values = [result["metrics"][stage].get(metric_name, 0.0) for result in results]
            print(f"    Avg {metric_name}: {sum(values) / len(values):.4f}")

    output_path = PROJECT_ROOT / "eval_result.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "results": results,
                "config": {
                    "model": config.model_name_or_path,
                    "reranker": config.reranker_model_or_path,
                    "dense_top_k": config.query_dense_top_k,
                    "sparse_top_k": config.query_sparse_top_k,
                    "fused_top_k": config.query_fused_top_k,
                    "rerank_top_k": config.reranker_top_k,
                },
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
