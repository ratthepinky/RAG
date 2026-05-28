"""
RAG 问答脚本 - API 模式
Embedding/Reranker 本地运行，LLM 调用 Kimi/Moonshot API
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch_mlu  # 必须先导入

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import BuildConfig, load_config
from app.embedder import load_embedding_model
from app.reranker import Reranker
from app.stores import query_qdrant_single_collection, query_sparse

LOGGER = logging.getLogger(__name__)

# 全局模型
_embedder_model = None
_reranker_model = None

# KIMI (Moonshot) API 配置
KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")
KIMI_BASE_URL = "https://api.moonshot.cn"


def configure_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def init_api_client():
    """初始化 API 客户端"""
    from openai import OpenAI
    return OpenAI(api_key=KIMI_API_KEY, base_url="https://api.moonshot.cn/v1")


def load_embedder_and_reranker(config: BuildConfig):
    """加载 Embedder 和 Reranker"""
    global _embedder_model, _reranker_model
    
    LOGGER.info("Loading Embedder and Reranker...")
    
    t0 = time.time()
    _embedder_model = load_embedding_model(config)
    LOGGER.info(f"Embedder loaded in {time.time()-t0:.1f}s")
    
    t0 = time.time()
    _reranker_model = Reranker(config)
    LOGGER.info(f"Reranker loaded in {time.time()-t0:.1f}s")
    
    LOGGER.info("Embedder and Reranker loaded!")


def generate_answer_via_api(question: str, context: str) -> tuple[str, float]:
    """使用 KIMI (Moonshot) API 生成回答"""
    system_prompt = """你是一个专业的医学知识助手。请根据给定的参考文档回答用户问题。

【回答规则】
1. 必须且只能依据给定的上下文文献回答，不得使用外部知识
2. 如果上下文信息不足以完整回答，请明确说明"根据现有资料，以下是可以确定的部分：..."
3. 回答结构要求：
   - 先给出核心结论（1-2句话）
   - 再分要点详细说明
   - 每个事实性陈述必须附引用 [n]，n为文献编号
4. 使用专业的医学术语，保持简洁准确
5. 如果涉及诊断/治疗建议，必须标注"请以临床医生判断为准"

【禁止事项】
- 禁止编造上下文中不存在的信息
- 禁止输出"根据当前检索到的资料无法确定"后没有补充说明

【引用格式】
在相关句子末尾使用[n]格式标注引用来源。"""

    user_message = f"问题：{question}\n\n参考文档：\n{context}\n\n请根据以上参考文档回答问题："

    t0 = time.time()
    
    client = init_api_client()
    response = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    
    answer = response.choices[0].message.content
    
    return answer, time.time() - t0


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    scores: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        chunk_id = hit.get("chunk_id", "")
        if chunk_id not in scores:
            scores[chunk_id] = hit.copy()
            scores[chunk_id]["dense_rank"] = rank
            scores[chunk_id]["sparse_rank"] = None

    for rank, hit in enumerate(sparse_hits, start=1):
        chunk_id = hit.get("chunk_id", "")
        if chunk_id not in scores:
            scores[chunk_id] = hit.copy()
            scores[chunk_id]["dense_rank"] = None
            scores[chunk_id]["sparse_rank"] = rank
        else:
            scores[chunk_id]["sparse_rank"] = rank

    for chunk_id, hit in scores.items():
        rrf_score = 0.0
        if hit["dense_rank"]:
            rrf_score += dense_weight / (k + hit["dense_rank"])
        if hit["sparse_rank"]:
            rrf_score += sparse_weight / (k + hit["sparse_rank"])
        hit["rrf_score"] = rrf_score

    return sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)


def dense_search_sequential(config: BuildConfig, query_vector: list[float], limit_per_coll: int) -> list[dict]:
    hits: list[dict] = []
    for collection_name in config.get_collection_names():
        hits.extend(query_qdrant_single_collection(config, collection_name, query_vector, limit_per_coll))
    hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return hits


def diversify_by_doc(hits: list[dict], final_top_k: int = 8, max_per_doc: int = 2) -> list[dict]:
    doc_counts: dict[str, int] = {}
    selected: list[dict] = []

    for hit in hits:
        doc_id = hit.get("doc_id", "unknown")
        if doc_counts.get(doc_id, 0) < max_per_doc:
            selected.append(hit)
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
            if len(selected) >= final_top_k:
                break

    if len(selected) < final_top_k:
        for hit in hits:
            if hit not in selected:
                selected.append(hit)
                if len(selected) >= final_top_k:
                    break

    return selected


def build_context(chunks: list[dict], max_chars: int = 8000) -> str:
    context_parts = []
    total_chars = 0

    for idx, chunk in enumerate(chunks, start=1):
        title = chunk.get("title", "Unknown")
        journal = chunk.get("journal", "")
        year = chunk.get("year", "")
        chunk_text = chunk.get("chunk_text", "")[:2000]

        header = f"[{idx}] {title}"
        if journal:
            header += f" ({journal}, {year})" if year else f" ({journal})"
        elif year:
            header += f" ({year})"

        chunk_str = f"{header}\n{chunk_text}\n"
        if total_chars + len(chunk_str) <= max_chars:
            context_parts.append(chunk_str)
            total_chars += len(chunk_str)

    return "\n".join(context_parts)


def query(question: str, config: BuildConfig | None = None) -> dict[str, Any]:
    """执行一次问答"""
    global _embedder_model, _reranker_model

    if config is None:
        config = load_config()

    timings = {}
    dense_top_k = max(1, int(config.query_dense_top_k))
    sparse_top_k = max(1, int(config.query_sparse_top_k))
    fused_top_k = max(1, int(config.query_fused_top_k))
    rerank_top_k = max(1, int(config.reranker_top_k))
    final_top_k = max(1, int(config.query_final_top_k))

    # Step 1: Embedding
    t0 = time.perf_counter()
    query_vector = _embedder_model.encode([question], normalize_embeddings=True)[0].tolist()
    timings["embedding"] = time.perf_counter() - t0

    # Step 2: Dense search
    t0 = time.perf_counter()
    dense_hits = dense_search_sequential(config, query_vector, limit_per_coll=dense_top_k)[:dense_top_k]
    timings["dense_search"] = time.perf_counter() - t0

    # Step 3: Sparse search
    t0 = time.perf_counter()
    sparse_hits = query_sparse(config, question, limit=sparse_top_k)
    timings["sparse_search"] = time.perf_counter() - t0

    # Step 4: RRF fusion
    t0 = time.perf_counter()
    fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, sparse_weight=2.0)[:fused_top_k]
    timings["fusion"] = time.perf_counter() - t0

    # Step 5: Rerank
    t0 = time.perf_counter()
    reranked_hits = _reranker_model.rerank(question, fused_hits, top_k=min(rerank_top_k, len(fused_hits)))
    timings["rerank"] = time.perf_counter() - t0

    # Step 6: Diversify
    selected_chunks = diversify_by_doc(reranked_hits, final_top_k=final_top_k, max_per_doc=2)

    # Step 7: Build context
    t0 = time.perf_counter()
    context = build_context(selected_chunks, max_chars=config.max_context_chars)
    timings["context_build"] = time.perf_counter() - t0

    # Step 8: LLM generate (via API)
    answer, llm_time = generate_answer_via_api(question, context)
    timings["llm_generate"] = llm_time

    total_time = sum(timings.values())

    LOGGER.info("\n" + "=" * 60)
    LOGGER.info(f"Total time: {total_time:.2f}s")
    LOGGER.info("Timings breakdown:")
    for step, t in timings.items():
        LOGGER.info(f"  - {step}: {t:.3f}s")
    LOGGER.info("=" * 60)

    return {
        "question": question,
        "answer": answer,
        "chunks": selected_chunks,
        "timings": timings,
        "total_time": total_time,
    }


def interactive_mode(config: BuildConfig):
    """交互式问答"""
    print("\n" + "=" * 60)
    print("FUO RAG Q&A System")
    print("Models loaded and ready!")
    print("=" * 60)
    print("Type 'quit' or 'exit' to exit")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except EOFError:
            break

        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not question:
            continue

        result = query(question, config)

        print("\n" + "-" * 60)
        print("Answer:")
        print(result["answer"])
        print("-" * 60)
        print(f"\nSources ({len(result['chunks'])} chunks):")
        for idx, chunk in enumerate(result["chunks"], start=1):
            title = chunk.get("title", "Unknown")[:50]
            score = chunk.get("rerank_score", chunk.get("score", 0))
            print(f"  [{idx}] {title}... (score: {score:.4f})")
        print("-" * 60)
        print(f"Total time: {result['total_time']:.2f}s")
        print()


if __name__ == "__main__":
    configure_logging()
    config = load_config()
    
    # 检查 API Key
    if not KIMI_API_KEY:
        print("ERROR: 请设置 KIMI_API_KEY 环境变量")
        print("  export KIMI_API_KEY='your-api-key'")
        sys.exit(1)
    
    # 加载 Embedder 和 Reranker（只执行一次）
    load_embedder_and_reranker(config)
    
    # 进入交互模式
    interactive_mode(config)
