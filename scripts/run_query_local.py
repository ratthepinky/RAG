"""
Enhanced Query Script with Reranker and Multi-Collection Support
Embedding + Reranker 本地运行，LLM 调用 Kimi API
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import BuildConfig, load_config
from app.embedder import load_embedding_model
from app.reranker import Reranker
from app.stores import query_qdrant_single_collection, query_sparse

LOGGER = logging.getLogger(__name__)

# 全局变量
_embedder_model = None
_reranker_model = None

# Kimi API 配置
KIMI_API_KEY = os.environ.get("KIMI_API_KEY", "")


def configure_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def check_llm_service() -> bool:
    """检查 Kimi API 是否可用"""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=KIMI_API_KEY, base_url="https://api.moonshot.cn/v1")
        client.models.list()
        LOGGER.info("Kimi API is ready!")
        return True
    except Exception as e:
        LOGGER.error(f"Kimi API not available: {e}")
    return False


def generate_answer(question: str, context: str) -> tuple[str, float]:
    """调用 Kimi API 生成回答"""
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

    try:
        from openai import OpenAI
        t0 = time.perf_counter()
        client = OpenAI(api_key=KIMI_API_KEY, base_url="https://api.moonshot.cn/v1")
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
        return answer, time.perf_counter() - t0
    except Exception as e:
        raise Exception(f"Kimi API error: {e}")


def release_embedder():
    """释放 Embedder 显存"""
    global _embedder_model
    
    if _embedder_model is not None:
        del _embedder_model
        _embedder_model = None
        from app.embedder import _load_embedding_model_cached
        if hasattr(_load_embedding_model_cached, 'cache_clear'):
            _load_embedding_model_cached.cache_clear()
        import gc
        gc.collect()
        LOGGER.info("Embedder released")


def reciprocal_rank_fusion(
    dense_hits: list[dict],
    sparse_hits: list[dict],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> list[dict]:
    """RRF 融合 Dense 和 Sparse 结果"""
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
    """按文档去重"""
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
    """构建上下文字符串"""
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
    """完整的问答流程"""
    global _embedder_model, _reranker_model
    
    if config is None:
        config = load_config()

    timings = {}
    dense_top_k = max(1, int(config.query_dense_top_k))
    sparse_top_k = max(1, int(config.query_sparse_top_k))
    fused_top_k = max(1, int(config.query_fused_top_k))
    rerank_top_k = max(1, int(config.reranker_top_k))
    final_top_k = max(1, int(config.query_final_top_k))

    # ====== Step 1: Embedding 检索 ======
    t0 = time.perf_counter()
    _embedder_model = load_embedding_model(config)
    query_vector = _embedder_model.encode([question], normalize_embeddings=True)[0].tolist()
    timings["embedding"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    dense_hits = dense_search_sequential(config, query_vector, limit_per_coll=dense_top_k)[:dense_top_k]
    timings["dense_search"] = time.perf_counter() - t0

    # ====== Step 2: Sparse 检索 ======
    t0 = time.perf_counter()
    sparse_hits = query_sparse(config, question, limit=sparse_top_k)
    timings["sparse_search"] = time.perf_counter() - t0

    # ====== Step 3: RRF 融合 ======
    t0 = time.perf_counter()
    fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, sparse_weight=2.0)[:fused_top_k]
    timings["fusion"] = time.perf_counter() - t0

    # ====== Step 4: Rerank ======
    t0 = time.perf_counter()
    _reranker_model = Reranker(config)
    reranked_hits = _reranker_model.rerank(question, fused_hits, top_k=min(rerank_top_k, len(fused_hits)))
    timings["rerank"] = time.perf_counter() - t0

    # ====== 释放 Embedder 和 Reranker ======
    release_embedder()
    del _reranker_model
    _reranker_model = None
    import gc
    gc.collect()

    # ====== Step 5: 去重 ======
    selected_chunks = diversify_by_doc(reranked_hits, final_top_k=final_top_k, max_per_doc=2)

    # ====== Step 6: 构建上下文 ======
    t0 = time.perf_counter()
    context = build_context(selected_chunks, max_chars=config.max_context_chars)
    timings["context_build"] = time.perf_counter() - t0

    # ====== Step 7: LLM 生成 (HTTP API) ======
    t0 = time.perf_counter()
    answer, llm_time = generate_answer(question, context)
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


def interactive_mode(config: BuildConfig | None = None):
    """交互式问答模式"""
    print("\n" + "=" * 60)
    print("FUO Knowledge Base Q&A")
    print("=" * 60)
    
    if not check_llm_service():
        print("ERROR: Kimi API is not available!")
        return
    
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
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Q&A with Reranker")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--config", type=str, help="Custom config file")

    args = parser.parse_args()

    configure_logging()

    config = load_config()

    if args.question:
        if not check_llm_service():
            print("ERROR: Kimi API is not available!")
            sys.exit(1)
        
        result = query(args.question, config)
        print("\n" + "=" * 60)
        print("Question:", result["question"])
        print("=" * 60)
        print("\nAnswer:")
        print(result["answer"])
        print("=" * 60)
        print(f"\nTotal time: {result['total_time']:.2f}s")
        print(f"Timings: {json.dumps(result['timings'], indent=2)}")
    else:
        interactive_mode(config)
