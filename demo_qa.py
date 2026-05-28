#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AII-Heyan Hybrid RAG Demo - 连续问答版

用途：
- 演示用交互式问答脚本
- 启动后可以连续手动输入医学问题
- 输入 q / quit / exit 退出
- 直接回车使用默认 FUO 演示问题

当前正式链路：
bge-m3 embedding
→ Qdrant dense retrieval
→ OpenSearch BM25 retrieval
→ hybrid_retrieval
→ retrieval_postprocess
→ Kimi / Moonshot LLM

运行：
cd /AII-heyan/ragtestv01_server_bundle_release
source /torch/venv3/pytorch/bin/activate
/torch/venv3/pytorch/bin/python demo_qa.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import load_config
from app.hybrid_retrieval import query_hybrid_retrieval


PROJECT_ROOT = Path("/AII-heyan/ragtestv01_server_bundle_release")

MODEL_PATH = (
    "/AII-heyan/ragtestv01_server_bundle_release/artifacts/model_cache/"
    "models--BAAI--bge-m3/snapshots/"
    "5617a9f61b028005a4858fdac845db406aefb181"
)

MAX_CONTEXT_CHARS = 9000
FINAL_TOP_K = 10

DEFAULT_QUESTION = (
    "患者持续发热两周，常规抗感染治疗效果不佳，"
    "血常规和胸片没有明显异常，应该如何进一步评估不明原因发热？"
)


def load_dotenv_simple(path: str | Path = ".env") -> None:
    """读取项目 .env。不会打印 key。"""
    env_path = Path(path)

    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path

    if not env_path.exists():
        print(f"WARNING: .env not found: {env_path}", flush=True)
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'").strip('"')


def format_score(value: Any) -> str:
    if value is None:
        return "None"

    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def build_context(hits: list[dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """把检索结果拼成 LLM 上下文。"""
    parts: list[str] = []
    total = 0

    for i, h in enumerate(hits, 1):
        text = h.get("chunk_text") or h.get("text") or h.get("content") or ""
        title = h.get("title") or ""
        section = h.get("section_name") or ""
        journal = h.get("journal") or ""
        year = h.get("year") or ""
        score = h.get("score")
        hybrid_score = h.get("hybrid_score")
        dense_score = h.get("dense_score")
        sparse_score = h.get("sparse_score")
        sources = h.get("_retrieval_sources")

        block = (
            f"[{i}]\n"
            f"title: {title}\n"
            f"section: {section}\n"
            f"journal/year: {journal} {year}\n"
            f"sources: {sources}\n"
            f"score: {score}, hybrid_score: {hybrid_score}, "
            f"dense_score: {dense_score}, sparse_score: {sparse_score}\n"
            f"content: {text}\n"
        )

        if total + len(block) > max_chars:
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


def print_retrieved_chunks(hits: list[dict[str, Any]], top_n: int = 8) -> None:
    print("\n" + "=" * 80)
    print("TOP RETRIEVED CHUNKS")
    print("=" * 80)

    for i, h in enumerate(hits[:top_n], 1):
        text = h.get("chunk_text") or h.get("text") or h.get("content") or ""

        print(f"\n#{i}")
        print("score        =", format_score(h.get("score")))
        print("hybrid_score =", format_score(h.get("hybrid_score")))
        print("dense_score  =", format_score(h.get("dense_score")))
        print("sparse_score =", format_score(h.get("sparse_score")))
        print("sources      =", h.get("_retrieval_sources"))
        print("title        =", h.get("title"))
        print("section      =", h.get("section_name"))
        print("doc_id       =", h.get("doc_id"))
        print("chunk_id     =", h.get("chunk_id"))
        print("text_preview =", text[:450].replace("\n", " "))


def make_prompt(question: str, context: str) -> str:
    return f"""你是一个医学知识库 RAG 问答助手。你需要根据知识库检索结果回答用户的问题。

用户问题：
{question}

知识库检索结果：
{context}

请回答用户问题。

请遵守以下原则：
- 以知识库检索结果为主要依据。
- 不要编造检索结果中没有的具体数据、阈值、检查项目、药物方案或诊断结论。
- 如果检索结果不能充分回答问题，请说明资料不足，并只做谨慎的一般性解释。
- 根据用户问题选择回答重点：用户问症状就重点讲症状和表现；问病因就重点讲病因；问检查就重点讲检查；问治疗就重点讲治疗和就医建议。
- 对医学问题保持保守，不建议用户自行使用处方药。
- 涉及持续发热、症状加重、诊断不明、严重不适等情况时，应提醒及时就医或由医生面诊/住院评估。

请用中文回答，尽量结构清晰、分点说明。
"""


def answer_one_question(
    question: str,
    cfg: Any,
    model: SentenceTransformer,
    client: OpenAI,
) -> None:
    """单轮 RAG 问答。"""
    t_start = time.time()

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(question)

    t1 = time.time()
    query_vector = model.encode([question], normalize_embeddings=True)[0].tolist()
    print("\nembed_seconds            =", round(time.time() - t1, 3), flush=True)

    t2 = time.time()
    hits = query_hybrid_retrieval(
        cfg,
        query_text=question,
        query_vector=query_vector,
        dense_top_k=30,
        sparse_top_k=30,
        fused_top_k=60,
        final_top_k=FINAL_TOP_K,
    )
    print("hybrid_retrieval_seconds =", round(time.time() - t2, 3), flush=True)
    print("hits                     =", len(hits), flush=True)

    print_retrieved_chunks(hits, top_n=8)

    context = build_context(hits)
    prompt = make_prompt(question, context)

    t3 = time.time()
    response = client.chat.completions.create(
        model=os.getenv("KIMI_MODEL", "moonshot-v1-8k"),
        messages=[
            {
                "role": "system",
                "content": "你是一个严谨、保守、基于证据的医学 RAG 问答助手。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(response.choices[0].message.content)

    print("\n" + "=" * 80)
    print("TIMING")
    print("=" * 80)
    print("llm_seconds              =", round(time.time() - t3, 3))
    print("total_seconds            =", round(time.time() - t_start, 3))


def main() -> None:
    os.chdir(PROJECT_ROOT)
    load_dotenv_simple(PROJECT_ROOT / ".env")

    cfg = load_config()

    print("\n" + "=" * 80)
    print("AII-Heyan Hybrid RAG Demo - 连续问答版")
    print("=" * 80)
    print("qdrant_url               =", cfg.qdrant_url)
    print("qdrant_collections       =", cfg.get_collection_names())
    print("opensearch_url           =", cfg.opensearch_url)
    print("opensearch_index_name    =", cfg.opensearch_index_name)
    print("=" * 80, flush=True)

    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 MOONSHOT_API_KEY 或 KIMI_API_KEY，请检查 .env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )

    print("\n正在加载 bge-m3 embedding 模型，首次加载可能需要一些时间。")
    t0 = time.time()
    model = SentenceTransformer(MODEL_PATH)
    print("model_load_seconds       =", round(time.time() - t0, 3), flush=True)

    print("\n" + "=" * 80)
    print("进入连续问答模式")
    print("=" * 80)
    print("输入医学问题后回车即可开始问答。")
    print("直接回车：使用默认 FUO 演示问题。")
    print("输入 q / quit / exit：退出。")
    print("=" * 80)

    while True:
        print()
        question = input("医学问题：").strip()

        if question.lower() in {"q", "quit", "exit"}:
            print("退出 demo。")
            break

        if not question:
            question = DEFAULT_QUESTION

        try:
            answer_one_question(
                question=question,
                cfg=cfg,
                model=model,
                client=client,
            )
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，本轮中断。输入 q 可退出，或继续输入下一个问题。")
        except Exception as exc:
            print("\n" + "=" * 80)
            print("ERROR")
            print("=" * 80)
            print(repr(exc))
            print("本轮失败，但 demo 仍会继续。你可以换一个问题，或输入 q 退出。")


if __name__ == "__main__":
    main()