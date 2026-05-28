from __future__ import annotations

import argparse
import re

import torch
import torch_mlu  # noqa: F401
from opensearchpy import OpenSearch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = "/AII-heyan/DeepSeek/DeepSeek-R1-Distill-Qwen-7B"
DEFAULT_INDEX_NAME = "ragtestv01_chunks"
DEFAULT_OPENSEARCH_URL = "http://localhost:9200"


def build_opensearch_client(url: str) -> OpenSearch:
    return OpenSearch(
        hosts=[url],
        timeout=60,
        use_ssl=url.startswith("https://"),
        verify_certs=False,
    )


def retrieve_chunks(client: OpenSearch, index_name: str, question: str, top_k: int):
    body = {
        "size": top_k,
        "_source": [
            "chunk_id",
            "doc_id",
            "title",
            "chunk_index",
            "chunk_type",
            "section_name",
            "chunk_text",
            "journal",
            "year",
        ],
        "query": {
            "multi_match": {
                "query": question,
                "fields": [
                    "chunk_text^3",
                    "title^2",
                    "journal",
                    "section_name",
                ],
                "type": "best_fields",
                "operator": "or"
            }
        },
    }
    resp = client.search(index=index_name, body=body)
    return resp["hits"]["hits"]


def build_context(hits) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        src = hit["_source"]
        block = (
            f"[参考资料 {i}]\n"
            f"doc_id: {src.get('doc_id', '')}\n"
            f"chunk_id: {src.get('chunk_id', '')}\n"
            f"title: {src.get('title', '')}\n"
            f"section_name: {src.get('section_name', '')}\n"
            f"journal: {src.get('journal', '')}\n"
            f"year: {src.get('year', '')}\n"
            f"text:\n{src.get('chunk_text', '')}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def strip_think(text: str) -> str:
    # 正常闭合的 think
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)
    # 如果 think 没闭合，直接从 <think> 开始裁掉
    text = re.sub(r"<think>.*$", "", text, flags=re.S)
    return text.strip()


def load_local_model(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to("mlu")

    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question: str, context: str, max_new_tokens: int):
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个严谨、简洁的中文医学文献问答助手。"
                "你必须只依据给定参考资料回答。"
                "如果参考资料不足以支持答案，就明确说：根据当前检索到的资料，无法确定。"
                "不要编造。不要展示思考过程。不要输出<think>标签中的内容。"
                "请直接给出最终答案。"
            ),
        },
        {
            "role": "user",
            "content": (
                "请基于以下参考资料回答问题。\n\n"
                f"参考资料：\n{context}\n\n"
                f"问题：\n{question}\n\n"
                "要求：只输出最终答案，不要输出分析过程。"
            ),
        },
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to("mlu") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    clean_text = strip_think(raw_text)
    return raw_text, clean_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--show-context", action="store_true")
    args = parser.parse_args()

    client = build_opensearch_client(DEFAULT_OPENSEARCH_URL)
    hits = retrieve_chunks(client, DEFAULT_INDEX_NAME, args.question, args.top_k)

    if not hits:
        print("没有检索到任何参考资料。")
        return

    print(f"[INFO] retrieved_hits = {len(hits)}")
    print("\n===== SOURCES =====")
    for i, hit in enumerate(hits, start=1):
        src = hit["_source"]
        print(
            f"[{i}] score={hit.get('_score')} "
            f"doc_id={src.get('doc_id')} "
            f"title={src.get('title', '')[:120]}"
        )

    context = build_context(hits)

    if args.show_context:
        print("\n===== CONTEXT =====")
        print(context)

    tokenizer, model = load_local_model(DEFAULT_MODEL_PATH)
    print(f"\n[INFO] model_device = {next(model.parameters()).device}")
    print(f"[INFO] model_dtype = {next(model.parameters()).dtype}")

    raw_text, clean_text = generate_answer(
        tokenizer=tokenizer,
        model=model,
        question=args.question,
        context=context,
        max_new_tokens=args.max_new_tokens,
    )

    if not clean_text:
        print("\n===== ANSWER =====")
        print("根据当前检索到的资料，无法生成稳定答案。建议提高 max_new_tokens 或更换更匹配的问题。")
        print("\n===== RAW ANSWER =====")
        print(raw_text)
        return

    print("\n===== ANSWER =====")
    print(clean_text)


if __name__ == "__main__":
    main()
