#!/usr/bin/env python3
"""
完整测试流程：删除旧库 -> 新建1000条数据 -> 用7B模型回答3个FUO问题并记录时间
"""
from __future__ import annotations

import sys
import os
import time
import shutil
from pathlib import Path
from datetime import datetime

# 强制离线模式
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "artifacts" / "model_cache"))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch_mlu  # noqa: F401
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

from app.config import load_config
from app.builder import build_index_batch, configure_logging


# FUO相关问题列表
FUO_QUESTIONS = [
    "什么是发热待查（FUO）？其诊断标准是什么？",
    "FUO的常见病因有哪些？如何进行鉴别诊断？",
    "对于不明原因发热的患者，应该进行哪些检查和处理流程？"
]

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """去除模型输出中的思考过程"""
    text = THINK_BLOCK_RE.sub("", text)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def delete_existing_indexes(config):
    """删除现有的Qdrant和OpenSearch索引"""
    print("\n" + "="*80)
    print("步骤1: 删除现有的索引库")
    print("="*80)
    
    # 删除Qdrant本地数据库
    qdrant_path = config.qdrant_url
    if qdrant_path.exists():
        print(f"[INFO] 删除Qdrant数据库: {qdrant_path}")
        shutil.rmtree(qdrant_path)
        print(f"[OK] Qdrant数据库已删除")
    else:
        print(f"[INFO] Qdrant路径不存在: {qdrant_path}")
    
    # 删除OpenSearch索引
    try:
        client = OpenSearch(
            hosts=[config.opensearch_url],
            timeout=30,
            use_ssl=config.opensearch_url.startswith("https://"),
            verify_certs=False,
        )
        
        if client.indices.exists(index=config.opensearch_index_name):
            print(f"[INFO] 删除OpenSearch索引: {config.opensearch_index_name}")
            client.indices.delete(index=config.opensearch_index_name)
            print(f"[OK] OpenSearch索引已删除")
        else:
            print(f"[INFO] OpenSearch索引不存在: {config.opensearch_index_name}")
    except Exception as e:
        print(f"[WARNING] 删除OpenSearch索引时出错: {e}")
    
    print("[OK] 所有旧索引已清理完成\n")


def build_new_index(config, sample_size=1000):
    """构建新的索引库"""
    print("\n" + "="*80)
    print(f"步骤2: 构建新索引库 (样本数: {sample_size})")
    print("="*80)
    
    start_time = time.time()
    
    # 配置日志
    log_file = config.logs_dir / f"rebuild_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    configure_logging(log_file)
    
    print(f"[INFO] 开始建库...")
    print(f"[INFO] 日志文件: {log_file}")
    
    result = build_index_batch(
        sample_size=sample_size,
        start_offset=0,
        config=config,
        state_file=None,
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n[OK] 建库完成!")
    print(f"  - 处理文档数: {result.get('docs_processed', 0)}")
    print(f"  - 生成chunks数: {result.get('chunks_created', 0)}")
    print(f"  - 耗时: {elapsed:.2f}秒 ({elapsed/60:.2f}分钟)")
    
    return result


def load_local_llm(model_path: str):
    """加载本地7B LLM模型"""
    print(f"\n[INFO] 正在加载LLM模型: {model_path}")
    
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
    print(f"[OK] LLM模型加载完成 (设备: {next(model.parameters()).device})")
    
    return tokenizer, model


def retrieve_context(config, question: str, top_k: int = 5):
    """从OpenSearch检索相关上下文"""
    try:
        client = OpenSearch(
            hosts=[config.opensearch_url],
            timeout=30,
            use_ssl=config.opensearch_url.startswith("https://"),
            verify_certs=False,
        )
        
        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": question,
                    "fields": [
                        "title^3",
                        "chunk_text^2",
                        "journal",
                        "section_name",
                    ],
                    "type": "best_fields",
                    "operator": "or",
                }
            },
        }
        
        response = client.search(index=config.opensearch_index_name, body=body)
        hits = response.get("hits", {}).get("hits", [])
        
        return hits
    except Exception as e:
        print(f"[ERROR] 检索失败: {e}")
        return []


def build_context(hits) -> str:
    """构建上下文字符串"""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        src = hit["_source"]
        block = (
            f"[参考资料 {i}]\n"
            f"标题: {src.get('title', '')}\n"
            f"doc_id: {src.get('doc_id', '')}\n"
            f"chunk_id: {src.get('chunk_id', '')}\n"
            f"章节: {src.get('section_name', '')}\n"
            f"期刊: {src.get('journal', '')}\n"
            f"年份: {src.get('year', '')}\n"
            f"内容:\n{src.get('chunk_text', '')}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def generate_answer(tokenizer, model, question: str, context: str, max_new_tokens: int = 512):
    """使用LLM生成答案"""
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个严谨、专业的医学知识库问答助手。"
                "你必须只依据给定参考资料回答问题。"
                "如果参考资料不足以支持答案，请明确说明。"
                "不要编造信息。不要展示思考过程。"
                "请直接给出最终答案，并在相关结论后附上引用编号，例如[1]。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"请基于以下参考资料回答问题。\n\n"
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


def test_fuo_questions(config, model_path: str):
    """测试3个FUO相关问题"""
    print("\n" + "="*80)
    print("步骤3: 使用7B模型回答3个FUO相关问题")
    print("="*80)
    
    # 加载模型
    tokenizer, model = load_local_llm(model_path)
    
    results = []
    
    for idx, question in enumerate(FUO_QUESTIONS, start=1):
        print(f"\n{'='*80}")
        print(f"问题 {idx}/3: {question}")
        print(f"{'='*80}")
        
        # 检索上下文
        print(f"[INFO] 正在检索相关文献...")
        retrieve_start = time.time()
        hits = retrieve_context(config, question, top_k=5)
        retrieve_time = time.time() - retrieve_start
        
        print(f"[OK] 检索到 {len(hits)} 条相关资料 (耗时: {retrieve_time:.2f}秒)")
        
        if not hits:
            print("[WARNING] 未找到相关资料")
            continue
        
        # 显示检索到的资料摘要
        print(f"\n检索到的参考资料:")
        for i, hit in enumerate(hits[:3], start=1):
            src = hit["_source"]
            score = hit.get("_score", 0)
            print(f"  [{i}] Score: {score:.4f} | 标题: {src.get('title', '')[:80]}")
        
        # 构建上下文
        context = build_context(hits)
        
        # 生成答案
        print(f"\n[INFO] 正在生成答案...")
        generate_start = time.time()
        raw_text, clean_text = generate_answer(
            tokenizer=tokenizer,
            model=model,
            question=question,
            context=context,
            max_new_tokens=512,
        )
        generate_time = time.time() - generate_start
        
        total_time = retrieve_time + generate_time
        
        # 显示结果
        print(f"\n{'─'*80}")
        print(f"【问题】{question}")
        print(f"{'─'*80}")
        print(f"\n【答案】")
        print(clean_text if clean_text else "根据当前检索到的资料，无法确定。")
        print(f"\n{'─'*80}")
        print(f"性能统计:")
        print(f"  - 检索耗时: {retrieve_time:.2f}秒")
        print(f"  - 生成耗时: {generate_time:.2f}秒")
        print(f"  - 总耗时: {total_time:.2f}秒")
        print(f"{'─'*80}")
        
        results.append({
            "question": question,
            "answer": clean_text,
            "retrieve_time": retrieve_time,
            "generate_time": generate_time,
            "total_time": total_time,
            "sources_count": len(hits),
        })
    
    return results


def print_summary(results, build_result, total_build_time):
    """打印总结报告"""
    print("\n" + "="*80)
    print("测试总结报告")
    print("="*80)
    
    print(f"\n📊 建库统计:")
    print(f"  - 处理文档数: {build_result.get('docs_processed', 0)}")
    print(f"  - 生成chunks数: {build_result.get('chunks_created', 0)}")
    print(f"  - 建库耗时: {total_build_time:.2f}秒 ({total_build_time/60:.2f}分钟)")
    
    print(f"\n❓ 问答测试统计:")
    print(f"  - 测试问题数: {len(results)}")
    
    if results:
        avg_retrieve = sum(r["retrieve_time"] for r in results) / len(results)
        avg_generate = sum(r["generate_time"] for r in results) / len(results)
        avg_total = sum(r["total_time"] for r in results) / len(results)
        
        print(f"  - 平均检索耗时: {avg_retrieve:.2f}秒")
        print(f"  - 平均生成耗时: {avg_generate:.2f}秒")
        print(f"  - 平均总耗时: {avg_total:.2f}秒")
        
        print(f"\n📝 详细结果:")
        for idx, r in enumerate(results, start=1):
            print(f"\n  问题{idx}:")
            print(f"    问题: {r['question'][:60]}...")
            print(f"    检索资料数: {r['sources_count']}")
            print(f"    总耗时: {r['total_time']:.2f}秒")
            print(f"    答案预览: {r['answer'][:100]}..." if r['answer'] else "    答案: 无")
    
    print(f"\n✅ 测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def main():
    # 加载配置
    config = load_config()
    
    # 设置模型路径
    model_path = "/AII-heyan/DeepSeek/DeepSeek-R1-Distill-Qwen-7B"
    
    print("\n" + "="*80)
    print("FUO知识库重建与测试流程")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"配置信息:")
    print(f"  - OpenSearch URL: {config.opensearch_url}")
    print(f"  - OpenSearch Index: {config.opensearch_index_name}")
    print(f"  - Qdrant URL: {config.qdrant_url}")
    print(f"  - Embedding Model: {config.model_name_or_path}")
    print(f"  - LLM Model: {model_path}")
    
    overall_start = time.time()
    
    # 步骤1: 删除旧索引
    delete_existing_indexes(config)
    
    # 步骤2: 构建新索引（1000条数据）
    build_start = time.time()
    build_result = build_new_index(config, sample_size=1000)
    total_build_time = time.time() - build_start
    
    # 步骤3: 测试FUO问题
    results = test_fuo_questions(config, model_path)
    
    # 打印总结
    print_summary(results, build_result, total_build_time)
    
    overall_elapsed = time.time() - overall_start
    print(f"\n⏱️  总耗时: {overall_elapsed:.2f}秒 ({overall_elapsed/60:.2f}分钟)")


if __name__ == "__main__":
    main()
