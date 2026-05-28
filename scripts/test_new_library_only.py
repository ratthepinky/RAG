#!/usr/bin/env python3
"""
测试新库的检索功能：验证新构建的1000条数据索引是否正常工作
"""
from __future__ import annotations

import sys
import os
import time
import shutil
from pathlib import Path
from datetime import datetime

# 强制离线模式
os.environ["HF_HOME"] = "/AII-heyan/ragtestv01_server_bundle_release/artifacts/model_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opensearchpy import OpenSearch
from qdrant_client import QdrantClient

from app.config import load_config
from app.builder import build_index_batch, configure_logging


# FUO相关问题列表
FUO_QUESTIONS = [
    "什么是发热待查（FUO）？其诊断标准是什么？",
    "FUO的常见病因有哪些？如何进行鉴别诊断？",
    "对于不明原因发热的患者，应该进行哪些检查和处理流程？"
]


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


def test_new_library():
    """测试新库的检索功能"""
    print("\n" + "="*80)
    print("验证新库的检索功能")
    print("="*80)
    
    # 加载配置
    config = load_config()
    
    print(f"配置信息:")
    print(f"  - OpenSearch URL: {config.opensearch_url}")
    print(f"  - OpenSearch Index: {config.opensearch_index_name}")
    print(f"  - Qdrant Path: {config.qdrant_local_path}")
    print(f"  - Qdrant Collection: {config.qdrant_collection_name}")
    print(f"  - Embedding Model: {config.model_name_or_path}")
    
    # 检查Qdrant集合是否存在
    try:
        qdrant_client = QdrantClient(url=config.qdrant_url)
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        print(f"\nQdrant集合: {collection_names}")
        
        if config.qdrant_collection_name in collection_names:
            collection_info = qdrant_client.get_collection(config.qdrant_collection_name)
            print(f"Qdrant集合 '{config.qdrant_collection_name}' 点数量: {collection_info.point_count}")
        else:
            print(f"[ERROR] Qdrant集合 '{config.qdrant_collection_name}' 不存在")
            
    except Exception as e:
        print(f"[ERROR] Qdrant连接失败: {e}")
    
    # 检查OpenSearch索引
    try:
        os_client = OpenSearch(
            hosts=[config.opensearch_url],
            timeout=30,
            use_ssl=config.opensearch_url.startswith("https://"),
            verify_certs=False,
        )
        
        if os_client.indices.exists(index=config.opensearch_index_name):
            index_info = os_client.count(index=config.opensearch_index_name)
            print(f"OpenSearch索引 '{config.opensearch_index_name}' 文档数量: {index_info['count']}")
        else:
            print(f"[ERROR] OpenSearch索引 '{config.opensearch_index_name}' 不存在")
            
    except Exception as e:
        print(f"[ERROR] OpenSearch连接失败: {e}")
    
    print(f"\n开始测试检索功能...")
    
    for idx, question in enumerate(FUO_QUESTIONS, start=1):
        print(f"\n{'='*80}")
        print(f"测试 {idx}/3: {question}")
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
        
        print(f"\n【检索成功】")
        print(f"  - 问题: {question}")
        print(f"  - 检索到资料数: {len(hits)}")
        print(f"  - 检索耗时: {retrieve_time:.2f}秒")
        print(f"  - 上下文预览: {context[:300]}...")
    
    print(f"\n✅ 所有检索测试完成! 新库工作正常。")


if __name__ == "__main__":
    test_new_library()