#!/usr/bin/env python3
"""
简单检索测试脚本 - 仅测试 OpenSearch 和 Qdrant 的检索功能，不需要 LLM
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# 强制离线模式，必须在导入任何 huggingface 相关模块之前设置
os.environ["HF_HOME"] = "/AII-heyan/ragtestv01_rebuild/artifacts/model_cache"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from opensearchpy import OpenSearch
from qdrant_client import QdrantClient

from app.config import load_config
from app.embedder import encode_texts


def test_opensearch(config, question: str, top_k: int = 3):
    """测试 OpenSearch sparse 检索"""
    print("\n" + "="*60)
    print("测试 OpenSearch Sparse 检索")
    print("="*60)
    
    try:
        client = OpenSearch(
            hosts=[config.opensearch_url],
            timeout=30,
            use_ssl=config.opensearch_url.startswith("https://"),
            verify_certs=False,
        )
        
        # 检查索引是否存在
        if not client.indices.exists(index=config.opensearch_index_name):
            print(f"[ERROR] 索引 {config.opensearch_index_name} 不存在")
            return False
        
        # 执行检索
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
        
        print(f"[OK] 检索到 {len(hits)} 条结果")
        
        if hits:
            print(f"\n前 {min(3, len(hits))} 条结果:")
            for i, hit in enumerate(hits[:3], start=1):
                src = hit["_source"]
                print(f"\n[{i}] Score: {hit.get('_score', 0):.4f}")
                print(f"    Title: {src.get('title', 'N/A')[:100]}")
                print(f"    Doc ID: {src.get('doc_id', 'N/A')}")
                print(f"    Chunk ID: {src.get('chunk_id', 'N/A')}")
                print(f"    Section: {src.get('section_name', 'N/A')}")
                print(f"    Text Preview: {src.get('chunk_text', '')[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] OpenSearch 检索失败: {e}")
        return False


def test_qdrant(config, question: str, top_k: int = 3):
    """测试 Qdrant dense 向量检索"""
    print("\n" + "="*60)
    print("测试 Qdrant Dense 向量检索")
    print("="*60)
    
    try:
        client = QdrantClient(url=config.qdrant_url)
        
        # 检查集合是否存在
        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]
        
        if config.qdrant_collection_name not in collection_names:
            print(f"[ERROR] 集合 {config.qdrant_collection_name} 不存在")
            print(f"[INFO] 可用集合: {collection_names}")
            return False
        
        # 生成查询向量
        print(f"[INFO] 正在生成问题向量...")
        query_vector = encode_texts([question], config)[0]
        
        # 执行检索
        response = client.query_points(
            collection_name=config.qdrant_collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        
        points = response.points
        print(f"[OK] 检索到 {len(points)} 条结果")
        
        if points:
            print(f"\n前 {min(3, len(points))} 条结果:")
            for i, point in enumerate(points[:3], start=1):
                payload = dict(point.payload or {})
                print(f"\n[{i}] Score: {point.score:.4f}")
                print(f"    Title: {payload.get('title', 'N/A')[:100]}")
                print(f"    Doc ID: {payload.get('doc_id', 'N/A')}")
                print(f"    Chunk ID: {payload.get('chunk_id', 'N/A')}")
                print(f"    Section: {payload.get('section_name', 'N/A')}")
                print(f"    Text Preview: {payload.get('chunk_text', '')[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Qdrant 检索失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="测试 RAG 检索功能（无需 LLM）")
    parser.add_argument("--question", default="什么是深度学习", help="测试问题")
    parser.add_argument("--top-k", type=int, default=3, help="返回结果数量")
    parser.add_argument("--test-opensearch", action="store_true", help="只测试 OpenSearch")
    parser.add_argument("--test-qdrant", action="store_true", help="只测试 Qdrant")
    args = parser.parse_args()
    
    config = load_config()
    
    print(f"[INFO] 配置信息:")
    print(f"  OpenSearch URL: {config.opensearch_url}")
    print(f"  OpenSearch Index: {config.opensearch_index_name}")
    print(f"  Qdrant Path: {config.qdrant_local_path}")
    print(f"  Qdrant Collection: {config.qdrant_collection_name}")
    print(f"  Embedding Model: {config.model_name_or_path}")
    print(f"\n[INFO] 测试问题: {args.question}")
    
    # 如果没有指定测试类型，则都测试
    if not args.test_opensearch and not args.test_qdrant:
        args.test_opensearch = True
        args.test_qdrant = True
    
    results = {}
    
    if args.test_opensearch:
        results["opensearch"] = test_opensearch(config, args.question, args.top_k)
    
    if args.test_qdrant:
        results["qdrant"] = test_qdrant(config, args.question, args.top_k)
    
    # 总结
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    
    all_passed = True
    for service, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {service.upper()}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[SUCCESS] 所有检索测试通过！系统可以正常问答（需要 LLM 支持）。")
    else:
        print("\n[WARNING] 部分检索测试失败，请检查相关服务。")


if __name__ == "__main__":
    main()
