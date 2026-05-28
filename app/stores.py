"""
Multi-Collection Qdrant Storage Module
支持多Collection分片 + 量化压缩
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from app.config import BuildConfig
from app.retrieval_postprocess import filter_rerank_dedupe_retrieval_hits

LOGGER = logging.getLogger(__name__)
_QDRANT_CLIENT_CACHE: dict[str, Any] = {}
QDRANT_PROGRESS_EVERY = 5
OPENSEARCH_BULK_BATCH_SIZE = 500
OPENSEARCH_PROGRESS_EVERY = 5


def _load_qdrant() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise ImportError("qdrant-client is required for Qdrant writes. Install requirements.txt first.") from exc
    return QdrantClient, models


def _load_opensearch() -> tuple[Any, Any]:
    try:
        from opensearchpy import OpenSearch
        from opensearchpy.helpers import bulk
    except ImportError as exc:
        raise ImportError("opensearch-py is required for OpenSearch writes. Install requirements.txt first.") from exc
    return OpenSearch, bulk


def _build_qdrant_client(config: BuildConfig) -> Any:
    QdrantClient, _ = _load_qdrant()
    key = str(config.qdrant_url)
    client = _QDRANT_CLIENT_CACHE.get(key)
    if client is None:
        client = QdrantClient(url=config.qdrant_url)
        _QDRANT_CLIENT_CACHE[key] = client
    return client


def _build_opensearch_client(config: BuildConfig) -> Any:
    OpenSearch, _ = _load_opensearch()
    return OpenSearch(
        hosts=[config.opensearch_url],
        timeout=30,
        use_ssl=config.opensearch_url.startswith("https://"),
        verify_certs=False,
    )


def _make_qdrant_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def ensure_qdrant_collections(config: BuildConfig, vector_size: int) -> dict[str, Any]:
    """
    创建多个 Qdrant Collection（分片）
    每个 Collection 使用 INT8 量化压缩
    """
    _, models = _load_qdrant()
    client = _build_qdrant_client(config)
    collection_names = config.get_collection_names()

    created_collections = []
    existing_collections = []

    for coll_name in collection_names:
        collections = client.get_collections().collections
        existing_names = {c.name for c in collections}

        if coll_name in existing_names:
            existing_collections.append(coll_name)
            continue

        # 构建向量配置
        vectors_config = models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        )

        # 添加量化配置（可选）
        if config.use_quantization:
            vectors_config.quantization_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                )
            )

        client.create_collection(
            collection_name=coll_name,
            vectors_config=vectors_config,
        )
        created_collections.append(coll_name)
        LOGGER.info(f"Created Qdrant collection: {coll_name} (quantized={config.use_quantization})")

    return {
        "created": created_collections,
        "existing": existing_collections,
        "vector_size": vector_size,
    }


def delete_all_qdrant_collections(config: BuildConfig) -> None:
    """删除所有 Qdrant Collection"""
    _, models = _load_qdrant()
    client = _build_qdrant_client(config)

    for coll_name in config.get_collection_names():
        try:
            client.delete_collection(collection_name=coll_name)
            LOGGER.info(f"Deleted Qdrant collection: {coll_name}")
        except Exception as e:
            LOGGER.warning(f"Failed to delete {coll_name}: {e}")


def upsert_qdrant_points_multi(
    config: BuildConfig,
    chunk_records: list[dict[str, Any]],
    vectors: list[list[float]]
) -> dict[str, int]:
    """
    向多个 Qdrant Collection 写入数据
    根据 doc_id 路由到不同的 Collection
    """
    if len(chunk_records) != len(vectors):
        raise ValueError("chunk_records and vectors length mismatch")

    _, models = _load_qdrant()
    client = _build_qdrant_client(config)

    # 按 collection 分组
    collections_data: dict[str, tuple[list, list]] = {}

    for record, vector in zip(chunk_records, vectors):
        doc_id = record.get("doc_id", "")
        coll_name = config.get_collection_for_doc(doc_id)

        if coll_name not in collections_data:
            collections_data[coll_name] = ([], [])

        collections_data[coll_name][0].append(record)
        collections_data[coll_name][1].append(vector)

    # 向每个 collection 写入
    total_written = 0
    collection_stats = {}

    for coll_name, (records, vecs) in collections_data.items():
        if not records:
            continue

        batch_size = max(1, config.qdrant_upsert_batch_size)
        written = 0

        for start in range(0, len(records), batch_size):
            record_batch = records[start:start + batch_size]
            vector_batch = vecs[start:start + batch_size]

            points = [
                models.PointStruct(
                    id=_make_qdrant_point_id(str(record["chunk_id"])),
                    vector=vector,
                    payload=_slim_payload(record),
                )
                for record, vector in zip(record_batch, vector_batch)
            ]

            client.upsert(collection_name=coll_name, points=points, wait=True)
            written += len(points)

        collection_stats[coll_name] = written
        total_written += written

        LOGGER.info(f"Upserted {written} points to {coll_name}")

    return collection_stats


def _slim_payload(record: dict[str, Any]) -> dict[str, Any]:
    """
    精简 payload，只保留检索必需字段
    减少存储空间
    """
    return {
        "chunk_id": record.get("chunk_id", ""),
        "doc_id": record.get("doc_id", ""),
        "chunk_text": record.get("chunk_text", ""),
        "title": record.get("title", ""),
        "section_name": record.get("section_name", ""),
        "journal": record.get("journal", ""),
        "year": record.get("year", ""),
        "chunk_type": record.get("chunk_type", ""),
    }


def query_qdrant_single_collection(
    config: BuildConfig,
    collection_name: str,
    query_vector: list[float],
    limit: int = 15
) -> list[dict[str, Any]]:
    """查询单个 Qdrant Collection (支持本地和服务端模式)"""
    _, models = _load_qdrant()
    client = _build_qdrant_client(config)

    try:
        # 本地模式使用 query_points
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        hits = []
        for result in results.points:
            payload = result.payload or {}
            payload["score"] = result.score
            hits.append(payload)

        return hits
    except Exception as e:
        LOGGER.warning(f"Query failed for {collection_name}: {e}")
        return []


def query_qdrant_multi_collection(
    config: BuildConfig,
    query_vector: list[float],
    limit_per_coll: int = 15
) -> list[dict[str, Any]]:
    """并行查询所有 Qdrant Collection"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    collection_names = config.get_collection_names()
    all_results = []

    with ThreadPoolExecutor(max_workers=config.num_qdrant_collections) as executor:
        futures = {
            executor.submit(query_qdrant_single_collection, config, coll, query_vector, limit_per_coll): coll
            for coll in collection_names
        }

        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                LOGGER.warning(f"Collection query failed: {e}")

    all_results.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)

    # 正式召回后处理：
    # 1. 过滤参考文献等低价值 chunk
    # 2. 按 doc_id 去重，避免同一文档 summary/body 重复挤占上下文
    # 3. 使用轻量关键词 bonus 调整排序
    return filter_rerank_dedupe_retrieval_hits(all_results)


# ============= OpenSearch 相关 =============


def ensure_opensearch_index(config: BuildConfig) -> dict[str, Any]:
    """确保 OpenSearch 索引存在（单索引，不分片）"""
    client = _build_opensearch_client(config)

    if client.indices.exists(index=config.opensearch_index_name):
        LOGGER.info(f"OpenSearch index already exists: {config.opensearch_index_name}")
        return {"index_name": config.opensearch_index_name, "created": False}

    # 构建索引配置（不使用 IK 分词器，兼容默认安装）
    index_settings = {
        "index": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        }
    }

    mapping = {
        "settings": index_settings,
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_type": {"type": "keyword"},
                "section_name": {"type": "keyword"},
                "chunk_text": {"type": "text"},
                "abstract": {"type": "text"},
                "title": {"type": "text"},
                "journal": {"type": "text"},
                "year": {"type": "keyword"},
                "source_type": {"type": "keyword"},
            }
        },
    }

    client.indices.create(index=config.opensearch_index_name, body=mapping)
    LOGGER.info(f"Created OpenSearch index: {config.opensearch_index_name}")
    return {"index_name": config.opensearch_index_name, "created": True}


def delete_opensearch_index(config: BuildConfig) -> None:
    """删除 OpenSearch 索引"""
    client = _build_opensearch_client(config)
    try:
        if client.indices.exists(index=config.opensearch_index_name):
            client.indices.delete(index=config.opensearch_index_name)
            LOGGER.info(f"Deleted OpenSearch index: {config.opensearch_index_name}")
    except Exception as e:
        LOGGER.warning(f"Failed to delete OpenSearch index: {e}")


def bulk_index_opensearch(config: BuildConfig, chunk_records: list[dict[str, Any]]) -> int:
    """批量索引到 OpenSearch"""
    client = _build_opensearch_client(config)
    _, bulk = _load_opensearch()

    actions = [
        {
            "_index": config.opensearch_index_name,
            "_id": record["chunk_id"],
            "_source": record,
        }
        for record in chunk_records
    ]

    if not actions:
        return 0

    total_written = 0
    total_batches = (len(actions) + OPENSEARCH_BULK_BATCH_SIZE - 1) // OPENSEARCH_BULK_BATCH_SIZE

    LOGGER.info(
        f"OpenSearch bulk started | docs={len(actions)} batch_size={OPENSEARCH_BULK_BATCH_SIZE} total_batches={total_batches}"
    )

    for batch_index, start in enumerate(range(0, len(actions), OPENSEARCH_BULK_BATCH_SIZE), start=1):
        batch_actions = actions[start:start + OPENSEARCH_BULK_BATCH_SIZE]
        bulk(client, batch_actions)
        total_written += len(batch_actions)

        if (
            total_batches <= OPENSEARCH_PROGRESS_EVERY
            or batch_index == total_batches
            or batch_index % OPENSEARCH_PROGRESS_EVERY == 0
        ):
            LOGGER.info(
                f"OpenSearch bulk progress | batches={batch_index}/{total_batches} docs={total_written}/{len(actions)}"
            )

    client.indices.refresh(index=config.opensearch_index_name)
    return total_written


def query_sparse(config: BuildConfig, question: str, limit: int = 30) -> list[dict[str, Any]]:
    """Sparse 检索（参考 querying.py 的分词查询方式）"""
    import re
    import jieba

    client = _build_opensearch_client(config)

    # 分词处理（参考 querying.py 的 tokenize_for_search）
    normalized = question.strip().lower()
    coarse_tokens = re.findall(r"[\w\u4e00-\u9fff]+", normalized)
    fine_tokens = []
    for token in coarse_tokens:
        if re.search(r"[\u4e00-\u9fff]", token):
            fine_tokens.extend(part.strip() for part in jieba.cut(token) if part.strip())
        else:
            fine_tokens.append(token)
    
    # 去重
    deduped = []
    seen = set()
    for token in fine_tokens:
        if len(token) <= 1 and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    
    query_text = " ".join(deduped)
    LOGGER.info(f"Sparse query (tokenized): {query_text}")

    query_body = {
        "size": limit,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "title^3",
                    "chunk_text^2",
                    "section_name^1.5",
                    "journal",
                ],
                "type": "best_fields",
                "operator": "or",
            }
        },
        "_source": ["chunk_id", "doc_id", "chunk_text", "title", "section_name", "journal", "year", "chunk_type"]
    }

    try:
        response = client.search(index=config.opensearch_index_name, body=query_body)
        hits = []

        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            source["score"] = hit["_score"]
            hits.append(source)

        LOGGER.info(f"Sparse query returned {len(hits)} hits")
        return hits

    except Exception as e:
        LOGGER.warning(f"Sparse query failed: {e}")
        return []


def get_chunks_by_ids(config: BuildConfig, chunk_ids: list[str]) -> dict[str, dict]:
    """根据 chunk_id 批量获取完整文档"""
    client = _build_opensearch_client(config)

    if not chunk_ids:
        return {}

    query_body = {
        "query": {
            "terms": {"chunk_id": chunk_ids}
        },
        "size": len(chunk_ids)
    }

    try:
        response = client.search(index=config.opensearch_index_name, body=query_body)
        result = {}

        for hit in response["hits"]["hits"]:
            chunk_id = hit["_source"]["chunk_id"]
            result[chunk_id] = hit["_source"]

        return result

    except Exception as e:
        LOGGER.warning(f"Failed to get chunks by ids: {e}")
        return {}


def get_qdrant_storage_size(config: BuildConfig) -> float:
    """获取 Qdrant 存储大小（GB）"""
    import shutil
    from pathlib import Path

    qdrant_path = Path(config.qdrant_local_path)
    if not qdrant_path.exists():
        return 0.0

    total_size = sum(f.stat().st_size for f in qdrant_path.rglob('*') if f.is_file())
    return total_size / (1024 ** 3)


def get_qdrant_collection_counts(config: BuildConfig) -> dict[str, int]:
    """获取每个 Collection 的文档数量"""
    _, models = _load_qdrant()
    client = _build_qdrant_client(config)

    counts = {}
    for coll_name in config.get_collection_names():
        try:
            info = client.get_collection(coll_name)
            counts[coll_name] = info.points_count
        except Exception:
            counts[coll_name] = 0

    return counts
