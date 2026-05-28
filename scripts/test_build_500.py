"""
测试建库性能 - 500条数据
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
import gc
import torch
import torch_mlu

from app.config import load_config
from app.chunker import build_chunks_for_document
from app.embedder import encode_texts
from app.stores import (
    upsert_qdrant_points_multi,
    bulk_index_opensearch,
    ensure_qdrant_collections,
    ensure_opensearch_index,
    get_qdrant_collection_counts,
)

gc.collect()
torch.mlu.empty_cache()

config = load_config({'embed_batch_size': 32})
files = sorted(config.cleaned_corpus_dir.glob('*.json'))[:500]
print(f"加载 {len(files)} 个文档...")

# 加载文档
docs = []
for f in files:
    with open(f, 'r', encoding='utf-8') as hf:
        docs.append(json.load(hf))

# Chunking
t0 = time.perf_counter()
chunks = []
for doc in docs:
    chunks.extend(build_chunks_for_document(doc, '', config))
chunk_time = time.perf_counter() - t0
print(f"Chunking: {chunk_time:.2f}s, {len(chunks)} chunks")

# Embedding
t0 = time.perf_counter()
vectors = encode_texts([c['chunk_text'] for c in chunks], config)
embed_time = time.perf_counter() - t0
print(f"Embedding: {embed_time:.2f}s, {len(vectors)} vectors")

# 创建 Collection
ensure_qdrant_collections(config, len(vectors[0]))
ensure_opensearch_index(config)

# 写入 Qdrant
t0 = time.perf_counter()
qdrant_stats = upsert_qdrant_points_multi(config, chunks, vectors)
qdrant_time = time.perf_counter() - t0
qdrant_total = sum(qdrant_stats.values())
print(f"Qdrant写入: {qdrant_time:.2f}s, {qdrant_total} points")

# 写入 OpenSearch
t0 = time.perf_counter()
os_written = bulk_index_opensearch(config, chunks)
os_time = time.perf_counter() - t0
print(f"OpenSearch写入: {os_time:.2f}s, {os_written} docs")

# 统计
counts = get_qdrant_collection_counts(config)
print(f"\nCollection counts: {counts}")

# 估算
total_corpus = len(list(config.cleaned_corpus_dir.glob('*.json')))
chunks_per_doc = len(chunks) / len(files)
total_chunks = int(total_corpus * chunks_per_doc)
estimated_total = (chunk_time + embed_time + qdrant_time + os_time) * (total_corpus / len(files))

print("\n" + "="*60)
print("性能统计")
print("="*60)
print(f"样本: {len(files)} 文档, {len(chunks)} chunks")
print(f"总耗时: {chunk_time + embed_time + qdrant_time + os_time:.2f}s")
print(f"预估速度: {len(files)/(chunk_time + embed_time):.1f} docs/s")
print("="*60)
print(f"语料库总量: {total_corpus} 文档")
print(f"预估总chunks: {total_chunks}")
print(f"预估总耗时: {estimated_total/60:.1f} 分钟 = {estimated_total/3600:.1f} 小时")
