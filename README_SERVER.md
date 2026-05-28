# RAG Server Bundle

This bundle is prepared for direct upload to your server.

## What Is Improved

- Multi-collection dense retrieval is globally sorted before RRF.
- Query scripts use config top-k values instead of hard-coded `5/5/10/6/3`.
- Reranker keeps both the head and tail of long chunks before scoring.
- Sparse retrieval also uses `section_name`.
- Body fallback chunks are added more often to improve coverage.
- Default retrieval parameters are tuned upward for better recall.
- `jieba` and `openai` are included in `requirements.txt`.
- Added `scripts/eval_retrieval.py` for stage-by-stage evaluation.

## Recommended Upload Layout

Upload this whole folder as one directory, for example:

```bash
/data/ragtestv01_server_bundle
```

Recommended server layout:

```text
ragtestv01_server_bundle/
  app/
  scripts/
  artifacts/
    model_cache/
  cleaned_corpus/
  logs/
  requirements.txt
  README_SERVER.md
```

If you place your data in `cleaned_corpus/` and your Hugging Face cache in `artifacts/model_cache/`, the bundle works out of the box without setting `CLEANED_CORPUS_DIR` or model cache paths.

## Install

```bash
cd /data/ragtestv01_server_bundle
python -m pip install -r requirements.txt
```

## Required Environment Variables

Minimal example when data and cache are already placed inside the bundle:

```bash
cd /data/ragtestv01_server_bundle
export OPENSEARCH_URL=http://localhost:9200
export OPENSEARCH_INDEX_NAME=rag_chunks
export KIMI_API_KEY=your_api_key
```

If your data is outside the bundle, then additionally set:

```bash
export CLEANED_CORPUS_DIR=/data/cleaned_corpus
```

Recommended retrieval settings:

```bash
export QUERY_DENSE_TOP_K=30
export QUERY_SPARSE_TOP_K=30
export QUERY_FUSED_TOP_K=60
export RERANK_TOP_K=20
export QUERY_FINAL_TOP_K=6
export MAX_CONTEXT_CHARS=8000
```

Recommended indexing settings:

```bash
export CHUNK_SIZE=1600
export MAX_CHUNKS_PER_DOC=12
export MAX_SECTION_SPLITS=6
export MIN_PARAGRAPH_LEN=40
export EMBED_MAX_SEQ_LENGTH=1024
```

## Build Index

Small smoke test:

```bash
python scripts/run_build.py --sample-size 100 --start-offset 0
```

Larger build:

```bash
python scripts/run_build.py --sample-size 3000 --start-offset 0
```

If your corpus is large, continue by batches with different `start-offset`.

## Query

Interactive:

```bash
python scripts/run_query_local.py
```

Single question:

```bash
python scripts/run_query_local.py --question "不明原因发热需要做哪些检查"
```

API mode:

```bash
python scripts/rag_client.py
```

## Evaluate Retrieval

The included `eval_queries.json` is only a lightweight keyword-based smoke test.

Run:

```bash
python scripts/eval_retrieval.py
```

This prints metrics for:

- `dense`
- `sparse`
- `fused`
- `rerank`

and writes `eval_result.json`.

## Important Note

This bundle should perform noticeably better than your current server setup, but final quality still depends on:

- how many documents you index
- whether the corpus has encoding noise
- whether OpenSearch is healthy
- whether your evaluation set is real ground truth instead of keyword matching

If you want the best final effect, rebuild the index with this bundle instead of only replacing the query scripts.
