# AII-Heyan Medical RAG

本仓库为 AII-Heyan 医学知识库 RAG 项目的代码交接仓库，包含数据清洗、文本切块、向量建库、OpenSearch 关键词检索、Hybrid Retrieval 融合检索、检索后处理以及 LLM 问答 Demo 相关代码。

## 当前主链路

```text
用户问题
→ bge-m3 query embedding
→ Qdrant Server dense retrieval
→ OpenSearch BM25 sparse retrieval
→ Hybrid Retrieval 融合排序
→ retrieval_postprocess 过滤、去重、轻量重排
→ 拼接 context
→ Kimi / Moonshot LLM 生成中文回答
```

## 重要说明

本仓库只包含代码，不包含以下内容：

* 原始语料
* cleaned_corpus
* Qdrant / OpenSearch 数据库
* embedding 模型缓存
* `.env` 真实密钥
* 构建日志

运行前需要根据 `.env.example` 自行配置环境变量，并准备语料、模型、Qdrant Server 和 OpenSearch。

## 目录结构

```text
app/
  config.py
  stores.py
  chunker.py
  builder.py
  embedder.py
  hybrid_retrieval.py
  retrieval_postprocess.py
  reranker.py

scripts/
  run_build.py
  run_resume.sh
  inspect_chunks.py
  eval_retrieval.py
  run_query_local.py
  test_retrieval_only.py
  test_rag_qa.py

demo_qa.py
run_demo.sh
.env.example
requirements.txt
requirements_freeze.txt
```

## 核心模块

* `app/config.py`：统一配置、环境变量读取、Qdrant / OpenSearch / 切块 / 检索参数。
* `app/stores.py`：Qdrant Server 与 OpenSearch 的写入和查询。
* `app/chunker.py`：对 cleaned corpus 中的文章进行切块。
* `app/embedder.py`：加载 embedding 模型并生成向量。
* `app/builder.py`：执行建库主流程。
* `app/hybrid_retrieval.py`：融合 Qdrant dense retrieval 与 OpenSearch BM25 retrieval。
* `app/retrieval_postprocess.py`：对召回结果进行低价值片段过滤、doc_id 去重和轻量重排。
* `demo_qa.py`：交互式医学问答 Demo。
* `run_demo.sh`：一键启动 Demo 的 shell 脚本。
* `scripts/run_build.py`：单批次建库脚本。
* `scripts/run_resume.sh`：分批续跑建库脚本。

## 环境变量

参考 `.env.example`，主要包括：

```text
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION_NAME=rag_chunks_full
NUM_QDRANT_COLLECTIONS=1

OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_INDEX_NAME=rag_chunks

CLEANED_CORPUS_DIR=/path/to/cleaned_corpus
MODEL_NAME_OR_PATH=/path/to/bge-m3

KIMI_API_KEY=
MOONSHOT_API_KEY=
OPENAI_API_KEY=
```

## 建库输入格式

默认输入目录为 cleaned corpus，每篇文章一个 JSON 文件。建议字段包括：

```json
{
  "doc_id": "PMCxxxx",
  "source_id": "PMCxxxx",
  "source_path": "原始文件路径",
  "source_type": "pmc_zh_translated",
  "title": "文章标题",
  "authors": [],
  "year": "2025",
  "journal": "期刊名",
  "language": "zh",
  "abstract": "摘要",
  "body_text": "正文",
  "keywords": [],
  "section_texts": [],
  "content_hash": "...",
  "quality_flags": [],
  "dedup_group_id": "...",
  "cleaning_notes": []
}
```

## 建库示例

单批建库：

```bash
python scripts/run_build.py \
  --start-offset 0 \
  --sample-size 100 \
  --cleaned-corpus-dir /path/to/cleaned_corpus
```

分批续跑：

```bash
bash scripts/run_resume.sh 0 30000 5000
```

## Demo 启动

```bash
bash run_demo.sh
```

运行前需保证：

1. Qdrant Server 已启动；
2. OpenSearch 已启动；
3. `.env` 或环境变量已配置；
4. embedding 模型路径有效；
5. Kimi / Moonshot API key 已配置。

## 注意事项

* 不要将 `.env`、API key、模型缓存、语料、Qdrant/OpenSearch 数据库提交到 Git。
* 本仓库不包含已经建好的数据库。
* 如果要更换新语料，应先生成新的 cleaned corpus，再新建独立 Qdrant collection 和 OpenSearch index。
* 不建议直接覆盖现有正式库，应采用新库并行方式验证。
