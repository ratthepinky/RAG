# ragtestv01_rebuild

极简、稳定优先的学术文献 RAG 建库项目。它只做这几件事：读取清洗后的 JSON、按结构优先分块、生成 embedding、写入本地 embedded Qdrant 和 OpenSearch，并支持断点续跑。

## 目录
```text
ragtestv01_rebuild/
  app/
    config.py
    chunker.py
    embedder.py
    stores.py
    builder.py
  scripts/
    inspect_chunks.py
    run_build.py
    run_resume.sh
  artifacts/
    qdrant_local/
    resume_state/
  logs/
  cleaned_corpus/
  README.md
  requirements.txt
```

## 关键文件
- `app/config.py`: 默认值、环境变量和 CLI 覆盖。
- `app/chunker.py`: summary + section 优先分块，`body_text` 只在块数不足时兜底。
- `app/embedder.py`: 延迟加载 embedding 模型，显式控制 `EMBED_BATCH_SIZE` 和 `EMBED_MAX_SEQ_LENGTH`。
- `app/stores.py`: embedded Qdrant 和 OpenSearch 的创建与批量写入。
- `app/builder.py`: 主流程和 state 写入。
- `scripts/inspect_chunks.py`: 单篇文档分块预览。
- `scripts/run_build.py`: 单批建库。
- `scripts/run_resume.sh`: 区间续跑，失败缩小批次且不回弹。

## 默认行为
- `sample_size=500`
- `start_offset=0`
- `end_offset=168000`
- `chunk_size=1200`
- `overlap=0`
- `max_chunks_per_doc=8`
- `max_section_splits=3`
- `min_paragraph_len=40`
- `EMBED_BATCH_SIZE=1`
- `EMBED_MAX_SEQ_LENGTH=1024`
- `QdrantClient(path="<project>/artifacts/qdrant_local")`

## 安装
服务器上建议使用目标 Python：

```bash
/torch/venv3/pytorch/bin/python -m pip install -r requirements.txt
```

本地如果只想预览分块，`inspect_chunks.py` 不依赖 Qdrant、OpenSearch、torch 或 sentence-transformers。

## 环境变量
- `CLEANED_CORPUS_DIR`
- `ARTIFACTS_DIR`
- `LOGS_DIR`
- `MODEL_NAME_OR_PATH`
- `QDRANT_LOCAL_PATH`
- `QDRANT_COLLECTION_NAME`
- `OPENSEARCH_URL`
- `OPENSEARCH_INDEX_NAME`
- `EMBED_BATCH_SIZE`
- `EMBED_MAX_SEQ_LENGTH`
- `CHUNK_SIZE`
- `OVERLAP`
- `MAX_CHUNKS_PER_DOC`
- `MAX_SECTION_SPLITS`
- `MIN_PARAGRAPH_LEN`
- `QDRANT_UPSERT_BATCH_SIZE`
- `SAMPLE_SIZE`
- `START_OFFSET`
- `END_OFFSET`

## 运行命令
分块预览：

```bash
/torch/venv3/pytorch/bin/python scripts/inspect_chunks.py --file /path/to/sample.json
```

单批建库：

```bash
EMBED_BATCH_SIZE=1 EMBED_MAX_SEQ_LENGTH=1024 /torch/venv3/pytorch/bin/python scripts/run_build.py --sample-size 500 --start-offset 0
```

自动续跑：

```bash
bash scripts/run_resume.sh 0 168000 500
```

## 日志和状态
- 单批日志默认写到 `logs/run_build_<start_offset>_<sample_size>.log`
- 续跑日志写到 `logs/build_resume_<START_OFFSET>_<END_OFFSET>.log`
- 状态文件写到 `artifacts/resume_state/build_resume_<START_OFFSET>_<END_OFFSET>.state`

state 文件字段：
- `current_offset`
- `current_sample_size`
- `last_success_docs`
- `last_success_chunks`
- `updated_at`

## 建库策略
- 每篇文档优先生成 1 个 `summary` chunk，内容来自 `title + abstract + keywords`
- `section_texts` 是主索引来源
- 单个 section 最多拆成 3 块
- 每篇文档最多 8 块
- 只有在 `summary + sections < 2` 时才使用 `body_text` 兜底
- 过滤 `Fig.`、`Figure`、`Table`、`Supplementary Information`、`Peer Review File` 和明显重复段落

## 首轮测试建议
先在服务器跑最保守 smoke test：

```bash
/torch/venv3/pytorch/bin/python scripts/inspect_chunks.py --file /path/to/sample.json
EMBED_BATCH_SIZE=1 EMBED_MAX_SEQ_LENGTH=1024 /torch/venv3/pytorch/bin/python scripts/run_build.py --sample-size 3 --start-offset 0
```

如果稳定，再逐步扩大：

```bash
/torch/venv3/pytorch/bin/python scripts/run_build.py --sample-size 10 --start-offset 0
/torch/venv3/pytorch/bin/python scripts/run_build.py --sample-size 25 --start-offset 0
/torch/venv3/pytorch/bin/python scripts/run_build.py --sample-size 100 --start-offset 0
/torch/venv3/pytorch/bin/python scripts/run_build.py --sample-size 500 --start-offset 0
```

本地验证只能覆盖分块逻辑、CLI 和流程结构，不能替代服务器上的 MLU 显存验证。
