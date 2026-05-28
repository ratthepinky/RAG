from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(slots=True)
class BuildConfig:
    # === 基础路径 ===
    project_root: Path
    cleaned_corpus_dir: Path
    artifacts_dir: Path
    logs_dir: Path

    # === 模型配置 ===
    model_name_or_path: str

    # === Qdrant 配置 ===
    qdrant_local_path: Path
    qdrant_url: str
    qdrant_collection_name: str
    qdrant_upsert_batch_size: int

    # === OpenSearch 配置 ===
    opensearch_url: str
    opensearch_index_name: str

    # === Embedding 配置 ===
    embed_batch_size: int
    embed_max_seq_length: int

    # === Chunking 配置 ===
    chunk_size: int
    overlap: int
    max_chunks_per_doc: int
    max_section_splits: int
    min_paragraph_len: int

    # === 建库范围 ===
    sample_size: int
    start_offset: int
    end_offset: int

    # === 多Collection配置 ===
    num_qdrant_collections: int = 5
    use_quantization: bool = True

    # === Reranker配置 ===
    reranker_model_or_path: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 20

    # === 查询参数 ===
    query_dense_top_k: int = 30
    query_sparse_top_k: int = 30
    query_fused_top_k: int = 60
    query_final_top_k: int = 6
    max_context_chars: int = 8000

    # === 建库控制 ===
    full_build_batch_size: int = 2000
    qdrant_warning_size_gb: float = 8.0

    # === OpenSearch配置 ===
    os_use_ik_analyzer: bool = True

    def ensure_runtime_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.artifacts_dir / "resume_state").mkdir(parents=True, exist_ok=True)
        self.qdrant_local_path.mkdir(parents=True, exist_ok=True)

    def get_collection_names(self) -> list[str]:
        """获取所有Qdrant collection名称"""
        if self.num_qdrant_collections == 1:
            return [self.qdrant_collection_name]
        return [f"{self.qdrant_collection_name}_{i}" for i in range(self.num_qdrant_collections)]

    def get_collection_for_doc(self, doc_id: str) -> str:
        """根据doc_id路由到对应的collection"""
        if self.num_qdrant_collections == 1:
            return self.qdrant_collection_name
        import hashlib
        hash_val = int(hashlib.md5(doc_id.encode()).hexdigest(), 16)
        idx = hash_val % self.num_qdrant_collections
        return f"{self.qdrant_collection_name}_{idx}"


ENV_MAP = {
    "cleaned_corpus_dir": "CLEANED_CORPUS_DIR",
    "artifacts_dir": "ARTIFACTS_DIR",
    "logs_dir": "LOGS_DIR",
    "model_name_or_path": "MODEL_NAME_OR_PATH",
    "qdrant_local_path": "QDRANT_LOCAL_PATH",
    "qdrant_url": "QDRANT_URL",
    "qdrant_collection_name": "QDRANT_COLLECTION_NAME",
    "opensearch_url": "OPENSEARCH_URL",
    "opensearch_index_name": "OPENSEARCH_INDEX_NAME",
    "embed_batch_size": "EMBED_BATCH_SIZE",
    "embed_max_seq_length": "EMBED_MAX_SEQ_LENGTH",
    "chunk_size": "CHUNK_SIZE",
    "overlap": "OVERLAP",
    "max_chunks_per_doc": "MAX_CHUNKS_PER_DOC",
    "max_section_splits": "MAX_SECTION_SPLITS",
    "min_paragraph_len": "MIN_PARAGRAPH_LEN",
    "qdrant_upsert_batch_size": "QDRANT_UPSERT_BATCH_SIZE",
    "sample_size": "SAMPLE_SIZE",
    "start_offset": "START_OFFSET",
    "end_offset": "END_OFFSET",
    "num_qdrant_collections": "NUM_QDRANT_COLLECTIONS",
    "use_quantization": "USE_QUANTIZATION",
    "reranker_model_or_path": "RERANKER_MODEL_OR_PATH",
    "reranker_top_k": "RERANKER_TOP_K",
    "query_dense_top_k": "QUERY_DENSE_TOP_K",
    "query_sparse_top_k": "QUERY_SPARSE_TOP_K",
    "query_fused_top_k": "QUERY_FUSED_TOP_K",
    "query_final_top_k": "QUERY_FINAL_TOP_K",
    "max_context_chars": "MAX_CONTEXT_CHARS",
    "full_build_batch_size": "FULL_BUILD_BATCH_SIZE",
    "qdrant_warning_size_gb": "QDRANT_WARNING_SIZE_GB",
    "os_use_ik_analyzer": "OS_USE_IK_ANALYZER",
}


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _pick_value(name: str, default: Any, overrides: Mapping[str, Any]) -> Any:
    override = overrides.get(name)
    if override is not None:
        return override
    env_name = ENV_MAP.get(name)
    if env_name and env_name in os.environ:
        return os.environ[env_name]
    return default


def _int_value(name: str, default: int, overrides: Mapping[str, Any]) -> int:
    return int(_pick_value(name, default, overrides))


def _bool_value(name: str, default: bool, overrides: Mapping[str, Any]) -> bool:
    val = _pick_value(name, default, overrides)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes", "on")
    return bool(val)


def _float_value(name: str, default: float, overrides: Mapping[str, Any]) -> float:
    val = _pick_value(name, default, overrides)
    if isinstance(val, float):
        return val
    return float(val)


def _str_value(name: str, default: str, overrides: Mapping[str, Any]) -> str:
    return str(_pick_value(name, default, overrides))


def _path_value(name: str, default: Path, overrides: Mapping[str, Any]) -> Path:
    return _resolve_path(_pick_value(name, str(default), overrides))


def load_config(cli_args: Mapping[str, Any] | None = None) -> BuildConfig:
    overrides = cli_args or {}
    project_root = Path(__file__).resolve().parents[1]
    artifacts_dir = _path_value("artifacts_dir", project_root / "artifacts", overrides)

    config = BuildConfig(
        project_root=project_root,
        cleaned_corpus_dir=_path_value("cleaned_corpus_dir", project_root / "cleaned_corpus", overrides),
        artifacts_dir=artifacts_dir,
        logs_dir=_path_value("logs_dir", project_root / "logs", overrides),
        model_name_or_path=_str_value("model_name_or_path", "BAAI/bge-m3", overrides),
        qdrant_local_path=_path_value("qdrant_local_path", artifacts_dir / "qdrant_local", overrides),
        qdrant_url=_str_value("qdrant_url", "http://127.0.0.1:6333", overrides),
        qdrant_collection_name=_str_value("qdrant_collection_name", "rag_chunks_full", overrides),
        opensearch_url=_str_value("opensearch_url", "http://localhost:9200", overrides),
        opensearch_index_name=_str_value("opensearch_index_name", "rag_chunks", overrides),
        embed_batch_size=_int_value("embed_batch_size", 16, overrides),
        embed_max_seq_length=_int_value("embed_max_seq_length", 1024, overrides),
        chunk_size=_int_value("chunk_size", 1600, overrides),
        overlap=_int_value("overlap", 150, overrides),
        max_chunks_per_doc=_int_value("max_chunks_per_doc", 12, overrides),
        max_section_splits=_int_value("max_section_splits", 6, overrides),
        min_paragraph_len=_int_value("min_paragraph_len", 40, overrides),
        qdrant_upsert_batch_size=_int_value("qdrant_upsert_batch_size", 128, overrides),
        sample_size=_int_value("sample_size", 1000, overrides),
        start_offset=_int_value("start_offset", 0, overrides),
        end_offset=_int_value("end_offset", 168000, overrides),
        num_qdrant_collections=_int_value("num_qdrant_collections", 1, overrides),
        use_quantization=_bool_value("use_quantization", True, overrides),
        reranker_model_or_path=_str_value("reranker_model_or_path", "BAAI/bge-reranker-v2-m3", overrides),
        reranker_top_k=_int_value("reranker_top_k", 20, overrides),
        query_dense_top_k=_int_value("query_dense_top_k", 30, overrides),
        query_sparse_top_k=_int_value("query_sparse_top_k", 30, overrides),
        query_fused_top_k=_int_value("query_fused_top_k", 60, overrides),
        query_final_top_k=_int_value("query_final_top_k", 6, overrides),
        max_context_chars=_int_value("max_context_chars", 8000, overrides),
        full_build_batch_size=_int_value("full_build_batch_size", 2000, overrides),
        qdrant_warning_size_gb=_float_value("qdrant_warning_size_gb", 8.0, overrides),
        os_use_ik_analyzer=_bool_value("os_use_ik_analyzer", True, overrides),
    )
    config.ensure_runtime_dirs()
    return config
