from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import BuildConfig

LOGGER = logging.getLogger(__name__)
EMBED_PROGRESS_EVERY = 20


def _load_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding. Install requirements.txt first."
        ) from exc
    return SentenceTransformer


def _detect_device() -> str:
    import torch  # type: ignore

    try:
        import torch_mlu  # type: ignore  # noqa: F401
    except Exception:
        pass

    if hasattr(torch, "mlu"):
        try:
            if torch.mlu.is_available():
                return "mlu"
        except Exception:
            pass

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


@lru_cache(maxsize=4)
def _load_embedding_model_cached(model_name_or_path: str, device: str):
    SentenceTransformer = _load_sentence_transformer()
    # 强制离线模式，使用本地缓存的模型
    import os
    from pathlib import Path
    
    # 设置 HuggingFace 缓存目录和离线模式
    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / "artifacts" / "model_cache"
    
    # 必须在导入 transformers 之前设置这些环境变量
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    
    # 如果 model_name_or_path 是模型名称而非路径，转换为本地路径
    if not Path(model_name_or_path).exists():
        # 尝试从缓存中查找
        snapshot_dir = cache_dir / "models--BAAI--bge-m3" / "snapshots"
        if snapshot_dir.exists():
            # 获取第一个有效的快照目录
            for item in snapshot_dir.iterdir():
                if item.is_dir() and not item.name.endswith('.bad'):
                    model_name_or_path = str(item)
                    break
    
    return SentenceTransformer(model_name_or_path, device=device, trust_remote_code=True, cache_folder=str(cache_dir))


def load_embedding_model(config: BuildConfig) -> Any:
    model_name_or_path = config.model_name_or_path
    device = _detect_device()
    model = _load_embedding_model_cached(model_name_or_path, device)
    model.max_seq_length = int(config.embed_max_seq_length)
    return model


def encode_texts(texts: list[str], config: BuildConfig) -> list[list[float]]:
    if not texts:
        return []

    model = load_embedding_model(config)
    batch_size = max(1, int(config.embed_batch_size))
    total = len(texts)
    total_batches = (total + batch_size - 1) // batch_size
    all_vectors: list[list[float]] = []

    model_device = getattr(model, "device", getattr(model, "_target_device", "unknown"))
    LOGGER.info(
        "Embedding started | texts=%s batch_size=%s total_batches=%s device=%s",
        total,
        batch_size,
        total_batches,
        model_device,
    )

    for batch_index, start in enumerate(range(0, total, batch_size), start=1):
        batch_texts = texts[start : start + batch_size]
        batch_vectors = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        all_vectors.extend(batch_vectors.tolist())

        done = start + len(batch_texts)
        if (
            total_batches <= EMBED_PROGRESS_EVERY
            or batch_index == total_batches
            or batch_index % EMBED_PROGRESS_EVERY == 0
        ):
            LOGGER.info(
                "Embedding progress | batches=%s/%s texts=%s/%s",
                batch_index,
                total_batches,
                done,
                total,
            )

    return all_vectors
