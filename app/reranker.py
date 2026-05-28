"""
Reranker Module - 使用 bge-reranker-v2-m3 对候选结果进行精排
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.config import BuildConfig

LOGGER = logging.getLogger(__name__)
RERANK_TEXT_HEAD_CHARS = 900
RERANK_TEXT_TAIL_CHARS = 300


def _detect_device() -> str:
    import torch

    try:
        import torch_mlu
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


def _setup_offline_mode():
    """设置离线模式，强制使用本地缓存模型"""
    import os
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / "artifacts" / "model_cache"

    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def _resolve_model_path(model_name_or_path: str) -> str:
    """解析模型路径，优先使用本地缓存"""
    from pathlib import Path

    path = Path(model_name_or_path)
    if path.exists():
        return model_name_or_path

    project_root = Path(__file__).resolve().parents[1]
    cache_dir = project_root / "artifacts" / "model_cache"

    # 尝试从缓存中查找
    model_dir = cache_dir / f"models--{model_name_or_path.replace('/', '--')}" / "snapshots"
    if model_dir.exists():
        for item in model_dir.iterdir():
            if item.is_dir() and not item.name.endswith('.bad'):
                return str(item)

    return model_name_or_path


@lru_cache(maxsize=2)
def load_reranker_model(model_name_or_path: str, device: str):
    """加载 Reranker 模型（带缓存）"""
    _setup_offline_mode()

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for reranker. Install requirements.txt first."
        ) from exc

    resolved_path = _resolve_model_path(model_name_or_path)
    LOGGER.info(f"Loading reranker model: {resolved_path} on {device}")

    model = CrossEncoder(
        resolved_path,
        device=device,
        max_length=512,
        trust_remote_code=True,
    )

    LOGGER.info("Reranker model loaded successfully")
    return model


class Reranker:
    """Reranker 封装类"""

    def __init__(self, config: BuildConfig | None = None):
        self.config = config or BuildConfig.__new__(BuildConfig)
        self._model = None
        self._device = None

    def _ensure_model(self):
        if self._model is None:
            self._device = _detect_device()
            self._model = load_reranker_model(self.config.reranker_model_or_path, self._device)
        return self._model

    def _prepare_document_text(self, candidate: dict[str, Any]) -> str:
        text = str(candidate.get("chunk_text", "") or "")
        if len(text) <= RERANK_TEXT_HEAD_CHARS + RERANK_TEXT_TAIL_CHARS:
            return text

        head = text[:RERANK_TEXT_HEAD_CHARS].rstrip()
        tail = text[-RERANK_TEXT_TAIL_CHARS:].lstrip()
        return f"{head}\n...\n{tail}"

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        """
        对候选结果进行精排

        Args:
            query: 用户问题
            candidates: 候选结果列表，每个元素包含 'chunk_text' 字段
            top_k: 返回前top_k个结果，默认使用配置的 reranker_top_k

        Returns:
            按相关性分数降序排列的结果列表
        """
        if not candidates:
            return []

        if top_k is None:
            top_k = self.config.reranker_top_k

        model = self._ensure_model()

        # 构建 (query, document) 对
        pairs = [(query, self._prepare_document_text(cand)) for cand in candidates]

        # 批量推理
        LOGGER.info(f"Reranking {len(pairs)} candidates on {self._device}")
        scores = model.predict(pairs, batch_size=32)

        # 将分数添加到候选结果中
        for cand, score in zip(candidates, scores):
            cand["rerank_score"] = float(score)

        # 按分数降序排列
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        LOGGER.info(f"Reranking done, top-{top_k} scores: {[c['rerank_score'] for c in reranked[:top_k]]}")

        return reranked[:top_k]

    def rerank_with_scores(self, query: str, candidates: list[dict[str, Any]], top_k: int | None = None) -> tuple[list[dict[str, Any]], list[float]]:
        """返回排序结果和对应的分数"""
        if not candidates:
            return [], []

        if top_k is None:
            top_k = self.config.reranker_top_k

        model = self._ensure_model()
        pairs = [(query, self._prepare_document_text(cand)) for cand in candidates]
        scores = model.predict(pairs, batch_size=32).tolist()

        # 组合并排序
        scored = [(cand, float(score)) for cand, score in zip(candidates, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        result_scores = []
        for cand, score in scored[:top_k]:
            cand["rerank_score"] = score
            results.append(cand)
            result_scores.append(score)

        return results, result_scores


# 全局实例（用于 CLI）
_global_reranker: Reranker | None = None


def get_global_reranker(config: BuildConfig | None = None) -> Reranker:
    """获取全局 Reranker 实例"""
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = Reranker(config)
    return _global_reranker
