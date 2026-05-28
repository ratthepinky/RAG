from __future__ import annotations

from typing import Any


def is_low_value_retrieval_hit(hit: dict[str, Any]) -> bool:
    """过滤参考文献、空文本、明显低价值 chunk。"""
    title = (hit.get("title") or "").strip().lower()
    section = (hit.get("section_name") or "").strip().lower()
    text = (
        hit.get("chunk_text")
        or hit.get("text")
        or hit.get("content")
        or ""
    ).strip()
    text_lower = text.lower()

    bad_keywords = [
        "参考文献",
        "references",
        "bibliography",
        "参考资料",
        "致谢",
        "acknowledgement",
        "acknowledgment",
    ]

    if any(k in title for k in bad_keywords):
        return True

    if any(k in section for k in bad_keywords):
        return True

    if len(text) < 80:
        return True

    # 参考文献型 chunk 常见特征：大量 doi / et al，通常不适合作为 LLM 上下文。
    if text_lower.count("doi") >= 2 or text_lower.count(" et al") >= 3:
        return True

    return False


def retrieval_relevance_bonus(hit: dict[str, Any]) -> int:
    """给诊断、病因、治疗、FUO 等任务相关 chunk 一个轻量 bonus。"""
    title = hit.get("title") or ""
    section = hit.get("section_name") or ""
    text = (
        hit.get("chunk_text")
        or hit.get("text")
        or hit.get("content")
        or ""
    )
    blob = f"{title} {section} {text}".lower()

    keywords = [
        "不明原因发热",
        "原因不明的发热",
        "发热待查",
        "反复发热",
        "间断发热",
        "长期发热",
        "fever of unknown origin",
        "fuo",
        "诊断",
        "筛查",
        "病因",
        "治疗",
        "处理",
        "检查",
        "结核",
        "感染",
        "肿瘤",
        "药物热",
        "自身免疫",
        "风湿",
        "高血压",
        "管理",
        "指南",
        "recommendation",
        "guideline",
    ]

    score = 0
    for kw in keywords:
        if kw.lower() in blob:
            score += 1

    return score


def filter_rerank_dedupe_retrieval_hits(
    hits: list[dict[str, Any]],
    *,
    dedupe_by_doc: bool = True,
) -> list[dict[str, Any]]:
    """
    正式召回后处理：
    1. 过滤参考文献/低价值 chunk
    2. 对任务相关 chunk 加轻量 bonus
    3. 按 doc_id 去重，避免同一文档 summary/body 重复挤占上下文
    """
    candidates: list[dict[str, Any]] = []

    for hit in hits:
        if is_low_value_retrieval_hit(hit):
            continue

        base_score = float(hit.get("score", 0.0) or 0.0)
        bonus = retrieval_relevance_bonus(hit)

        # bonus 只做轻微扰动，避免完全压过向量相似度。
        # 例：base 0.68 + bonus 4 * 0.015 = 0.74
        hit["_rerank_score"] = base_score + 0.006 * bonus
        hit["_relevance_bonus"] = bonus

        candidates.append(hit)

    candidates.sort(
        key=lambda item: float(item.get("_rerank_score", 0.0)),
        reverse=True,
    )

    if not dedupe_by_doc:
        return candidates

    deduped: list[dict[str, Any]] = []
    seen_doc_ids: set[str] = set()

    for hit in candidates:
        doc_id = hit.get("doc_id") or hit.get("chunk_id")
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        deduped.append(hit)

    return deduped
