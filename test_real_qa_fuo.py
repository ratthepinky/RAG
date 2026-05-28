import os
import time
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from app.config import load_config
from app.hybrid_retrieval import query_hybrid_retrieval


MODEL_PATH = "/AII-heyan/ragtestv01_server_bundle_release/artifacts/model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

QUESTION = "患者持续发热两周，常规抗感染治疗效果不佳，血常规和胸片没有明显异常，应该如何进一步评估不明原因发热？"

TOP_K = 20
MAX_CONTEXT_CHARS = 9000


def load_dotenv_simple(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        print("WARNING: .env not found", flush=True)
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'").strip('"')


load_dotenv_simple()


def is_low_value_chunk(h):
    title = (h.get("title") or "").strip().lower()
    section = (h.get("section_name") or "").strip().lower()
    text = (h.get("chunk_text") or h.get("text") or h.get("content") or "").strip()
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
    if text_lower.count("doi") >= 2 or text_lower.count(" et al") >= 3:
        return True

    return False


def relevance_bonus(h):
    title = h.get("title") or ""
    section = h.get("section_name") or ""
    text = h.get("chunk_text") or h.get("text") or h.get("content") or ""
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
    ]

    score = 0
    for kw in keywords:
        if kw.lower() in blob:
            score += 1
    return score


def filter_and_rerank_hits(hits):
    candidates = []

    for h in hits:
        if is_low_value_chunk(h):
            continue

        base_score = float(h.get("score", 0.0) or 0.0)
        bonus = relevance_bonus(h)

        h["_rerank_score"] = base_score + 0.015 * bonus
        h["_relevance_bonus"] = bonus
        candidates.append(h)

    candidates.sort(key=lambda x: float(x.get("_rerank_score", 0.0)), reverse=True)

    deduped = []
    seen_doc_ids = set()

    for h in candidates:
        doc_id = h.get("doc_id") or h.get("chunk_id")
        if doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(doc_id)
        deduped.append(h)

    return deduped


def build_context(hits):
    hits = filter_and_rerank_hits(hits)

    parts = []
    total = 0

    for i, h in enumerate(hits, 1):
        text = h.get("chunk_text") or h.get("text") or h.get("content") or ""
        title = h.get("title") or ""
        section = h.get("section_name") or ""
        year = h.get("year") or ""
        journal = h.get("journal") or ""
        score = h.get("score")
        rerank_score = h.get("_rerank_score")
        bonus = h.get("_relevance_bonus")

        block = (
            f"[{i}] score={score}, rerank_score={rerank_score}, bonus={bonus}\n"
            f"title: {title}\n"
            f"journal/year: {journal} {year}\n"
            f"section: {section}\n"
            f"content: {text}\n"
        )

        if total + len(block) > MAX_CONTEXT_CHARS:
            break

        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


def print_hits(title, hits, n=8):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for i, h in enumerate(hits[:n], 1):
        print(f"\n#{i}")
        print("score =", h.get("score"))
        print("rerank_score =", h.get("_rerank_score"))
        print("bonus =", h.get("_relevance_bonus"))
        print("title =", h.get("title"))
        print("section =", h.get("section_name"))
        print("journal =", h.get("journal"))
        print("year =", h.get("year"))
        text = h.get("chunk_text") or ""
        print("text_preview =", text[:500].replace("\n", " "))


def main():
    print("=" * 80)
    print("REAL QA TEST: FUO / 不明原因发热")
    print("question =", QUESTION)
    print("=" * 80, flush=True)

    cfg = load_config()
    print("qdrant_url =", cfg.qdrant_url)
    print("collections =", cfg.get_collection_names(), flush=True)

    t0 = time.time()
    model = SentenceTransformer(MODEL_PATH)
    print("model_load_seconds =", round(time.time() - t0, 3), flush=True)

    t1 = time.time()
    query_vector = model.encode([QUESTION], normalize_embeddings=True)[0].tolist()
    print("embed_seconds =", round(time.time() - t1, 3), flush=True)

    t2 = time.time()
    hits = query_hybrid_retrieval(
        cfg,
        query_text=QUESTION,
        query_vector=query_vector,
        dense_top_k=30,
        sparse_top_k=30,
        fused_top_k=60,
        final_top_k=TOP_K,
    )
    print("hybrid_retrieval_seconds =", round(time.time() - t2, 4), flush=True)
    print("raw_hits =", len(hits), flush=True)

    filtered_hits = filter_and_rerank_hits(hits)
    print("filtered_hits =", len(filtered_hits), flush=True)

    print_hits("TOP RETRIEVED CHUNKS - RAW", hits, n=8)
    print_hits("TOP RETRIEVED CHUNKS - FILTERED/RERANKED", filtered_hits, n=8)

    context = build_context(hits)

    prompt = f"""你是一名严谨、保守、基于检索资料的医学RAG问答助手。请严格优先依据给定检索资料回答。

重要约束：
- 不要编造检索资料中没有的具体阈值、定义、检查项目或治疗方案。
- 如果你的医学常识与检索资料中的定义或表述不同，必须优先采用检索资料，并写成“检索资料中将……定义为……”。
- 如果检索资料只支持一般性建议，不要把建议说成确定诊断或唯一方案。
- 不要直接建议患者自行使用抗生素、激素或抗结核药；如提到诊断性治疗，必须强调应由医生在不影响进一步检查、权衡风险后进行。
- 回答必须包含“需要医生面诊/住院评估”的安全提醒。

用户问题：
{QUESTION}

检索资料：
{context}

请用中文回答，要求：
1. 先基于检索资料说明“不明原因发热/FUO”或长期发热的基本评估思路；
2. 分层列出常见病因方向，并说明哪些来自检索资料；
3. 给出下一步检查与处理建议，按“病史体征—基础检查—进一步检查—随访/转诊”组织；
4. 明确哪些情况需要尽快就医或住院评估；
5. 最后用一小段说明本次检索资料的不足，避免过度确定。
"""

    api_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 MOONSHOT_API_KEY 或 KIMI_API_KEY 环境变量")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )

    t3 = time.time()
    resp = client.chat.completions.create(
        model=os.getenv("KIMI_MODEL", "moonshot-v1-8k"),
        messages=[
            {"role": "system", "content": "你是一个严谨、保守、基于证据的医学RAG问答助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(resp.choices[0].message.content)
    print("\nllm_seconds =", round(time.time() - t3, 3))
    print("total_seconds =", round(time.time() - t0, 3))


if __name__ == "__main__":
    main()
