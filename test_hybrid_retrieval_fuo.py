import time
from pathlib import Path
import os

from sentence_transformers import SentenceTransformer

from app.config import load_config
from app.hybrid_retrieval import query_hybrid_retrieval


def load_dotenv_simple(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip().strip("'").strip('"')


load_dotenv_simple()

MODEL_PATH = "/AII-heyan/ragtestv01_server_bundle_release/artifacts/model_cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"

QUESTION = "患者持续发热两周，常规抗感染治疗效果不佳，血常规和胸片没有明显异常，应该如何进一步评估不明原因发热？"


def main():
    cfg = load_config()

    print("qdrant_url =", cfg.qdrant_url)
    print("collections =", cfg.get_collection_names())
    print("opensearch_url =", cfg.opensearch_url)
    print("opensearch_index_name =", cfg.opensearch_index_name)

    t0 = time.time()
    model = SentenceTransformer(MODEL_PATH)
    print("model_load_seconds =", round(time.time() - t0, 3), flush=True)

    t1 = time.time()
    vec = model.encode([QUESTION], normalize_embeddings=True)[0].tolist()
    print("embed_seconds =", round(time.time() - t1, 3), flush=True)

    t2 = time.time()
    hits = query_hybrid_retrieval(
        cfg,
        query_text=QUESTION,
        query_vector=vec,
        dense_top_k=30,
        sparse_top_k=30,
        fused_top_k=60,
        final_top_k=10,
    )
    print("hybrid_seconds =", round(time.time() - t2, 4), flush=True)
    print("hits =", len(hits), flush=True)

    for i, h in enumerate(hits, 1):
        print("=" * 80)
        print("#", i)
        print("score =", h.get("score"))
        print("hybrid_score =", h.get("hybrid_score"))
        print("dense_score =", h.get("dense_score"))
        print("sparse_score =", h.get("sparse_score"))
        print("sources =", h.get("_retrieval_sources"))
        print("title =", h.get("title"))
        print("section =", h.get("section_name"))
        print("doc_id =", h.get("doc_id"))
        print("chunk_id =", h.get("chunk_id"))
        print("text =", (h.get("chunk_text") or "")[:500].replace("\n", " "))


if __name__ == "__main__":
    main()